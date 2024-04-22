import torch
import os
import lpips
import numpy as np
import open3d as o3d
from src.dataloader import get_data_loader
from torch.utils.tensorboard import SummaryWriter
from src.lidarnerf import LiDARNeRF
from .utils import depth_inv_to_color
from .optimizer import Adan
from tqdm import tqdm
from skimage.metrics import structural_similarity
from .sds import SDSLoss
import torch.nn.functional as F
import cv2
from PIL import Image

loss_fn_alex = lpips.LPIPS(net='alex')


def get_exp_name_model(cfg):

    exp_name_model = f"{cfg['num_mlp_feat_layers']}_"
    exp_name_model += f"{cfg['num_mlp_alpha_layers']}_"
    exp_name_model += f"{cfg['num_mlp_rgb_layers']}_"
    exp_name_model += f"{cfg['voxel_feature_dim']}_"
    exp_name_model += f"{cfg['mlp_feat_dim']}_"
    exp_name_model += f"{cfg['nerf_rgb_loss_weight']}_"
    exp_name_model += f"{cfg['nerf_depth_loss_weight']}_"
    exp_name_model += f"{cfg['gan_color_loss_weight']}_"
    exp_name_model += f"{cfg['gan_loss_weight']}_"
    exp_name_model += f"{cfg['depth_loss_error_range']}_"
    exp_name_model += f"{cfg['lr']}"

    return exp_name_model


def get_exp_name_data(cfg):

    return cfg['name_data']


def compute_metrics(rgb_est, rgb_gt):

    # assume input pixel value range with -1~1
    x1 = (rgb_gt.detach().cpu().numpy() + 1)/2
    x2 = (rgb_est.detach().cpu().numpy() + 1)/2

    mask = x1 != 0
    diff = (x1-x2)[mask]

    mse = (diff**2).sum()/mask.sum()
    psnr = -10*np.log10(mse)
    ssim = structural_similarity(x1, x2, channel_axis=-1, data_range=1)

    # normalize to -1~1 for lpips
    x1 = rgb_gt.detach().cpu().permute(2, 0, 1)
    x2 = rgb_est.detach().cpu().permute(2, 0, 1)
    lpips = loss_fn_alex(x1, x2).item()

    return {'psnr': psnr, 'ssim': ssim, 'lpips': lpips, 'mse': mse}


class Trainer:

    def __init__(self, cfg, eval_only=True):

        exp_name_data = get_exp_name_data(cfg)
        exp_name_model = get_exp_name_model(cfg)

        exp_name = f"{exp_name_data}_{exp_name_model}_{cfg['log_id'][:4]}"
        
        path_lidar_map = os.path.join(cfg['path_data_folder'], 'argoverse_2_maps', f"{cfg['log_id']}_{cfg['name_data']}.ply")

        print('Load lidar map from', path_lidar_map)
        map_lidar = np.asarray(o3d.io.read_point_cloud(path_lidar_map).points)

        self.net = LiDARNeRF(cfg, map_lidar).to(cfg['device'])
        e_start = self.net.load_weights(cfg['path_weights'], exp_name, pretrained_path=cfg['path_pretrained_weight'])
        exp_name = f"{exp_name_data}_{exp_name_model}_{cfg['log_id'][:4]}_sds"
        self.log_writer = SummaryWriter(os.path.join(cfg['path_log'], 'logs', exp_name))

        self.e_start = e_start
        self.e_end = int(cfg['num_epoch'])
        self.val_interval = int(cfg['val_interval'])
        self.iter_log_interval = int(cfg['iter_log_interval'])
        self.path_weights = cfg['path_weights']
        self.device = cfg['device']

        self.dataloader_train = get_data_loader(cfg, exp_name_data, 'train')
        self.dataloader_val = get_data_loader(cfg, exp_name_data, 'val')
        self.exp_name = exp_name

        self.use_controlnet = True if cfg["sd_model"]=="controlnet" else False
        print("Using controlnet: ", self.use_controlnet)

        self.sds = SDSLoss(sd_model=cfg["sd_model"], device=self.device)
        # self.init_sds_loss("victorian street very photorealistic")
        self.init_sds_loss("streets covered in snow")

        # for evaluation

        self.list_metric = ['psnr', 'ssim', 'lpips', 'mse']
        self.num_iter = -1
        self.eval_only = eval_only

        # Step 3. Create optimizer and training parameters
        lr = 1e-4
        self.optimizer = Adan(
            self.net.parameters(),
            lr=2*lr,
            eps=1e-8,
            weight_decay=2e-5,
            max_grad_norm=5.0,
            foreach=False,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda iter: 1)
        self.scaler = torch.cuda.amp.GradScaler()
    

    def init_sds_loss(self, prompt):
        print("Generating prompt embeddings...")
        self.prompt_embeds, self.negative_prompt_embeds = self.sds.get_prompt_embeddings(prompt)
        print(self.prompt_embeds.shape, self.negative_prompt_embeds.shape)
    

    def get_sds_loss(self, nerf_img, use_canny=True):
        prompt_embeds = self.prompt_embeds
        negative_prompt_embeds = self.negative_prompt_embeds
        # img_for_canny = nerf_img.clone().detach()
        nerf_img = (nerf_img.permute(2,0,1) + 1)/2
        nerf_img = nerf_img.unsqueeze(0)
        nerf_img = F.interpolate(nerf_img, scale_factor=(2,2))
        img_for_canny = nerf_img.clone().detach()[0]
        log_nerf_img = img_for_canny.permute(1,2,0).cpu().numpy()
        latents = self.sds.encode_imgs(nerf_img)

        # canny edge detection
        if use_canny:
            _, H, W = img_for_canny.shape
            img_for_canny = (img_for_canny.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
            low_threshold = 96
            high_threshold = 163
            canny_img = cv2.Canny(img_for_canny, low_threshold, high_threshold)
            canny_pil = Image.fromarray(canny_img)
            canny_cond = self.sds.prepare_image(
                image = canny_pil,
                width = W,
                height = H,
                batch_size = 1,
                num_images_per_prompt = 1,
                device = self.device,
                dtype = self.sds.precision_t,
                do_classifier_free_guidance = True,
                guess_mode = False
            )
            control_img_embeds = canny_cond
        else:
            control_img_embeds = None
            canny_pil = None
        # grayscale = cv2.cvtColor(img_for_canny, cv2.COLOR)

        loss = self.sds.sds_loss(latents, prompt_embeds, negative_prompt_embeds, control_img_embeds)
        return loss, canny_pil, log_nerf_img

    def run(self):

        # with torch.no_grad():
        #     self.run_val(self.dataloader_val, self.e_start-1)

        if not self.eval_only:
            for e in range(self.e_start, self.e_end):
                print('epoch', e)
                self.num_iter = e * len(self.dataloader_train)
                if e % self.val_interval == 0 and e != self.e_start:
                    with torch.no_grad():
                        self.run_val(self.dataloader_val, e)

                self.run_train(self.dataloader_train)

    def run_train(self, dataloader):
        print('train')
        for i, batch in tqdm(enumerate(dataloader)):
            self.optimizer.zero_grad()
            # only support batch size == 1
            assert len(batch) == 1
            data_dict = batch[0]
            for key in data_dict:
                data_dict[key] = data_dict[key].to(self.device)

            self.num_iter += 1
            output = self.net(data_dict, training=True)

            sds_loss, control_img, log_nerf_img = self.get_sds_loss(output["img_rgb_nerf"], self.use_controlnet)
            depth_loss = output["dict_loss"]["loss_lidar_depth"]
            loss = sds_loss + depth_loss

            # Save Images
            if self.num_iter % self.iter_log_interval == 0:
                for k in output['dict_loss'].keys():
                    self.log_writer.add_scalar(f'train/{k}', output['dict_loss'][k], self.num_iter)
                print(f"Loss iter {i}: sds_loss: {sds_loss.item()}, depth_loss: {depth_loss.item()}")
                log_nerf_img = cv2.resize(log_nerf_img*255, (256,256))
                cv2.imwrite("logs/train_image_logs/nerf_img_curr.png", log_nerf_img[:,:,::-1])
                cv2.imwrite("logs/train_image_logs/gt_img_curr.png", (output["img_rgb_gt"].detach().cpu().numpy()[:,:,::-1]+1)/2*255)
                if control_img is not None:
                    control_img.save("logs/train_image_logs/canny_image.png")


            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()

    def run_val(self, dataloader, e_val):
        print('val')
        dict_metrics_1, dict_metrics_2 = {}, {}
        for m in self.list_metric:
            dict_metrics_1[m] = []
            dict_metrics_2[m] = []

        for i, batch in enumerate(dataloader):

            # only support batch size == 1
            assert len(batch) == 1
            data_dict = batch[0]
            for key in data_dict:
                data_dict[key] = data_dict[key].to(self.device)

            # Forward Pass of Model
            output = self.net(data_dict)

            eval_1 = compute_metrics(output['img_rgb_gt'], output['img_rgb_nerf'])
            # eval_2 = compute_metrics(output['img_rgb_gt'], output['img_rgb_gan'])

            for m in self.list_metric:
                dict_metrics_1[m].append(eval_1[m])
                # dict_metrics_2[m].append(eval_2[m])

            if (i+1) % 5 == 0:  # just to visualize some val results
                for key in output:
                    if 'depth' in key:  # show inverse depth map
                        depth_inv = output[key].cpu().numpy()[:, :, 0]
                        depth_color = depth_inv_to_color(depth_inv)

                        self.log_writer.add_image(f'val_img_{key}/{i}', torch.from_numpy(depth_color).permute(2, 0, 1), self.num_iter)
                    elif 'rgb' in key:
                        mask = output[key].sum(axis=2) == 0
                        # img = (output[key] + 1)/2
                        img = output[key]
                        img[mask] = 0
                        img = img.permute(2, 0, 1).cpu()
                        img = img.cpu()
                        self.log_writer.add_image(f'val_img_{key}/{i}', img, self.num_iter)

        for m in self.list_metric:
            print(m, f'{np.mean(dict_metrics_1[m]):.4f}',
                  f'{np.std(dict_metrics_1[m]):.4f}',
                  f'{np.mean(dict_metrics_2[m]):.4f}',
                  f'{np.std(dict_metrics_2[m]):.4f}')

            self.log_writer.add_scalar(f'{m}/mean_stage_1', np.mean(dict_metrics_1[m]), e_val)
            self.log_writer.add_scalar(f'{m}/std_stage_1', np.std(dict_metrics_1[m]), e_val)
            self.log_writer.add_scalar(f'{m}/mean_stage_2', np.mean(dict_metrics_2[m]), e_val)
            self.log_writer.add_scalar(f'{m}/std_stage_2', np.std(dict_metrics_2[m]), e_val)

        self.net.save_weights(e_val, self.exp_name, self.path_weights, np.mean(dict_metrics_2['psnr']))
