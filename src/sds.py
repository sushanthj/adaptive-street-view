import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionControlNetPipeline, DDPMScheduler, StableDiffusionPipeline, UniPCMultistepScheduler, ControlNetModel


class SDSLoss:

    def __init__(self, sd_model="controlnet", device="cuda:0", t_range=[0.02, 0.98], output_dir="output"):
        if sd_model == "2.1":
            sd_model_key = "stabilityai/stable-diffusion-2-1-base"
        if sd_model == "controlnet":
            controlnet_key = "lllyasviel/sd-controlnet-canny"
            # controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32, use_safetensors=True)
        
        self.H = 512
        self.W = 512
        self.num_inference_steps = 50
        self.output_dir = output_dir
        self.device = device
        self.precision_t = torch.float32

        # create_model
        if sd_model == "2.1":
            sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_key, torch_dtype=self.precision_t).to(self.device)
        elif sd_model == "controlnet":
            controlnet = ControlNetModel.from_pretrained(controlnet_key, torch_dtype=self.precision_t, use_safetensors=True)
            sd_pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=self.precision_t, use_safetensors=True).to(self.device)
            del controlnet
            self.controlnet = sd_pipe.controlnet
            self.prepare_image = sd_pipe.prepare_image
        
        self.vae = sd_pipe.vae
        self.tokenizer = sd_pipe.tokenizer
        self.text_encoder = sd_pipe.text_encoder
        self.encode_prompt = sd_pipe.encode_prompt
        self.unet = sd_pipe.unet

        if sd_model == "2.1":
            self.scheduler = DDIMScheduler.from_pretrained(sd_model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        elif sd_model == "controlnet":
            self.scheduler = UniPCMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler", torch_dtype=self.precision_t)
            # self.scheduler = UniPCMultistepScheduler.from_config(sd_pipe.scheduler.config)
        del sd_pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenient access
    

    @torch.no_grad()
    def get_prompt_embeddings(self, prompt):
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            self.device,
            1,
            True,
            "", # negative prompt
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=(None),
            clip_skip=None,
        )
        return prompt_embeds, negative_prompt_embeds
    
    def encode_imgs(self, img):
        # check the shape of the image should be 512x512
        assert img.shape[-2:] == (512, 512), "Image shape should be 512x512"

        img = 2 * img - 1  # [0, 1] => [-1, 1]

        posterior = self.vae.encode(img).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents
    

    def sds_loss(self, latents, prompt_embeds, negative_prompt_embeds=None, control_img_embeds=None, guidance_scale=20, grad_scale=1, cond_scale=5.0):
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            (latents.shape[0],),
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            noise = torch.randn_like(latents)
            z_t = torch.sqrt(self.alphas[t])*latents + torch.sqrt(1-self.alphas[t])*noise
            z_t = torch.cat([z_t]*2)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            if control_img_embeds is not None:
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    z_t,
                    t,
                    encoder_hidden_states = prompt_embeds,
                    controlnet_cond = control_img_embeds,
                    conditioning_scale = cond_scale,
                    guess_mode = False,
                    return_dict = False
                )
            else:
                down_block_res_samples=None
                mid_block_res_sample=None

            noise_pred = self.unet(
                z_t,
                t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False
            )[0]

            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale*(noise_pred_text - noise_pred_uncond)
            w = 1 - self.alphas[t]
            g = grad_scale*w*(-noise_pred + noise)
            target = latents+g
        
        loss = F.mse_loss(latents, target) / 2.0
        return loss

