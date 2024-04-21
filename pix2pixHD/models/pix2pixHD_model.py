import numpy as np
import torch
import os
from torch.autograd import Variable

from .base_model import BaseModel
from . import networks


class Pix2PixHD(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        if self.isTrain:
            self.model_names = ['G', 'D', 'Enc']
        else:
            # self.model_names = ['G', 'Enc']
            self.model_names = ['G']
        
        # assume input_nc is always 3
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global,
                                      opt.n_local_enhancers, opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = opt.input_nc + opt.output_nc
            # if not opt.no_instance:
            #     netD_input_nc+=1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
                                          use_sigmoid, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        
        
        # ignore encoder network for now
        # if self.gen_features:
        #     raise NotImplementedError("Define an encoder network. Not defined!")
        

        # TODO: Load networks for inference / resuming from checkpoints
        # Check if can be done from nerf model

        if self.isTrain:
            self.old_lr = opt.lr

            self.criterionGAN = networks.GANLoss(use_lsgan = not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss() # might not be be needed if self.gen_features is False
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # initialize optimizers
            if opt.niter_fix_global > 0:
                finetune_list = set()
                params_dict = dict(self.netG.named_parameters())
                paramsG = []
                for k,v in params_dict.items():
                    if k.startswith('model'+str(opt.n_local_enhancers)):
                        paramsG+=[v]
                        finetune_list.add(k.split('.')[0])
                print(f"Only training local enhancer network for {opt.niter_fix_global} epochs")
                print("Finetuned layers are \n", sorted(finetune_list))
            else:
                paramsG = list(self.netG.parameters())
            
            self.optimizer_G = torch.optim.Adam(paramsG, lr=opt.lr, betas=(opt.beta1, 0.999))

            paramsD = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(paramsD, lr=opt.lr, betas=(opt.beta1, 0.999))
    

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        if input['B'] is None:  # only supports AtoB
            self.real_B = None
        else:
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
    

    def forward(self, training=True):
        fake_image = self.netG.forward(self.real_A)
        if self.use_mask:
            self.fake_B = torch.zeros_like(fake_image)
            self.fake_B[:,:,self.mask]  = fake_image[:,:,self.mask]
        else:
            self.fake_B = fake_image
    
    def backward_D(self):
        # fake detection
        fake_AB = torch.cat((self.real_A, self.fake_B[:,0:3].detach()), 1)
        fake_AB[:,:,self.dyn_mask] = 0
        pred_fake = self.netD.forward(fake_AB)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # real
        if self.real_B is not None:
            real_AB = torch.cat((self.real_A, self.real_B[:,0:3]), 1)
            real_AB[:,:,self.dyn_mask] = 0
            pred_real = self.netD.forward(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True)
        else:
            self.loss_D_real = 0
        
        self.loss_D = self.gan_loss_weight * (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward(retain_graph=True)
    
    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_VGG = self.criterionVGG(self.fake_B[:,:3,...], self.real_B[:,:3,...])
        self.loss_G = self.loss_G + self.opt.lambda_feat*self.loss_G_VGG + self.gan_loss_weight*self.loss_G_GAN
        self.loss_G.backward(retain_graph=True)
    
    def optimize_parameters(self, map_feat, do_forward=True):
        if do_forward:
            self.forward()
        
        if self.gan_loss_weight != 0:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()

            torch.nn.utils.clip_grad_norm_(self.netD.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(map_feat, 1)

            self.optimizer_D.step()

            self.set_requires_grad(self.netD, False)
        
        self.optimizer_G.zero_grad()
        self.backward_G()
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 1)
        self.optimizer_G.step()
