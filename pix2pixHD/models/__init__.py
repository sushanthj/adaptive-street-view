# import os
# import torch
# import sys

# class BaseModel(torch.nn.Module):
#     def name(self):
#         return 'BaseModel'

#     def initialize(self, opt):
#         self.opt = opt
#         self.gpu_ids = opt.gpu_ids
#         self.isTrain = opt.isTrain
#         self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
#         self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

#     def set_input(self, input):
#         self.input = input

#     def forward(self):
#         pass

#     # used in test time, no backprop
#     def test(self):
#         pass

#     def get_image_paths(self):
#         pass

#     def optimize_parameters(self):
#         pass

#     def get_current_visuals(self):
#         return self.input

#     def get_current_errors(self):
#         return {}

#     def save(self, label):
#         pass

#     # helper saving function that can be used by subclasses
#     def save_network(self, network, network_label, epoch_label, gpu_ids):
#         save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
#         save_path = os.path.join(self.save_dir, save_filename)
#         torch.save(network.cpu().state_dict(), save_path)
#         if len(gpu_ids) and torch.cuda.is_available():
#             network.cuda()

#     # helper loading function that can be used by subclasses
#     def load_network(self, network, network_label, epoch_label, save_dir=''):        
#         save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
#         if not save_dir:
#             save_dir = self.save_dir
#         save_path = os.path.join(save_dir, save_filename)        
#         if not os.path.isfile(save_path):
#             print('%s not exists yet!' % save_path)
#             if network_label == 'G':
#                 raise('Generator must exist!')
#         else:
#             #network.load_state_dict(torch.load(save_path))
#             try:
#                 network.load_state_dict(torch.load(save_path))
#             except:   
#                 pretrained_dict = torch.load(save_path)                
#                 model_dict = network.state_dict()
#                 try:
#                     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
#                     network.load_state_dict(pretrained_dict)
#                     if self.opt.verbose:
#                         print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
#                 except:
#                     print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
#                     for k, v in pretrained_dict.items():                      
#                         if v.size() == model_dict[k].size():
#                             model_dict[k] = v

#                     if sys.version_info >= (3,0):
#                         not_initialized = set()
#                     else:
#                         from sets import Set
#                         not_initialized = Set()                    

#                     for k, v in model_dict.items():
#                         if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
#                             not_initialized.add(k.split('.')[0])
                    
#                     print(sorted(not_initialized))
#                     network.load_state_dict(model_dict)                  

#     def update_learning_rate():
#         pass
"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""

import importlib
from pix2pixHD.models.base_model import BaseModel


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "pix2pix.models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
