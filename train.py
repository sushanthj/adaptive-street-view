import configparser
import argparse
import os
from src.trainer2 import Trainer

 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--log_id',  type=str, required=True)
    parser.add_argument('--path_log',  type=str, default='logs')
    parser.add_argument('--exp_name_prefix',  type=str, required=True)
    parser.add_argument('--text_prompt',  type=str, required=True)
    parser.add_argument('--guidance_scale',  type=float, required=True)
    parser.add_argument('--sd_model',  type=str, required=True) # options = "2.1" or "controlnet"
    parser.add_argument('--controlnet_input', type=str, required=True)
    parser.add_argument('--seq_id',  type=str, default='seq2')
    parser.add_argument('--name_config',  type=str,  required=True)
    parser.add_argument('--name_data', choices=['clean', 'noisy'], type=str, required=True)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()
    print(args)

    cfg = configparser.ConfigParser()
    cfg.read(os.path.join('configs', args.name_config))
    cfg = cfg['DEFAULT']

    data_cfg = configparser.ConfigParser()
    data_cfg.read('configs/data_ids.ini')
    data_cfg = data_cfg['DEFAULT']
    cfg['log_id'] = data_cfg[args.seq_id]
    # cfg['log_id'] = args.log_id
    cfg['path_log'] = args.path_log

    setattr(args, 'log_id', cfg['log_id'])
    cfg['name_data'] = str(args.name_data)
    cfg['text_prompt'] = str(args.text_prompt)
    cfg['guidance_scale'] = str(args.guidance_scale)
    cfg['sd_model'] = str(args.sd_model)
    cfg['controlnet_input'] = str(args.controlnet_input)
    cfg['exp_name_prefix'] = str(args.exp_name_prefix)
    print(cfg)

    trainer = Trainer(cfg, eval_only=args.eval_only)
    trainer.run()
