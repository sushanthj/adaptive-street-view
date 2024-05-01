#!/bin/bash

# ControlNet with Depth, guidance 20, saving images at 7th epoch
# python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="control_seq_1_depth" --text_prompt="streets covered in snow" --guidance_scale=20 --sd_model="controlnet" --controlnet_input="depth" --seq_id="seq1"
# python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="control_seq_2_depth" --text_prompt="streets covered in snow" --guidance_scale=20 --sd_model="controlnet" --controlnet_input="depth" --seq_id="seq2"
# python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="control_seq_3_depth" --text_prompt="streets covered in snow" --guidance_scale=20 --sd_model="controlnet" --controlnet_input="depth" --seq_id="seq3"
# python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="control_seq_4_depth" --text_prompt="streets covered in snow" --guidance_scale=20 --sd_model="controlnet" --controlnet_input="depth" --seq_id="seq4"
# python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="control_seq_5_depth" --text_prompt="streets covered in snow" --guidance_scale=20 --sd_model="controlnet" --controlnet_input="depth" --seq_id="seq5"
# python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="control_seq_6_depth" --text_prompt="streets covered in snow" --guidance_scale=20 --sd_model="controlnet" --controlnet_input="depth" --seq_id="seq6"

# # ControlNet with Canny, guidance 20, saving images at 7th epoch
# python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="control_seq_1_canny" --text_prompt="streets covered in snow" --guidance_scale=20 --sd_model="controlnet" --controlnet_input="canny" --seq_id="seq1"
# python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="control_seq_2_canny" --text_prompt="streets covered in snow" --guidance_scale=20 --sd_model="controlnet" --controlnet_input="canny" --seq_id="seq2"
# python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="control_seq_3_canny" --text_prompt="streets covered in snow" --guidance_scale=20 --sd_model="controlnet" --controlnet_input="canny" --seq_id="seq3"
# python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="control_seq_4_canny" --text_prompt="streets covered in snow" --guidance_scale=20 --sd_model="controlnet" --controlnet_input="canny" --seq_id="seq4"
# python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="control_seq_5_canny" --text_prompt="streets covered in snow" --guidance_scale=20 --sd_model="controlnet" --controlnet_input="canny" --seq_id="seq5"
# python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="control_seq_6_canny" --text_prompt="streets covered in snow" --guidance_scale=20 --sd_model="controlnet" --controlnet_input="canny" --seq_id="seq6"

# Run Without Controlnet on all sequences at guidance 70 for snow prompt
python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="vanilla_seq_ablations_1_guide70" --text_prompt="streets covered in snow" --guidance_scale=70 --sd_model="2.1" --controlnet_input="None" --seq_id="seq1"
python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="vanilla_seq_ablations_2_guide70" --text_prompt="streets covered in snow" --guidance_scale=70 --sd_model="2.1" --controlnet_input="None" --seq_id="seq2"
python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="vanilla_seq_ablations_3_guide70" --text_prompt="streets covered in snow" --guidance_scale=70 --sd_model="2.1" --controlnet_input="None" --seq_id="seq3"
python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="vanilla_seq_ablations_4_guide70" --text_prompt="streets covered in snow" --guidance_scale=70 --sd_model="2.1" --controlnet_input="None" --seq_id="seq4"
python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="vanilla_seq_ablations_5_guide70" --text_prompt="streets covered in snow" --guidance_scale=70 --sd_model="2.1" --controlnet_input="None" --seq_id="seq5"
python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="vanilla_seq_ablations_6_guide70" --text_prompt="streets covered in snow" --guidance_scale=70 --sd_model="2.1" --controlnet_input="None" --seq_id="seq6"

# Run Without Controlnet on seq_2, with varying guidance
python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="vanilla_guide_ablations_g30_seq2" --text_prompt="streets covered in snow" --guidance_scale=30 --sd_model="2.1" --controlnet_input="None" --seq_id="seq2"
python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="vanilla_guide_ablations_g50_seq2" --text_prompt="streets covered in snow" --guidance_scale=50 --sd_model="2.1" --controlnet_input="None" --seq_id="seq2"
python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="vanilla_guide_ablations_g70_seq2" --text_prompt="streets covered in snow" --guidance_scale=70 --sd_model="2.1" --controlnet_input="None" --seq_id="seq2"
python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="vanilla_guide_ablations_g90_seq2" --text_prompt="streets covered in snow" --guidance_scale=90 --sd_model="2.1" --controlnet_input="None" --seq_id="seq2"


# Run Seq_2 on 4 different prompts
python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="vanilla_prompt_ablations_snow" --text_prompt="streets covered in snow, extremely detailed, photorealistic" --guidance_scale=70 --sd_model="2.1" --controlnet_input="None" --seq_id="seq2"
python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="vanilla_prompt_ablations_cement" --text_prompt="streets with cement roads, extremely detailed, photorealistic" --guidance_scale=70 --sd_model="2.1" --controlnet_input="None" --seq_id="seq2"
python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="vanilla_prompt_ablations_apocalypse" --text_prompt="post apocalypse scene, extremely detailed, photorealistic" --guidance_scale=70 --sd_model="2.1" --controlnet_input="None" --seq_id="seq2"
python3 train.py --name_data=clean --name_config=config.ini --exp_name_prefix="vanilla_prompt_ablations_fire" --text_prompt="houses on fire, extremely detailed, photorealistic" --guidance_scale=70 --sd_model="2.1" --controlnet_input="None" --seq_id="seq2"