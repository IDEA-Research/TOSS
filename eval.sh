# old
# python eval_gso.py --exp_name debug --dataset_name gso \
# --model_cfg /comp_robot/shiyukai/TOSS/models/toss_old.yaml \
# --resume_path /comp_robot/wangjianan/ControlNet/exp/objaverse800k/pose_obj800k_finetune_eps_cross_attn_temp_ucg0.5_img_ucg0.05_cap3d_rectify_fp16_res256_batch256/ckpt/epoch=204.ckpt \
# --num_gpus 1 --batch_size 1 --img_size 256 --eval_guidance_scale 3 --eval_caption rerank --eval_use_ema_scope

python eval_cli.py --exp_name debug --dataset_name gso \
    --eval_image assets/kunkun.png \
    --eval_prompt "A toy figurine wearing overalls and holding a basketball"\
    --model_cfg models/toss_old.yaml \
    --resume_path ckpt/old/epoch=204.ckpt \
    --num_gpus 1 --batch_size 1 --img_size 256 --eval_guidance_scale 3 --eval_caption rerank --eval_use_ema_scope


# A figurine wearing overalls and holding a basketball
# A dark silver sports car
# Guardians of the Galaxy Galactic Battlers Rocket Raccoon



# vae
# python eval_cli.py --exp_name debug --dataset_name gso \
#     --eval_image assets/kunkun.png \
#     --eval_prompt "A toy figurine wearing overalls and holding a basketball"\
#     --model_cfg models/toss_vae.yaml \
#     --resume_path ckpt/vae/epoch=204.ckpt \
#     --num_gpus 1 --batch_size 1 --img_size 256 --eval_guidance_scale 3 --eval_caption rerank --eval_use_ema_scope

# python eval_cli.py --exp_name debug --dataset_name gso \
#     --eval_image assets/rocket.png \
#     --eval_prompt "Guardians of the Galaxy Galactic Battlers Rocket Raccoon"\
#     --model_cfg models/toss_vae.yaml \
#     --resume_path ckpt/vae/epoch=204.ckpt \
#     --num_gpus 1 --batch_size 1 --img_size 256 --eval_guidance_scale 3 --eval_caption rerank --eval_use_ema_scope

# python eval_cli.py --exp_name debug --dataset_name gso \
#     --eval_image assets/car.png \
#     --eval_prompt "A dark silver sports car"\
#     --model_cfg models/toss_vae.yaml \
#     --resume_path ckpt/vae/epoch=204.ckpt \
#     --num_gpus 1 --batch_size 1 --img_size 256 --eval_guidance_scale 3 --eval_caption rerank --eval_use_ema_scope