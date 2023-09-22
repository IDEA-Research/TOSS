# debug
# CUDA_VISIBLE_DEVICES=2 python pose_train.py --exp_name debug --dataset_name objaverse_car \
# --model_cfg models/debug.yaml \
# --resume_path /comp_robot/shiyukai/ControlNet/models/miniSD.ckpt \
# --num_gpus 1 --num_epochs 2000 --lr 1e-4 --logger_freq 20 --ckpt_interval 50 --acc_grad 1


# old
# CUDA_VISIBLE_DEVICES=0 python pose_train.py --exp_name debug --dataset_name objaverse_car \
#     --model_cfg models/toss_old.yaml \
#     --resume_path ckpt/old/epoch=204.ckpt \
#     --num_gpus 1 --num_epochs 2000 --lr 1e-4 --logger_freq 20 --ckpt_interval 50 --acc_grad 1

# vae
CUDA_VISIBLE_DEVICES=0 python pose_train.py \
    --exp_name debug --dataset_name objaverse_car \
    --model_cfg models/toss_vae.yaml \
    --resume_path ckpt/vae/epoch=204.ckpt \
    --num_gpus 1 --num_epochs 2000 --lr 1e-4 --logger_freq 20 --ckpt_interval 50 --acc_grad 1
