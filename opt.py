import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    # common args for all datasets
    parser.add_argument('--root_dir', type=str, default="/comp_robot/shiyukai/dataset/nerf/Synthetic_NeRF/Chair/",
                        help='root directory of dataset')
    parser.add_argument('--eval_root_dir', type=str, default='/comp_robot/mm_generative/data/GSO/views',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='nsvf', help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument('--model_cfg', type=str, default='./models/cldm_pose_v15.yaml',
                        help='cfg path of model')
    parser.add_argument('--model_low_cfg', type=str, default='./models/cldm_pose_v15.yaml',
                        help='cfg path of low-level model')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')
    parser.add_argument('--resume_path', type=str, default='./models/control_sd15_pose_ini.ckpt',
                        help='resume path')
    parser.add_argument('--resume_path_low', type=str, default='./models/control_sd15_pose_ini.ckpt',
                        help='resume path for low-level model')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='train from resume')
    parser.add_argument('--text', type=str, default="a yellow lego bulldozer sitting on top of a table",
                        help='text prompt')
    parser.add_argument('--uncond_pose', type=bool, default=False,
                        help='set delta pose zero')
    parser.add_argument('--img_size', type=int, default=512,
                        help='size of img')
    parser.add_argument('--acc_grad', type=int, default=None,
                        help='accumulate grad')
    parser.add_argument('--eval_guidance_scale', type=float, default=1,
                        help='guidance scale for eval')  
    parser.add_argument('--eval_guidance_scale2', type=float, default=1,
                        help='guidance scale for eval')  
    parser.add_argument('--eval_guidance_scale_low', type=float, default=1,
                        help='guidance scale for eval')     
    parser.add_argument('--eval_use_ema_scope', action="store_true",
                        help='ema_scop for eval')     
    parser.add_argument('--eval_caption', type=str, default="origin",
                        help='caption mode for eval')               
    parser.add_argument('--inf_img_path', type=str, default="./exp/inference/img/008.png",
                        help='caption mode for eval') 
    parser.add_argument('--test_sub', action='store_true', default=False,
                        help='test on part of eval') 
    parser.add_argument('--divide_steps', type=int, default=800,
                        help='divide steps for stage model') 
    parser.add_argument('--attn_t', type=int, default=800,
                        help='timesteps in viz attn') 
    parser.add_argument('--layer_name', type=str, default="",
                        help='timesteps in viz attn') 
    parser.add_argument('--output_mode_attn', type=str, default="masked",
                        help='timesteps in viz attn') 
    parser.add_argument('--img_ucg', type=float, default=0.0,
                        help='ucg for img') 
    parser.add_argument('--register_scheduler', action='store_true', default=False,
                        help='whether to register noise scheduler')
    parser.add_argument('--pose_enc', type=str, default="freq",
                        help='encoding for camera pose') 
                        

    # training options
    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of rays in a batch')
    parser.add_argument('--log_interval_epoch', type=int, default=1,
                        help='interval of logging info')
    parser.add_argument('--ckpt_interval', type=int, default=10,
                        help='interval of ckpt')
    parser.add_argument('--logger_freq', type=int, default=20,
                        help='logger_freq')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics (experimental)')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real dataset only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')
    parser.add_argument("--eval_image", type=str, default="", help="path to eval image")
    parser.add_argument("--eval_prompt", type=str, default="", help="prompt for eval image")

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')

    # modified model paras
    parser.add_argument("--fuse_fn", type=str, default="trilinear_interp",
                        help='fuse function for codebook')
    parser.add_argument("--deformable_hash", type=str, default="no_deformable",
                        help='use deformable hash or not, deformable_codebook / deformable_sample')
    parser.add_argument("--deformable_hash_speedup", type=str, default="no_speedup",
                        help='use deformable hash speedup or not, no_speedup / sampling / clustering')
    parser.add_argument("--n_levels", type=int, default=16, help='n_levels of codebook')
    parser.add_argument("--finest_res", type=int, default=1024, help='finest resolultion for hashed embedding')
    parser.add_argument("--base_res", type=int, default=16, help='base resolultion for hashed embedding')
    parser.add_argument("--n_features_per_level", type=int, default=2, help='n_features_per_level')
    parser.add_argument("--log2_hashmap_size", type=int, default=19, help='log2 of hashmap size')
    parser.add_argument("--max_samples", type=int, default=1024, help='max sample points in a ray')
    parser.add_argument('--offset_mode', action='store_true', default=False,
                        help='use offset in codebook or not')
    parser.add_argument('--record_offset', action='store_true', default=False,
                        help='record offset in codebook or not')
    parser.add_argument('--update_interval', type=int, default=16,
                        help='update interval for density map')
    parser.add_argument('--deformable_lr', type=float,
                        help='learning rate of offset')
    parser.add_argument('--multi_scale_lr', type=float,
                        help='adaptive learning rate of multi scale offset')
    parser.add_argument('--mlp_lr', type=float,
                        help='adaptive learning rate of mlp')
    parser.add_argument('--feature_lr', type=float, default=0.01, 
                        help='learning rate of feature')
    parser.add_argument('--position_loss', type=float,
                        help='apply l2 loss on position of codebook points or not')
    parser.add_argument('--var_loss', type=float,
                        help='apply loss on variance of codebook points position or not')
    parser.add_argument('--offset_loss', type=float,
                        help='apply l2 loss on offsets of codebook points or not')
    parser.add_argument('--warmup', type=int, default=256, 
                        help='warmup steps')
    parser.add_argument('--check_codebook', type=int,  
                        help='check codebook steps')
    parser.add_argument('--reinit_start', type=int, default=1000, 
                        help='start reinit for codebook check')
    parser.add_argument('--reinit_end', type=int, default=2000, 
                        help='end reinit for codebook check')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='threshold for codebook check')
    parser.add_argument('--noise', type=float, default=0., 
                        help='random noise for codebook check')
    parser.add_argument('--degree', type=float, default=1., 
                        help='degree for distance inverse interpolation')
    parser.add_argument('--limit_func', type=str, default="sigmoid", 
                        help='function on limit of offset')
    parser.add_argument('--table_size', type=int, default=5, 
                        help='table_size for grid')
    parser.add_argument('--offset_grid', type=int, default=2, 
                        help='area allowed to offset')
    parser.add_argument('--grid_hashmap_size', type=int, default=19, 
                        help='hashmap_size for grid')
    parser.add_argument('--warmup_epochs', type=float, default=0, 
                        help='warmup epochs for lr scheduler')

    # deformable sample
    parser.add_argument('--multi_offset', type=int, default=8, 
                        help='num of deformable offsets for each sample')
    


    return parser.parse_args()
