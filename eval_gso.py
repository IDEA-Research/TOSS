from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from ldm.util import instantiate_from_config
import torch
from omegaconf import OmegaConf

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import argparse
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch import autocast
import numpy as np
import math
from einops import rearrange
from pathlib import Path
import json
import os, sys, random, shutil
from omegaconf import DictConfig, ListConfig, OmegaConf
import matplotlib.pyplot as plt
from contextlib import nullcontext
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.lpip_similarity import LPIPS
import glob
from skimage.io import imread
from einops import rearrange
import random

# configure
from opt import get_opts
from pytorch_lightning import seed_everything
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pixelnerf_src")))

seed_everything(40)
device_idx = 0
device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')

def load_model_from_config(config, ckpt, device, verbose=True):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, T, use_ema_scope=False, prompt=None, img_ucg=0.):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    ema_scope = model.ema_scope if use_ema_scope else nullcontext
    with precision_scope('cuda'):
        with ema_scope('Sampling...'):
            # hint
            c_cat = input_im
            # text
            uc_cross = model.get_unconditional_conditioning(n_samples)
            c = model.get_learned_conditioning(prompt) if scale > 1.0 else uc_cross
            # camera pose
            delta_pose = T[None, :].repeat(n_samples, 1).to(c.device)
            # concat for concat pipline
            in_concat = model.encode_first_stage(((input_im*2-1).to(c.device))).mode().detach()

            cond = {}
            cond['delta_pose'] = delta_pose
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [c_cat]
            cond['in_concat'] = [in_concat]

            # uc
            uc = {}
            uc['delta_pose'] = delta_pose
            uc['c_crossattn'] = [uc_cross]
            uc['c_concat'] = [c_cat]
            uc['in_concat'] = [in_concat]
            if img_ucg > 0.:
                uc['in_concat'] = [in_concat*0]

            shape = [4, h // 8, w // 8]
            x_T = torch.randn(in_concat.shape, device=c.device)
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=x_T)
            print(samples_ddim.shape)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
        
def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model
        
def get_model(config, device, ckpt='105000.ckpt'):
    config = OmegaConf.load(config)
    print('Instantiating LatentDiffusion...')
    model = load_model_from_config(config, ckpt, device)
    print('Done.')
    return model

def preprocess_image(input_im):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
    input_im = np.asarray(input_im, dtype=np.float32) / 255.0
    # (H, W, 4) array in [0, 1].

    # old method: thresholding background, very important
    # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

    # new method: apply correct method of compositing to avoid sudden transitions / thresholding
    # (smoothly transition foreground to white background based on alpha values)
    if input_im.shape[-1] == 4:
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

    input_im = input_im[:, :, 0:3]
    # (H, W, 3) array in [0, 1].

    return input_im

# %%
def cartesian_to_spherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    z = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.arctan2(xyz[:,1], xyz[:,0])
    return np.array([theta, azimuth, z])

def get_T(target_RT, cond_RT, pose_enc="freq"):
    R, T = target_RT[:3, :3], target_RT[:, -1]
    T_target = -R.T @ T

    R, T = cond_RT[:3, :3], cond_RT[:, -1]
    T_cond = -R.T @ T

    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])
    
    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond
    
    if pose_enc == "freq":
        d_T = torch.tensor([d_theta.item(), d_azimuth.item(), d_z.item()])
    elif pose_enc == "identity":
        d_T = torch.tensor([d_theta.item(), d_azimuth.item(), d_z.item()])
    elif pose_enc == "zero":
        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])

    return d_T


if __name__ == '__main__':
    # Configs
    hparams = get_opts()
    logger_freq = hparams.logger_freq
    sd_locked = True
    only_mid_control = False
    cfgs = OmegaConf.load(hparams.model_cfg)

    # hparams for eval
    root_dir = hparams.eval_root_dir
    uids = os.listdir(root_dir)
    num_views_per_scene = 16
    h, w = hparams.img_size, hparams.img_size
    precision = 'fp32'
    scale = hparams.eval_guidance_scale
    n_samples = hparams.batch_size
    use_ema_scope = hparams.eval_use_ema_scope
    eval_caption = hparams.eval_caption
    img_ucg = hparams.img_ucg
    pose_enc = hparams.pose_enc
    ddim_steps = 75
    ddim_eta = 1.0

    # TODO: complete the caption path
    if eval_caption == "random":
        caption_path = "/comp_robot/mm_generative/data/GSO/random_views_17views.json"
    elif eval_caption == "rerank":
        caption_path = "/comp_robot/mm_generative/data/GSO/valid_paths_clip_recap.json"
    else:
        caption_path = "/comp_robot/mm_generative/data/.objaverse/hf-objaverse-v1/cars-vehicles/valid_paths_car_vehicles_eval.json"
    
    # for implausible samples
    # caption_path = "/comp_robot/mm_generative/data/implausible_data/implausible_caption.json"

    assert os.path.exists(caption_path), f'caption path {caption_path} does not exist'
    caption_map = json.load(open(caption_path, 'r'))

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = instantiate_from_config(cfgs.model)
    model.load_state_dict(load_state_dict(hparams.resume_path, location='cpu'))
    # model = load_model_from_config(config=cfgs, ckpt=hparams.resume_path, device=device)
    # reweight noise scheduer
    if hparams.register_scheduler:
        model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000, linear_start=0.00085, linear_end=0.016)
    model.learning_rate = hparams.lr
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    # model.control_model.uncond_pose = hparams.uncond_pose 
    model = model.to(device)   
    model.eval()

    # path
    if "debug" in hparams.exp_name:
        out_folder = f'./exp/{hparams.dataset_name}/{hparams.exp_name}/eval/{os.path.splitext(os.path.split(hparams.resume_path)[-1])[0]}/guidance{scale}'
    else:
        out_folder = f"/comp_robot/shiyukai/ControlNet/exp/{hparams.dataset_name}/{hparams.exp_name}/eval/{os.path.splitext(os.path.split(hparams.resume_path)[-1])[0]}/guidance{scale}"  
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # build model
    sampler = DDIMSampler(model)

    for uid in uids:
        print(f'Processing {uid}...')
        filename = os.path.join(root_dir, uid)
        cond_im = Image.open(os.path.join(filename, '000.png'))
        cond_im = preprocess_image(cond_im)
        cond_im = transforms.ToTensor()(cond_im).unsqueeze(0).to(device)
        # cond_im = cond_im * 2.0 - 1.0
        cond_im = transforms.functional.resize(cond_im, [h, w])
        
        cond_RT = np.load(os.path.join(filename, '000.npy'))
        prompt = caption_map[uid]
        if isinstance(prompt, str):
            prompt = prompt
        elif isinstance(prompt, dict):
            # randomly sample one caption
            prompt = random.choice(list(prompt.values()))
        for idx in tqdm(range(0, num_views_per_scene+1)):
            target_RT = np.load(os.path.join(filename, f'{idx:03d}.npy'))
            T = get_T(target_RT, cond_RT, pose_enc)
            # TODO: integrate prompt into the model inference pipeline @yukai. Need to modify the `sample_model` function
            x_samples_ddim = sample_model(cond_im, model, sampler, precision=precision, scale=scale, \
                    n_samples=n_samples, ddim_steps=ddim_steps, ddim_eta=ddim_eta, T=T, h=h, w=w, \
                    use_ema_scope=use_ema_scope, prompt=prompt, img_ucg=img_ucg)
            
            # save results
            assert x_samples_ddim.shape[0] == 1
            x_samples_ddim = x_samples_ddim[0].cpu().numpy()
            x_samples_ddim = 255.0 * rearrange(x_samples_ddim, 'c h w -> h w c')
            save_dir = os.path.join(out_folder, uid)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            Image.fromarray(x_samples_ddim.astype(np.uint8)).save(os.path.join(save_dir, f'{idx:03d}.png'))
            
    # %%
    # glob all images from the processed folder
    images_gt = glob.glob('/comp_robot/mm_generative/data/GSO/views_eval/*/*.png')
    images_gt = sorted(images_gt)
    images_pred = glob.glob(os.path.join(out_folder, '*/*.png'))
    images_pred = sorted(images_pred)

    images_gt_arr = np.stack([imread(img) for img in images_gt])
    images_pred_arr = np.stack([imread(img) for img in images_pred])

    images_gt_arr = rearrange(images_gt_arr, 'b h w c -> b c h w') 
    images_pred_arr = rearrange(images_pred_arr, 'b h w c -> b c h w')

    # PSNR, SSIM, LPIPS
    psnr_score = psnr(images_gt_arr, images_pred_arr, data_range=255)
    print('PSNR:', psnr_score)
    ssim_score = ssim(images_gt_arr, images_pred_arr, data_range=255, channel_axis=1)
    print('SSIM', ssim_score)

    image_gt_tensor = torch.from_numpy(images_gt_arr) / 255.0 * 2 - 1
    image_pred_tensor = torch.from_numpy(images_pred_arr) / 255.0 * 2 - 1
    lpips = LPIPS(net_type='vgg')
    lpip_score = lpips(image_gt_tensor, image_pred_tensor)
    print('LPIPS', lpip_score.item())

    # FID
    import subprocess
    extra_command = f' --kid-subset-size {len(images_gt)}' if len(images_gt) < 1000 else ''
    command = f"fidelity --gpu 0 --fid --kid --input1 /comp_robot/mm_generative/data/GSO/views_eval/ --input2 {out_folder} --samples-find-deep" + extra_command

    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    print(output)