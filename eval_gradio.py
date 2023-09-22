from share import *
import gradio as gr
import imageio
from functools import partial
from cldm.model import load_state_dict
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torch
from torchvision import transforms
from torch import autocast
import numpy as np
import math
from einops import rearrange
from pathlib import Path
import os, shutil
from omegaconf import OmegaConf
from contextlib import nullcontext
from PIL import Image
from einops import rearrange

# configure
from opt import get_opts
from pytorch_lightning import seed_everything

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


def get_T_from_relative(x, y, z, pose_enc="freq")->torch.Tensor:
    """
    Args:
        x: relative polar degree
        y: relative azimuth degree
        z: relative distance
        
    example:
        (0., -90., 0.): left view
        (0., 90., 0.): right view
        (0., 180., 0.): back view
        (-90., 0., 0.): top view
        (90., 0., 0.): bottom view
    """
    if pose_enc in ["freq","identity"]:
        d_T = torch.tensor([math.radians(x), math.radians(y), z])
    elif pose_enc == "zero":
        d_T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
    else:
        raise NotImplementedError
    return d_T


def look_around_iterator(sep=45):
    current_values = [0, -180., 0]
    while current_values[1] < 180:
        yield tuple(current_values)
        current_values[1] += sep

def load_model(device, _hparams, sd_locked, only_mid_control, cfgs):
    model = instantiate_from_config(cfgs.model)
    model.load_state_dict(load_state_dict(_hparams.resume_path, location='cpu'))
    # reweight noise scheduer
    if _hparams.register_scheduler:
        model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000, linear_start=0.00085, linear_end=0.016)
    model.learning_rate = _hparams.lr
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    # model.control_model.uncond_pose = _hparams.uncond_pose 
    model = model.to(device)   
    model.eval()
    return model

_TITLE = "TOSS"
GRADIO_RES_DIR = "./gradio_res"

def generate_loop_views(
    h, w, precision, n_samples, 
    use_ema_scope, 
    pose_enc, 
    ddim_steps, 
    ddim_eta, 
    model, 
    sampler, 
    cond_im, 
    prompt, 
    out_folder, 
    sep,
    prompt_scale,
    img_ucg, 
    progress=gr.Progress()
):
    # preprocess image
    cond_im = preprocess_image(cond_im)
    cond_im = transforms.ToTensor()(cond_im).unsqueeze(0).to(device)
    # cond_im = cond_im * 2.0 -  1.0
    cond_im = transforms.functional.resize(cond_im, [h, w])
    
     # path for saving results
    if prompt is not None and len(prompt) > 0:
        out_folder = os.path.join(GRADIO_RES_DIR, out_folder)
        if os.path.exists(out_folder):
            shutil.rmtree(out_folder)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

    # generating ...
    assert 360 % sep == 0, "View separation should be divisible with 360"
    progress(0, desc="Starting...")
    for i,(dx,dy,dz) in enumerate(progress.tqdm(look_around_iterator(sep=sep))):
        T = get_T_from_relative(dx, dy, dz, pose_enc)
                            # TODO: integrate prompt into the model inference pipeline @yukai. Need to modify the `sample_model` function
        x_samples_ddim = sample_model(cond_im, model, sampler, precision=precision, scale=prompt_scale, \
                                    n_samples=n_samples, ddim_steps=ddim_steps, ddim_eta=ddim_eta, T=T, h=h, w=w, \
                                    use_ema_scope=use_ema_scope, prompt=prompt, img_ucg=img_ucg)
                            
                            # save results
        assert x_samples_ddim.shape[0] == 1
        x_samples_ddim = x_samples_ddim[0].cpu().numpy()
        x_samples_ddim = 255.0 * rearrange(x_samples_ddim, 'c h w -> h w c')
        save_dir = out_folder
        Image.fromarray(x_samples_ddim.astype(np.uint8)).save(os.path.join(save_dir, f'{i}.png'))
        if i % 10 == 0:
            progress(i, desc="Generating...")
        yield Image.fromarray(x_samples_ddim.astype(np.uint8))
        
def save_gif(save_dir):
    save_dir = os.path.join(GRADIO_RES_DIR, save_dir)
    images = []
    total_views = len(list(Path(save_dir).glob('*.png')))
    for i in range(total_views):
        images.append(imageio.imread(os.path.join(save_dir, f'{i}.png')))
    imageio.mimsave(os.path.join(save_dir, 'look_around.gif'), images, duration=0.1)
    return os.path.join(save_dir, 'look_around.gif')

if __name__ == '__main__':
    hparams = get_opts()
    sd_locked = True
    only_mid_control = False
    model_type = "vae"
    if model_type == "old":
        hparams.model_cfg = "models/toss_old.yaml"
        hparams.resume_path = "ckpt/old/epoch=204.ckpt"
    elif model_type == "vae":
        hparams.model_cfg = "models/toss_vae.yaml"
        hparams.resume_path = "ckpt/vae/epoch=204.ckpt"
        
    h, w = 256, 256
    precision = 'fp32'
    n_samples = 1
    use_ema_scope = True
    pose_enc = hparams.pose_enc
    ddim_steps = 75
    ddim_eta = 1.0

    # set config
    cfgs = OmegaConf.load(hparams.model_cfg)
    # Load model
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = load_model(device, hparams, sd_locked, only_mid_control, cfgs)
    # build model
    sampler = DDIMSampler(model)
                    
    demo = gr.Blocks(title=_TITLE)
    with demo:
        gr.Markdown('# ' + _TITLE)
        
        with gr.Row():
            cond_im, prompt = None, None
            with gr.Column(scale=0.5):
                cond_img = gr.Image(type='pil', image_mode='RGBA', source='upload',
                                    label='Input image of single object')
                
                # prompt
                prompt = gr.Textbox(label='Prompt', interactive=True)
                
                # saving
                out_folder = gr.Textbox(label="Output Folder", interactive=True, placeholder="e.g. ./results")
                
                sep = gr.Slider(0, 180, 10, label="View Separation (should be divisible with 360)",
                                interactive=True)
                
                prompt_scale = gr.Slider(0.0, 10.0, 3.0, label="CFG for Prompt", interactive=True)
                img_ucg = gr.Slider(0.0, 10.0, 0.0, label="CFG for Cond Image", interactive=True)
                
            with gr.Column(scale=0.5):
                # generate views
                generate_button = gr.Button("Generate Views")
                save_button = gr.Button("Save as GIF")
                with gr.Column(scale=0.25):
                    novel_display = gr.Image(type="pil", label="Novel Views")
                with gr.Column(scale=0.25):
                    gif_display = gr.Image(type="filepath", label="GIF")

        generate_loop_views_fn = partial(generate_loop_views, h, w, precision, n_samples, 
                                        use_ema_scope, pose_enc, ddim_steps, ddim_eta, model, sampler)
        generate_button.click(
            fn=generate_loop_views_fn,
            inputs=[cond_img, prompt, out_folder, sep, prompt_scale, img_ucg],
            outputs=novel_display
        )
        
        save_button.click(
            fn=save_gif,
            inputs=[out_folder],
            outputs=gif_display
        )
        
    demo.launch(enable_queue=True, share=True, server_name="192.168.52.143", server_port=8501)