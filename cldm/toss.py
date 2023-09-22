import einops
import torch
import torch as th
import torch.nn as nn
from contextlib import nullcontext
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer, SpatialTransformer_gate
from ldm.modules.diffusionmodules.openaimodel import UNetModel, UNetModel_pose, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# viz
from viz import save_image_tensor2cv2
import os


class UNetModel(UNetModel_pose):
    def forward(self, x, timesteps=None, context=None, y=None, delta_pose=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        
        # concat
        if "CA" in self.temp_attn: 
            emb = torch.cat([emb, emb], dim=0)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        # camera pose embed
        if (self.pose_enc == "freq") or (self.pose_enc == "vae"):
            dir_embed = self.dir_encoder(delta_pose[:,0:2])
            pos_embed = self.z_encoder(delta_pose[:,-1:])
            pos_emb = self.pose_net(torch.cat([dir_embed, pos_embed], dim=-1)).unsqueeze(1)
        elif (self.pose_enc == "identity") or (self.pose_enc == "zero"):
            pos_emb = self.pose_net(delta_pose).unsqueeze(1)
        else:
            pos_emb = None

        # module
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context, pos_emb)
            hs.append(h)
        h = self.middle_block(h, emb, context, pos_emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, pos_emb)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)[:x.shape[0]//2] if "CA" in self.temp_attn else self.id_predictor(h)
        else:
            return self.out(h)[:x.shape[0]//2] if "CA" in self.temp_attn else self.out(h)



class TOSS(LatentDiffusion):
    def __init__(self, control_key, only_mid_control, ucg_123=0., loss_weight="eps", \
                max_timesteps=1000, min_timesteps=0, finetune=False, scheduler_config=None, \
                img_ucg=0., half_sample=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.ucg_123 = ucg_123
        self.loss_weight = loss_weight
        print("Using loss weight: ", self.loss_weight)

        self.max_timesteps = max_timesteps
        self.min_timesteps = min_timesteps

        # finetune trunk or not
        self.finetune = finetune
        self.img_ucg = img_ucg
        self.count = 0
        self.half_sample = half_sample

        # load img embedding
        # model = instantiate_from_config(cond_stage_img_config)
        # self.cond_stage_model_img = model.eval()
        # for param in self.cond_stage_model_img.parameters():
        #     param.requires_grad = False

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, training=True, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        # camera pose
        delta_pose = batch['delta_pose']
        delta_pose = delta_pose.to(self.device)
        if bs is not None:
            delta_pose = delta_pose[:bs]
        # hint
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)       
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        # encode
        concat = self.encode_first_stage(((control*2-1).to(self.device))).mode().detach() # rectify scale

        # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if self.training and self.ucg_123>0:
            uncond = self.ucg_123
            random = torch.rand(x.size(0), device=x.device)
            prompt_mask = rearrange(random < uncond, "n -> n 1 1")

            # z.shape: [8, 4, 64, 64]; c.shape: [8, 1, 768]
            with torch.enable_grad():
                null_prompt = self.get_learned_conditioning([""]).detach()
                c = torch.where(prompt_mask, null_prompt, c)

            # ucg for input img
            if self.img_ucg > 0:
                uncond = self.img_ucg
                random = torch.rand(x.size(0), device=x.device)
                input_mask = 1 - rearrange((random >= self.ucg_123 - uncond).float() * (random < self.ucg_123 + uncond).float(), "n -> n 1 1 1")
                concat = input_mask * concat

        return x, dict(c_crossattn=[c], c_concat=[control], delta_pose=delta_pose, in_concat=[concat])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        # input img emb as cross attn
        # hint = torch.cat(cond['c_concat'], 1)*2-1
        # clip_emb = self.get_learned_img_conditioning(hint).detach()

        if self.half_sample:
            null_prompt = self.get_learned_conditioning([""]).detach().repeat(x_noisy.shape[0],1,1)
            eps = diffusion_model(x=torch.cat([x_noisy] + cond['in_concat'], dim=0), \
            timesteps=t, context=cond_txt, delta_pose=cond['delta_pose'], null_prompt=null_prompt)
        else:
            eps = diffusion_model(x=torch.cat([x_noisy] + cond['in_concat'], dim=0), \
                timesteps=t, context=cond_txt, delta_pose=cond['delta_pose'])

        return eps

    # @torch.no_grad()
    # def get_learned_img_conditioning(self, c):
    #     if self.cond_stage_forward is None:
    #         if hasattr(self.cond_stage_model_img, 'encode') and callable(self.cond_stage_model_img.encode):
    #             c = self.cond_stage_model_img.encode(c)
    #             if isinstance(c, DiagonalGaussianDistribution):
    #                 c = c.mode()
    #         else:
    #             c = self.cond_stage_model_img(c)
    #     else:
    #         assert hasattr(self.cond_stage_model_img, self.cond_stage_forward)
    #         c = getattr(self.cond_stage_model_img, self.cond_stage_forward)(c)
    #     return c

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=True, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=True, plot_progressive_rows=False,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N, training=False)
        delta_p = c['delta_pose'][:N,...]
        in_concat = c["in_concat"][0][:N]
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid
        
        # ddim sample
        ddim_steps = 50
        x_T = None
        # noise = torch.randn_like(z)
        # x_T = self.q_sample(x_start=z, t=(torch.full(fill_value=999, size=(z.shape[0],))).to(self.device).long(), noise=noise)

        if sample:
            # get denoise row
            uc_cat = c_cat
            uc_cross = self.get_unconditional_conditioning(N)
            if self.img_ucg > 0.:
                uc_inconcat = torch.zeros_like(in_concat).to(self.device)
            else:
                uc_inconcat = in_concat
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross], "delta_pose":delta_p, "in_concat":[uc_inconcat]}
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [uc_cross], "delta_pose":delta_p, "in_concat":[in_concat]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             x_T=x_T
                                             )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            # plot denoise
            if plot_denoise_rows:
                # denoise_grid = self._get_denoise_row_from_list(z_denoise_row['x_inter'])
                # log["ddim_cfg_denoise_inter"] = denoise_grid
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row['pred_x0'])
                log["ddim_ucg_denoise_pred"] = denoise_grid

        # ddim ucg sampling
        if unconditional_guidance_scale >= 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            if self.img_ucg > 0.:
                uc_inconcat = torch.zeros_like(in_concat).to(self.device)
            else:
                uc_inconcat = in_concat
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross], "delta_pose":delta_p, "in_concat":[uc_inconcat]}
            # uc_full = None
            samples_cfg, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], "delta_pose":delta_p, "in_concat":[in_concat]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             x_T=x_T
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            # plot denoise
            if plot_denoise_rows:
                # denoise_grid = self._get_denoise_row_from_list(z_denoise_row['x_inter'])
                # log["ddim_denoise_inter"] = denoise_grid
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row['pred_x0'])
                log["ddim_denoise_pred"] = denoise_grid

        # ddpm sampling
        if plot_progressive_rows:
            with ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(cond={"c_concat": [c_cat], "c_crossattn": [c], "delta_pose":delta_p, "in_concat":[in_concat]},
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            b, c, h, w = cond["c_concat"][0].shape
            shape = (self.channels, h // 8, w // 8)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, \
                                                log_every_t=10, noise_dropout=0.0, verbose=False, **kwargs)
        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())

        # get cross attn
        model_params = []
        cross_attn_params = []
        pose_net = []
        for n, m in self.model.named_parameters(): 
            if 'attn1' in n or 'attn_mid' in n:
                cross_attn_params.append(m)
                print(n)
            elif 'pose_net' in n:
                pose_net.append(m)
                print(n)
            else:
                model_params.append(m)
        
        if self.finetune:
            opt = torch.optim.AdamW([{"params": model_params, "lr": lr},
                        {"params": cross_attn_params, "lr": lr},
                        {"params": pose_net, "lr": lr},], lr=lr)
        else:
            opt = torch.optim.AdamW([{"params": model_params, "lr": lr},
                            {"params": cross_attn_params, "lr": lr*2},
                            {"params": pose_net, "lr": lr*10},], lr=lr)
        
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler

        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
    
    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(self.min_timesteps, self.max_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # sup on x0
        # pred_x0 = self.predict_start_from_noise(x_noisy, t=t, noise=model_output)
        # loss_x0 = self.get_loss(pred_x0, x_start, mean=False).mean([1, 2, 3])

        # sup on noise
        loss_eps = self.get_loss(model_output, noise, mean=False).mean([1, 2, 3])

        # weight loss
        if self.loss_weight == 'eps':
            loss_simple = loss_eps
        elif self.loss_weight == 'x0':
            loss_simple = loss_eps*(1-self.alphas_cumprod[t])/self.alphas_cumprod[t]
        elif self.loss_weight == 'mix':
            pred_x0 = self.predict_start_from_noise(x_noisy, t=t, noise=model_output)
            loss_x0 = self.get_loss(pred_x0, x_start, mean=False).mean([1, 2, 3])
            loss_simple = loss_eps + loss_x0
        elif self.loss_weight == 'sds':
            loss_simple = loss_eps*(1-self.alphas_cumprod[t])
        elif self.loss_weight == 'recip_eps':
            loss_simple = loss_eps/self.alphas_cumprod[t]

        # average
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        # higher weight for high timestep
        steps = self.lvlb_weights.shape[0]-1
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        # self.timestep_weight = self.timestep_weight.to(self.device)
        # loss_vlb = (self.timestep_weight[t] * loss_vlb).mean()

        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        # viz
        # save_path = "/home/shiyukai/ControlNet/exp/viz/res512"
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        # target_viz = self.decode_first_stage(x_start)*0.5+0.5
        # noise_viz = self.decode_first_stage(x_noisy)*0.5+0.5
        # cond_viz = (torch.cat(cond['c_concat'], 1))
        # pred_x0 = self.predict_start_from_noise(x_noisy, t=t, noise=model_output)
        # pred_viz = self.decode_first_stage(pred_x0)*0.5+0.5
        # for i in range(t.shape[0]):
        #     save_path_t = f'{save_path}/{t[i]}/'
        #     if not os.path.exists(save_path_t):
        #         os.mkdir(save_path_t)
        #     save_image_tensor2cv2(target_viz[i], f'{save_path_t}/target_{t[i]}_{self.count}.png')
        #     save_image_tensor2cv2(cond_viz[i], f'{save_path_t}/cond_{t[i]}_{self.count}.png')
        #     save_image_tensor2cv2(noise_viz[i], f'{save_path_t}/noise_{t[i]}_{self.count}.png')
        #     save_image_tensor2cv2(pred_viz[i], f'{save_path_t}/pred_{t[i]}_{self.count}.png')
        #     self.count += 1

        return loss, loss_dict