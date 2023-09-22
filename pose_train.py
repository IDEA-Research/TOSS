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

# configure
from opt import get_opts
import pdb


if __name__ == '__main__':
    # Configs
    hparams = get_opts()
    logger_freq = hparams.logger_freq
    sd_locked = True
    only_mid_control = False
    cfgs = OmegaConf.load(hparams.model_cfg)


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = instantiate_from_config(cfgs.model)

    # get missing keys
    if hparams.resume_path != './models/control_sd15_pose_ini.ckpt':
        missing, unexpected = model.load_state_dict(load_state_dict(hparams.resume_path, location='cpu'), strict=False) 
        print(f"Restored from {hparams.resume_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")

    # model.load_state_dict(load_state_dict(hparams.resume_path, location='cpu'), strict=False)

    # reweight noise scheduer
    if hparams.register_scheduler:
        model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000, linear_start=0.00085, linear_end=0.016)
    model.learning_rate = hparams.lr
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    # model.control_model.uncond_pose = hparams.uncond_pose
    model.eval()

    # data
    if 'objaverse' in hparams.dataset_name:
        if "car" in hparams.dataset_name:
            dataloader = instantiate_from_config(cfgs.data_car)
            print("Using objaverse car data!")
        elif "800k" in hparams.dataset_name:
            dataloader = instantiate_from_config(cfgs.data800k)
        else:
            dataloader = instantiate_from_config(cfgs.data)
        dataloader.prepare_data()
        dataloader.setup()
    elif "srn" in hparams.dataset_name:
        if "chair" in hparams.dataset_name:
            dataloader = instantiate_from_config(cfgs.srn_chairs)
        else:
            dataloader = instantiate_from_config(cfgs.srn_data)
        dataloader.prepare_data()
        dataloader.setup()
    else:
        kwargs = {'root_dir': hparams.root_dir}
        dataset = dataset_dict[hparams.dataset_name](split=hparams.split, text=hparams.text, img_size=hparams.img_size, **kwargs)
        dataloader = DataLoader(dataset, num_workers=0, batch_size=hparams.batch_size, shuffle=True)
    

    # Train!
    save_dir = f'./exp/{hparams.dataset_name}/{hparams.exp_name}/'
    logger = TensorBoardLogger(save_dir=save_dir,
                               name=hparams.exp_name, default_hp_metric=False)
    img_logger = ImageLogger(batch_frequency=logger_freq, epoch_frequency=hparams.log_interval_epoch)
    ckpt_cb = ModelCheckpoint(dirpath=f'{save_dir}/ckpt/',
                              filename='{epoch:d}',
                              save_weights_only=False,
                              every_n_epochs=hparams.ckpt_interval,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [img_logger, ckpt_cb]
    trainer = pl.Trainer(gpus=hparams.num_gpus, callbacks=callbacks, \
                        precision=16, \
                        # amp_backend='apex', amp_level="O2", \
                        check_val_every_n_epoch=20,
                        logger=logger, max_epochs=hparams.num_epochs, \
                        resume_from_checkpoint=hparams.resume_path if hparams.resume else None,
                        # strategy="ddp",
                        accumulate_grad_batches=8//hparams.num_gpus \
                            if hparams.acc_grad==None else hparams.acc_grad,
                        plugins=DDPPlugin(find_unused_parameters=False),
                        accelerator="ddp"
                        )
    
    # trainer.validate(model, dataloader)
    trainer.fit(model, dataloader)

