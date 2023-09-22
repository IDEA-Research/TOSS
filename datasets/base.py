from torchvision import transforms as T

from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import torch
import random
import math
import pdb
from viz import save_image_tensor2cv2


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, split='train', text="a green chair", img_size=512, downsample=1.0):
        self.img_w, self.img_h = img_size, img_size
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.define_transforms()
        self.text = text

    def read_intrinsics(self):
        raise NotImplementedError

    def define_transforms(self):
        self.transform = transforms.Compose([T.ToTensor(), transforms.Resize(size=(self.img_w, self.img_h))])
        # self.transform = T.ToTensor()

    def __len__(self):
        # if self.split.startswith('train'):
        #     return 1000
        return len(self.poses)
    
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])
    
    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        d_T = torch.tensor([d_theta.item(), d_azimuth.item(), d_z.item()])
        return d_T

    def get_T_w2c(self, target_RT, cond_RT):
        T_target = target_RT[:, -1]
        T_cond = cond_RT[:, -1]

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        # d_z = (z_target - z_cond) / np.max(z_cond) * 2
        # pdb.set_trace()
        
        d_T = torch.tensor([d_theta.item(), d_azimuth.item(), d_z.item()])
        return d_T

    def __getitem__(self, idx):
        # camera pose and img
        poses = self.poses[idx]
        img = self.imgs[idx]
        prompt = self.text

        # condition
        # idx_cond = idx % 5
        # idx_cond = 1
        idx_cond = random.randint(0, 4)
        # idx_cond = random.randint(0, len(self.poses)-1)
        poses_cond = self.poses[idx_cond]
        img_cond = self.imgs[idx_cond]

        # if len(self.imgs)>0: # if ground truth available
        #     img_rgb = imgs[:, :,:3]
        #     img_rgb_cond = imgs_cond[:, :,:3]
            # if imgs.shape[-1] == 4: # HDR-NeRF data
            #     sample['exposure'] = rays[0, 3] # same exposure for all rays

        # Normalize target images to [-1, 1].
        target = (img.float() / 127.5) - 1.0
        # Normalize source images to [0, 1].
        condition = img_cond.float() / 255.0
        # save_image_tensor2cv2(condition.permute(2,0,1), "./viz_fig/input_cond.png")
        
        # get delta pose
        # delta_pose = self.get_T(target_RT=poses, cond_RT=poses_cond)
        delta_pose = self.get_T_w2c(target_RT=poses, cond_RT=poses_cond) 

        return dict(jpg=target, txt=prompt, hint=condition, delta_pose=delta_pose)