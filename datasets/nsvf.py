import torch
import glob
import numpy as np
import os
from PIL import Image
from einops import rearrange
from tqdm import tqdm
import json
import cv2
import numpy as np
import pdb
from viz import save_image_tensor2cv2
import random
import math
from torchvision import transforms

from .ray_utils import get_ray_directions

from .base import BaseDataset


class NSVFDataset(BaseDataset):
    def __init__(self, root_dir, split='train', text="a green chair", downsample=1.0, img_size=512, **kwargs):
        super().__init__(root_dir, split, text, downsample=downsample, img_size=img_size)
        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            xyz_min, xyz_max = \
                np.loadtxt(os.path.join(root_dir, 'bbox.txt'))[:6].reshape(2, 3)
            self.shift = (xyz_max+xyz_min)/2
            self.scale = (xyz_max-xyz_min).max()/2 * 1.05 # enlarge a little

            # hard-code fix the bound error for some scenes...
            if 'Mic' in self.root_dir: self.scale *= 1.2
            elif 'Lego' in self.root_dir: self.scale *= 1.1

            self.read_meta(split)

    def read_intrinsics(self):
        if 'Synthetic' in self.root_dir or 'Ignatius' in self.root_dir:
            with open(os.path.join(self.root_dir, 'intrinsics.txt')) as f:
                fx = fy = float(f.readline().split()[0]) * self.downsample
            if 'Synthetic' in self.root_dir:
                # orginal w h
                # w = h = int(800*self.downsample)
                w = self.img_w
                h = self.img_h
                fx = fy = fx * self.img_w / 800
            else:
                w, h = int(1920*self.downsample), int(1080*self.downsample)

            K = np.float32([[fx, 0, w/2],
                            [0, fy, h/2],
                            [0,  0,   1]])
        else:
            K = np.loadtxt(os.path.join(self.root_dir, 'intrinsics.txt'),
                           dtype=np.float32)[:3, :3]
            if 'BlendedMVS' in self.root_dir:
                w, h = int(768*self.downsample), int(576*self.downsample)
            elif 'Tanks' in self.root_dir:
                w, h = int(1920*self.downsample), int(1080*self.downsample)
            K[:2] *= self.downsample

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split):
        self.poses = []
        self.imgs = []

        if split == 'test_traj': # BlendedMVS and TanksAndTemple
            if 'Ignatius' in self.root_dir:
                poses_path = \
                    sorted(glob.glob(os.path.join(self.root_dir, 'test_pose/*.txt')))
                poses = [np.loadtxt(p) for p in poses_path]
            else:
                poses = np.loadtxt(os.path.join(self.root_dir, 'test_traj.txt'))
                poses = poses.reshape(-1, 4, 4)
            for pose in poses:
                c2w = pose[:3]
                c2w[:, 0] *= -1 # [left down front] to [right down front]
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
                self.poses += [c2w]
        else:
            if split == 'train': prefix = '0_'
            elif split == 'trainval': prefix = '[0-1]_'
            elif split == 'val': prefix = '1_'
            elif 'Synthetic' in self.root_dir: prefix = '2_' # test set for synthetic scenes
            elif split == 'test': prefix = '1_' # test set for real scenes
            else: raise ValueError(f'{split} split not recognized!')
            imgs = sorted(glob.glob(os.path.join(self.root_dir, 'rgb', prefix+'*.png')))
            poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose', prefix+'*.txt')))

            print(f'Loading {len(imgs)} {split} images ...')
            for img, pose in tqdm(zip(imgs, poses)):
                c2w = np.loadtxt(pose)[:3]
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
                self.poses += [c2w]
                img = Image.open(img).resize((self.img_w, self.img_h), Image.LANCZOS)
                img = self.transform(img)
                
                img = rearrange(img, 'c h w -> h w c')
                if 'Jade' in self.root_dir or 'Fountain' in self.root_dir:
                    # these scenes have black background, changing to white
                    img[torch.all(img<=0.1, dim=-1)] = 1.0
                if img.shape[-1] == 4:
                    img = img[:,:,:3]*img[:,:, -1:]+(1-img[:,:, -1:]) # blend A to RGB
                
                self.imgs += [img]

            self.imgs = torch.stack(self.imgs) # (N_images, h, w, ?)
            self.imgs = self.imgs.mul_(255).add_(0.5).clamp_(0, 255)
        # self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)


# for nerf
class NSVFDataset_v2(BaseDataset):
    def __init__(self, root_dir, split='train', text="a green chair", downsample=1.0, img_size=512, **kwargs):
        self.img_w, self.img_h = img_size, img_size
        super().__init__(root_dir, split, text, downsample)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            xyz_min, xyz_max = \
                np.loadtxt(os.path.join(root_dir, 'bbox.txt'))[:6].reshape(2, 3)
            self.shift = (xyz_max+xyz_min)/2
            self.scale = (xyz_max-xyz_min).max()/2 * 1.05 # enlarge a little

            # hard-code fix the bound error for some scenes...
            if 'Mic' in self.root_dir: self.scale *= 1.2
            elif 'Lego' in self.root_dir: self.scale *= 1.1 * 3

            self.read_meta(split)
    
    def define_transforms(self):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(self.img_w, self.img_w))])

    def read_intrinsics(self):
        if 'Synthetic' in self.root_dir or 'Ignatius' in self.root_dir:
            with open(os.path.join(self.root_dir, 'intrinsics.txt')) as f:
                fx = fy = float(f.readline().split()[0]) * self.downsample
            if 'Synthetic' in self.root_dir:
                # orginal w h
                # w = h = int(800*self.downsample)
                w = self.img_w
                h = self.img_h
                fx = fy = fx * self.img_w / 800
            else:
                w, h = int(1920*self.downsample), int(1080*self.downsample)

            K = np.float32([[fx, 0, w/2],
                            [0, fy, h/2],
                            [0,  0,   1]])
        else:
            K = np.loadtxt(os.path.join(self.root_dir, 'intrinsics.txt'),
                           dtype=np.float32)[:3, :3]
            if 'BlendedMVS' in self.root_dir:
                w, h = int(768*self.downsample), int(576*self.downsample)
            elif 'Tanks' in self.root_dir:
                w, h = int(1920*self.downsample), int(1080*self.downsample)
            K[:2] *= self.downsample

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split):
        self.poses = []
        self.imgs = []

        if split == 'test_traj': # BlendedMVS and TanksAndTemple
            if 'Ignatius' in self.root_dir:
                poses_path = \
                    sorted(glob.glob(os.path.join(self.root_dir, 'test_pose/*.txt')))
                poses = [np.loadtxt(p) for p in poses_path]
            else:
                poses = np.loadtxt(os.path.join(self.root_dir, 'test_traj.txt'))
                poses = poses.reshape(-1, 4, 4)
            for pose in poses:
                c2w = pose[:3]
                c2w[:, 0] *= -1 # [left down front] to [right down front]
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
                self.poses += [c2w]
        else:
            if split == 'train': prefix = '0_'
            elif split == 'trainval': prefix = '[0-1]_'
            elif split == 'val': prefix = '1_'
            elif 'Synthetic' in self.root_dir: prefix = '2_' # test set for synthetic scenes
            elif split == 'test': prefix = '1_' # test set for real scenes
            else: raise ValueError(f'{split} split not recognized!')
            imgs = sorted(glob.glob(os.path.join(self.root_dir, 'rgb', prefix+'*.png')))
            poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose', prefix+'*.txt')))

            print(f'Loading {len(imgs)} {split} images ...')
            for img, pose in tqdm(zip(imgs, poses)):
                c2w = np.loadtxt(pose)[:3]
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
                self.poses += [c2w]
                img = Image.open(img)
                img = self.transform(img)
                
                img = rearrange(img, 'c h w -> h w c')
                if 'Jade' in self.root_dir or 'Fountain' in self.root_dir:
                    # these scenes have black background, changing to white
                    img[torch.all(img<=0.1, dim=-1)] = 1.0
                if img.shape[-1] == 4:
                    img = img[:,:,:3]*img[:,:, -1:]+(1-img[:,:, -1:]) # blend A to RGB
                
                self.imgs += [img]

            self.imgs = torch.stack(self.imgs) # (N_images, h, w, ?)
            self.imgs = self.imgs.mul_(255).add_(0.5).clamp_(0, 255)

    def __getitem__(self, idx):
        # camera pose and img
        poses = self.poses[idx]
        img = self.imgs[idx]
        prompt = self.text

        # condition
        # idx_cond = idx % 5
        idx_cond = random.randint(0, len(self.poses)-1)
        poses_cond = self.poses[idx_cond]
        img_cond = self.imgs[idx_cond]

        # Normalize target images to [-1, 1].
        target = (img.float() / 127.5) - 1.0
        # Normalize source images to [0, 1].
        condition = img_cond.float() / 255.0

        return dict(jpg=target, txt=prompt, hint=condition, cond_pose=torch.FloatTensor(poses_cond), \
                        target_pose=torch.FloatTensor(poses), intrinsics=self.K)



class NSVFDataset_all(BaseDataset):
    def __init__(self, root_dir, split='train', text="a green chair", downsample=1.0, img_size=512, **kwargs):
        super().__init__(root_dir, split, text, downsample)
        self.name_list = ["Ship", "Lego","Mic","Materials","Ficus","Hotdog","Drums"]
        self.text_dic = {"Ship": "a 3d model of a pirate ship floating in the ocean", 
                        "Lego": "a yellow lego bulldozer sitting on top of a table",
                        "Mic":"a microphone on a tripod on a white background",
                        "Materials": "a group of different colored balls on a white background",
                        "Ficus":"a small tree in a black pot on a white background",
                        "Hotdog": "a plate with two hot dogs and ketchup on it",
                        "Drums":"a red drum set on a white background"}
        self.img_list = []
        self.pose_list = []

        for i in range(len(self.name_list)):
            self.root_dir = root_dir + self.name_list[i]
            self.read_intrinsics()
            if kwargs.get('read_meta', True):
                xyz_min, xyz_max = \
                    np.loadtxt(os.path.join(self.root_dir, 'bbox.txt'))[:6].reshape(2, 3)
                self.shift = (xyz_max+xyz_min)/2
                self.scale = (xyz_max-xyz_min).max()/2 * 1.05 # enlarge a little

                # hard-code fix the bound error for some scenes...
                if 'Mic' in self.root_dir: self.scale *= 1.2
                elif 'Lego' in self.root_dir: self.scale *= 1.1

                self.read_meta(split)

    def read_intrinsics(self):
        if 'Synthetic' in self.root_dir or 'Ignatius' in self.root_dir:
            with open(os.path.join(self.root_dir, 'intrinsics.txt')) as f:
                fx = fy = float(f.readline().split()[0]) * self.downsample
            if 'Synthetic' in self.root_dir:
                # orginal w h
                # w = h = int(800*self.downsample)
                w = h = 512
            else:
                w, h = int(1920*self.downsample), int(1080*self.downsample)

            K = np.float32([[fx, 0, w/2],
                            [0, fy, h/2],
                            [0,  0,   1]])
        else:
            K = np.loadtxt(os.path.join(self.root_dir, 'intrinsics.txt'),
                           dtype=np.float32)[:3, :3]
            if 'BlendedMVS' in self.root_dir:
                w, h = int(768*self.downsample), int(576*self.downsample)
            elif 'Tanks' in self.root_dir:
                w, h = int(1920*self.downsample), int(1080*self.downsample)
            K[:2] *= self.downsample

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split):
        self.poses = []
        self.imgs = []

        if split == 'test_traj': # BlendedMVS and TanksAndTemple
            if 'Ignatius' in self.root_dir:
                poses_path = \
                    sorted(glob.glob(os.path.join(self.root_dir, 'test_pose/*.txt')))
                poses = [np.loadtxt(p) for p in poses_path]
            else:
                poses = np.loadtxt(os.path.join(self.root_dir, 'test_traj.txt'))
                poses = poses.reshape(-1, 4, 4)
            for pose in poses:
                c2w = pose[:3]
                c2w[:, 0] *= -1 # [left down front] to [right down front]
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
                self.poses += [c2w]
        else:
            if split == 'train': prefix = '0_'
            elif split == 'trainval': prefix = '[0-1]_'
            elif split == 'val': prefix = '1_'
            elif 'Synthetic' in self.root_dir: prefix = '2_' # test set for synthetic scenes
            elif split == 'test': prefix = '1_' # test set for real scenes
            else: raise ValueError(f'{split} split not recognized!')
            imgs = sorted(glob.glob(os.path.join(self.root_dir, 'rgb', prefix+'*.png')))
            poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose', prefix+'*.txt')))

            print(f'Loading {len(imgs)} {split} images ...')
            for img, pose in tqdm(zip(imgs, poses)):
                c2w = np.loadtxt(pose)[:3]
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
                self.poses += [c2w]
                img = Image.open(img).resize((512,512), Image.LANCZOS)
                img = self.transform(img)
                
                img = rearrange(img, 'c h w -> h w c')
                if 'Jade' in self.root_dir or 'Fountain' in self.root_dir:
                    # these scenes have black background, changing to white
                    img[torch.all(img<=0.1, dim=-1)] = 1.0
                if img.shape[-1] == 4:
                    img = img[:,:,:3]*img[:,:, -1:]+(1-img[:,:, -1:]) # blend A to RGB
                
                self.imgs += [img]

            self.imgs = torch.stack(self.imgs) # (N_images, h, w, ?)
            self.imgs = self.imgs.mul_(255).add_(0.5).clamp_(0, 255)

            # save to list
            self.img_list.append(self.imgs)
            self.pose_list.append(self.poses)
        # self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)

    def __len__(self):
        return len(self.poses)*len(self.name_list)
    
    def __getitem__(self, idx):
        # dataset_idx
        dataset_idx = idx%len(self.name_list)
        # camera pose and img
        idx = idx%len(self.poses)
        poses = self.pose_list[dataset_idx][idx]
        img = self.img_list[dataset_idx][idx]
        prompt = self.text_dic[self.name_list[dataset_idx]]

        # condition
        # idx_cond = idx % 5
        idx_cond = random.randint(0,4)
        # idx_cond = random.randint(0, len(self.poses)-1)
        poses_cond = self.pose_list[dataset_idx][idx_cond]
        img_cond = self.img_list[dataset_idx][idx_cond]

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
        # delta pose (3,)
        delta_pose = self.get_T_w2c(target_RT=poses, cond_RT=poses_cond) 

        return dict(jpg=target, txt=prompt, hint=condition, delta_pose=delta_pose)