import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
import pytorch_lightning as pl
import random
import matplotlib.pyplot as plt
import os, sys, json
import math
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from ldm.util import instantiate_from_config
import pdb


class SRNDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, train=None, validation=None,
                 test=None, num_workers=4, caption="recaption", **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.caption = caption

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation
        if test is not None:
            dataset_config = test

        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x, 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)


    def train_dataloader(self):
        dataset = SRNDataset(root_dir=self.root_dir, total_view=50, validation=False, \
                                image_transforms=self.image_transforms, caption=self.caption)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = SRNDataset(root_dir=self.root_dir, total_view=251, validation=True, \
                                image_transforms=self.image_transforms, caption=self.caption)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(SRNDataset(root_dir=self.root_dir, total_view=251, validation=False, test=True),
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, caption=self.caption)


class SRNDataset(Dataset):
    def __init__(self,
        root_dir,
        image_transforms=[],
        ext="png",
        postprocess=None,
        return_paths=False,
        total_view=4,
        validation=False,
        test=False,
        caption="random",
        test_sub=False,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        if caption == 'random':
            if validation:
                paths_file = f"/comp_robot/mm_generative/data/nerf/srn_chairs/val_caption_allviews.json"
            elif test:
                if test_sub:
                    paths_file = "/comp_robot/mm_generative/data/nerf/srn_chairs/test_sub_caption_allviews.json"
                else:
                    paths_file = "/comp_robot/mm_generative/data/nerf/srn_chairs/test_caption_allviews.json"
            else:
                paths_file = "/comp_robot/mm_generative/data/nerf/srn_chairs/train_caption_allviews.json"
        elif caption == "rerank":
            if validation:
                paths_file = "/comp_robot/mm_generative/data/nerf/srn_chairs/val_caption_clip_rerank.json"
            elif test:
                if test_sub:
                    paths_file = "/comp_robot/mm_generative/data/nerf/srn_chairs/test_sub_caption_clip_rerank.json"
                else:
                    paths_file = "/comp_robot/mm_generative/data/nerf/srn_chairs/test_caption_clip_rerank.json"
            else:
                paths_file = "/comp_robot/mm_generative/data/nerf/srn_chairs/train_caption_clip_rerank.json"
        else:
            raise(f"caption type {caption} not supported")
        
        if validation:
            self.root_dir = os.path.join(self.root_dir, 'chairs_val')
        elif test:
            if test_sub:
                self.root_dir = os.path.join(self.root_dir, 'chairs_test_sub')
            else:
                self.root_dir = os.path.join(self.root_dir, 'chairs_test')
        else:
            self.root_dir = os.path.join(self.root_dir, 'chairs_train')
        self.uids = [os.path.join(self.root_dir, uid) for uid in os.listdir(self.root_dir)]
        print('============= length of dataset %d =============' % len(self.uids))


        with open(paths_file) as f:
            self.paths = json.load(f)
            if isinstance(self.paths, dict):
                self.paths_2_prompts = self.paths
                self.paths = list(self.paths.keys())
            elif isinstance(self.paths, list):
                self.paths = self.paths
                raise NotImplementedError("not implemented for list, not sure how to handle prompts")
            else:
                raise NotImplementedError
    
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.uids)
        
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

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(f'Error loading image {path}')
            sys.exit()
        if img.shape[-1] == 4:
            img = img[:,:,:3]*img[:,:, -1:]+(1-img[:,:, -1:])
        # img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img
    
    def get_3x4_RT_matrix_from_cam2world(self, pose_path):
        '''Convert cam2world matrix to 3x4 P matrix.'''
        cam2world = np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
        W = np.linalg.inv(cam2world)
        R, T = W[:3, :3], W[:3, 3]
        P = np.hstack([R, T[:, None]])
        return P

    def __getitem__(self, index):
        data = {}
        index_target, index_cond = random.sample(range(self.total_view), 2) # without replacement
        uid_folder = self.uids[index]
        if self.return_paths:
            data["path"] = str(uid_folder)

        color = [1., 1., 1., 1.]
        if isinstance(self.paths_2_prompts[self.paths[index]], dict):
            # randomly select a prompt 
            prompt_dict = self.paths_2_prompts[self.paths[index]]
            prompt = prompt_dict[random.choice(list(prompt_dict.keys()))]
        else:
            prompt = self.paths_2_prompts[self.paths[index]]

        try:
            target_im = self.process_im(self.load_im(os.path.join(uid_folder, 'rgb', '%06d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(uid_folder, 'rgb', '%06d.png' % index_cond), color))
            target_RT = self.get_3x4_RT_matrix_from_cam2world(os.path.join(uid_folder, 'pose', '%06d.txt' % index_target))
            cond_RT = self.get_3x4_RT_matrix_from_cam2world(os.path.join(uid_folder, 'pose', '%06d.txt' % index_cond))
        except:
            uid_folder = self.uids[0]
            target_im = self.process_im(self.load_im(os.path.join(uid_folder, 'rgb', '%06d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(uid_folder, 'rgb', '%06d.png' % index_cond), color))
            target_RT = self.get_3x4_RT_matrix_from_cam2world(os.path.join(uid_folder, 'pose', '%06d.txt' % index_target))
            cond_RT = self.get_3x4_RT_matrix_from_cam2world(os.path.join(uid_folder, 'pose', '%06d.txt' % index_cond))
        
        # Normalize target images to [-1, 1]. No need to normalize cond_im
        # !! ugly, but this is the only way to accomodate the LDM code
        # !! target_im and comd_im need to convert to 'b h w c' format, target_im will be converted to 'b c h w' format in the LDM/ddpm part, but the cond_im must be converted to 'b h w c' format in the `get_input()`
        target_im = (target_im - 0.5) / 0.5 
        # delta pose (4,)
        delta_pose = self.get_T(target_RT, cond_RT)

        # upsample
        target_im = F.interpolate(target_im.unsqueeze(0).permute(0,3,1,2), [256,256], mode='bilinear', align_corners=False).permute(0,2,3,1).squeeze(0)
        cond_im = F.interpolate(cond_im.unsqueeze(0).permute(0,3,1,2), [256,256], mode='bilinear', align_corners=False).permute(0,2,3,1).squeeze(0)

        # dict
        data = dict(jpg=target_im, txt=prompt, hint=cond_im, delta_pose=delta_pose)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
    

class SRNTestDataset(SRNDataset):
    def __init__(self,
        root_dir,
        image_transforms=[],
        ext="png",
        postprocess=None,
        return_paths=False,
        total_view=251,
        caption="random",
        test_sub=False,
        ) -> None:
        """Create a dataset for evaluation
        """
        super().__init__(
            root_dir, image_transforms, ext, postprocess, return_paths, total_view, 
            test=True, caption=caption, test_sub=test_sub
        )
    
    def __getitem__(self, index):
        data = {}
        uid_folder = self.uids[index]
        
        refv_index = 64
        color = [1., 1., 1., 1.]
        if isinstance(self.paths_2_prompts[self.paths[index]], dict):
            # randomly select a prompt 
            prompt_dict = self.paths_2_prompts[self.paths[index]]
            prompt = prompt_dict[random.choice(list(prompt_dict.keys()))]
        else:
            prompt = self.paths_2_prompts[self.paths[index]]

        cond_im = self.process_im(self.load_im(os.path.join(uid_folder, 'rgb', '%06d.png' % refv_index), color))
        cond_RT = self.get_3x4_RT_matrix_from_cam2world(os.path.join(uid_folder, 'pose', '%06d.txt' % refv_index))

        target_ims = []
        Ts = []
        paths = []
        for i in range(self.total_view):
            target_im = self.process_im(self.load_im(os.path.join(uid_folder, 'rgb', '%06d.png' % i), color))
            target_RT = self.get_3x4_RT_matrix_from_cam2world(os.path.join(uid_folder, 'pose', '%06d.txt' % i))
            T = self.get_T(target_RT, cond_RT)

            target_ims.append(target_im)
            Ts.append(T)
            paths.append(os.path.join(uid_folder, 'rgb', '%06d.png' % i))


        data["txt"] = prompt
        data["jpgs"] = torch.stack(target_ims, dim=0)
        data["delta_poses"] = torch.stack(Ts, dim=0)
        data["hint"] = cond_im

        if self.return_paths:
            data["paths"] = paths
        return data