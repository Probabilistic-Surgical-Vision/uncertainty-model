import os
from PIL import Image, ImageFile
from glob import glob
from torch.utils.data import Dataset
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True


class KittiEigenStereoLoader(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir= root_dir
        if split =='train':
            self.all_paths= np.load('./Data_files_split/kitti_splits/eigen_paths/eigen_train_paths.npy')

        elif split=='val' or split=='test':
            self.all_paths= np.load('./Data_files_split/kitti_splits/eigen_paths/eigen_val_paths.npy')
        #test is actually meant to be the eigen paths used for testing! i.e. same as in the eval.py file.
        self.transform = transform

    def __getitem__(self, idx):
        split_path= self.all_paths[idx].split(' ') # so 0 = left path, 1 = right path
        left_image = Image.open('{}/{}'.format(self.root_dir, split_path[0])).convert('RGB')
        right_image = Image.open('{}/{}'.format(self.root_dir, split_path[1])).convert('RGB')
            
        sample = {'left_image': left_image, 'right_image': right_image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.all_paths)


class KittiLoaderCustom(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.left_paths= sorted(glob('{}/**/**/image_02/data/*.png'.format(root_dir)))
        self.right_paths= sorted(glob('{}/**/**/image_03/data/*.png'.format(root_dir)))

        if split =='train':
            self.left_paths= self.left_paths[0: int(0.63*len(self.left_paths))]
            self.right_paths= self.right_paths[0: int(0.63*len(self.right_paths))]

        elif split=='val' or split=='test':
            self.left_paths= self.left_paths[int(0.95*len(self.left_paths)): len(self.left_paths)]
            self.right_paths= self.right_paths[int(0.95*len(self.right_paths)): len(self.right_paths)]
        
        self.transform = transform

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx]).convert('RGB')
        right_image = Image.open(self.right_paths[idx]).convert('RGB')
            
        sample = {'left_image': left_image, 'right_image': right_image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.left_paths)


class KittiStereo2015Loader(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.left_paths= '{}/image_2/'.format(root_dir)
        self.right_paths= '{}/image_3/'.format(root_dir)
        self.kitti_imgs= np.load('./kitti_eval_files.npy')
        self.transform = transform

    def __getitem__(self, idx):
        left_image = Image.open('{}/{}'.format(self.left_paths, self.kitti_imgs[idx])).convert('RGB')
        right_image = Image.open('{}/{}'.format(self.right_paths, self.kitti_imgs[idx])).convert('RGB')
            
        sample = {'left_image': left_image, 'right_image': right_image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return np.shape(self.kitti_imgs)[0]