import os
from PIL import Image, ImageFile
from glob import glob
from torch.utils.data import Dataset
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

class SCAREDLoader(Dataset):
    def __init__(self, root_dir, split, transform=None, n_val_samples=600):
        if split =='train' or split=='val':
            self.left_paths_all= sorted(glob('{}/train/**/**/frames_resized/left/*.png'.format(root_dir)))
            self.right_paths_all= sorted(glob('{}/train/**/**/frames_resized/right/*.png'.format(root_dir)))
            
            if split=='train':
                self.left_paths= self.left_paths_all[0:len(self.left_paths_all)-n_val_samples]
                self.right_paths= self.right_paths_all[0:len(self.right_paths_all)-n_val_samples]
    
            elif split=='val':
                self.left_paths= self.left_paths_all[len(self.left_paths_all)-n_val_samples::]
                self.right_paths= self.right_paths_all[len(self.right_paths_all)-n_val_samples::]

        elif split=='test':
            self.left_paths= sorted(glob('{}/test/**/**/frames_resized/left/*.png'.format(root_dir)))
            self.right_paths= sorted(glob('{}/test/**/**/frames_resized/right/*.png'.format(root_dir)))
            
        
        self.transform = transform

    def __getitem__(self, idx):
        #print(self.left_paths[idx])
        left_image = Image.open(self.left_paths[idx]).convert('RGB')
        right_image = Image.open(self.right_paths[idx]).convert('RGB')
            
        sample = {'left_image': left_image, 'right_image': right_image}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def __len__(self):
        return len(self.left_paths)


class SCAREDTestLoader(Dataset):
    def __init__(self, dataset, keyframe, transform=None):
        
        root_dir=r'/mnt/398441BC350EAA68/Data/CV/endoscopic_challenge_1280x1024'
        
        self.left_paths_all= sorted(glob('{}/test_dataset_{}/keyframe_{}/left/*.png'.format(root_dir, dataset, keyframe)))
        self.right_paths_all= sorted(glob('{}/test_dataset_{}/keyframe_{}/right/*.png'.format(root_dir, dataset, keyframe)))
        if len(self.right_paths_all)!= len(self.left_paths_all):
            self.right_paths_all=self.left_paths_all
        #self.left_disp_all = sorted(glob('{}/test_dataset_{}/keyframe_{}/left_disp/*.png'.format(root_dir, dataset, keyframe)))
        #/mnt/398441BC350EAA68/Data/CV/endoscopic_challenge_1280x1024/test_dataset_9/keyframe_0/right

        self.transform = transform
    
    def load_gt_disp_scared(self, path):
        #all_disps= sorted(glob('{}/*.png'.format(path)))
        #gt_disparities = []
        #for i in range(len(all_disps)):
        disp = cv2.imread(path, -1)
        disp = disp.astype(np.float32) / 256
        return disp


    def __getitem__(self, idx):
        #print(self.left_paths[idx])
        left_image = Image.open(self.left_paths_all[idx]).convert('RGB')
        right_image = Image.open(self.right_paths_all[idx]).convert('RGB')
        #left_disp = self.load_gt_disp_scared(self.left_disp_all[idx])

        sample = {'left_image': left_image, 'right_image': right_image}

        if self.transform:
            sample = self.transform(sample)

        return sample#, left_disp
    
    def __len__(self):
        return len(self.left_paths_all)