import glob
import os.path

from typing import Optional
from PIL import Image, ImageFile

from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class HamlynMultiviewDataset(Dataset):

    LEFT_PATH = "image_0"
    RIGHT_PATH = "image_1"
    EXTENSION = "png"

    def __init__(self, root: str, split: str,
                 transform: Optional[object] = None,
                 limit: Optional[int] = None):

        if split not in ("train", "test"):
            raise ValueError("Split must be either 'train' or 'test'.")

        left_image_glob = os.path.join(root, split, self.LEFT_PATH,
                                       f"*.{self.EXTENSION}")

        right_image_glob = os.path.join(root, split, self.RIGHT_PATH,
                                        f"*.{self.EXTENSION}")

        left_images = sorted(glob.glob(left_image_glob))[:limit]
        right_images = sorted(glob.glob(right_image_glob))[:limit]

        self.images = tuple(zip(left_images, right_images))
        
        self.transform = transform

    def __getitem__(self, idx):
        left_path, right_path = self.images[idx]

        left_image = Image.open(left_path).convert('RGB')
        right_image = Image.open(right_path).convert('RGB')

        if self.transform is not None:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)
           
        return left_image, right_image

    def __len__(self):
        return len(self.images)
