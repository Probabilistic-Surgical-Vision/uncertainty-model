import glob
import os.path

from typing import Optional
from PIL import Image, ImageFile

from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CityScapesMultiviewDataset(Dataset):

    LEFT_PATH = "leftImg8bit_sequence"
    RIGHT_PATH = "rightImg8bit_sequence"
    EXTENSION = "png"

    def __init__(self, root: str, split: str,
                 transform: Optional[object] = None,
                 limit: Optional[int] = None):

        if split not in ("train", "val", "test"):
            raise ValueError("Split must be either 'train', 'val' or 'test'.")

        left_image_glob = os.path.join(root, self.LEFT_PATH, split,
                                       "**", f"*.{self.EXTENSION}")

        right_image_glob = os.path.join(root, self.RIGHT_PATH, split,
                                        "**", f"*.{self.EXTENSION}")

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
           
        return dict(left=left_image, right=right_image)

    def __len__(self):
        return len(self.images)
