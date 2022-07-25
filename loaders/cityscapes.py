import glob
import os.path

from typing import Optional
from PIL import Image, ImageFile

from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CityScapesDataset(Dataset):

    LEFT_PATH = "leftImg8bit"
    RIGHT_PATH = "rightImg8bit"
    EXTENSION = "png"

    def __init__(self, root: str, split: str,
                 transform: Optional[object] = None,
                 limit: Optional[int] = None):

        if split not in ("train", "val", "test"):
            raise ValueError("Split must be either 'train', 'val' or 'test'.")

        left_glob = os.path.join(root, self.LEFT_PATH, split,
                                 "**", f"*.{self.EXTENSION}")

        right_glob = os.path.join(root, self.RIGHT_PATH, split,
                                  "**", f"*.{self.EXTENSION}")

        self.lefts = sorted(glob.glob(left_glob))[:limit]
        self.rights = sorted(glob.glob(right_glob))[:limit]

        if len(self.lefts) != len(self.rights):
            raise Exception(f"Number of left images ({len(self.lefts)}) does "
                            "not match the number of right images "
                            f"({len(self.rights)}) in the dataset.")

        self.transform = transform

    def __getitem__(self, idx):
        left_path = self.lefts[idx]
        right_path = self.rights[idx]

        left = Image.open(left_path).convert('RGB')
        right = Image.open(right_path).convert('RGB')

        image_pair = {"left": left, "right": right}

        if self.transform is not None:
            image_pair = self.transform(image_pair)

        return image_pair

    def __len__(self):
        return len(self.lefts)
