import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as T2
from .util.image_utils import RRC
from .util.video_utils import sample_frame
from .util import CropMAEStrategies

from PIL import Image

class CropMAE_PIPE(data.Dataset):
    def __init__(self, files, args):   
        self.crop_strategy = getattr(CropMAEStrategies, args.crop_strategy)(args)
        self.args = args
        self.files = files

    def __len__(self):
        return len(self.files)

class CropMAE_Image_Pipe(CropMAE_PIPE):
    def __getitem__(self, idx):
        with Image.open(self.files[idx]) as img:
            img = img.convert("RGB")
            return self.crop_strategy(img)
        
class CropMAE_Video_Pipe(CropMAE_PIPE):
    def __getitem__(self, idx):
        video_path = self.files[idx]
        frames = []
        for f1 in sample_frame(video_path, repeated_sampling=self.args.repeated_sampling_factor):
            frames.append(self.crop_strategy(f1))
        return torch.stack(frames, dim=0)