import torchvision.transforms.v2 as T2
from torchvision.transforms import functional as F
import torch.utils.data as data
import random
import torch
from .util.image_utils import DoubleRRC
from .util.video_utils import sample_frames


class DoubleRandomResizedCrop:

    def __init__(
            self,
            hflip_p=0.5,
            size=(224, 224),
            scale=(0.5, 1.0),
            ratio=(3./4., 4./3.),
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
            use_same_crop=True
        ):
        self.hflip_p = hflip_p
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.use_same_crop = use_same_crop
        self.resized_crop = DoubleRRC(size, scale, ratio, interpolation, antialias)
        self.post_process = T2.Compose([
            T2.ToImage(),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img1, img2):
        
        # Crop
        cropped_img_1, cropped_img_2 = self.resized_crop(img1, img2, same_crop=self.use_same_crop)
        
        # Flip
        if random.random() < self.hflip_p:
            cropped_img_1 = F.hflip(cropped_img_1)
            cropped_img_2 = F.hflip(cropped_img_2)

        # Convert to float32 tensor + Normalize
        cropped_img_1 = self.post_process(cropped_img_1)
        cropped_img_2 = self.post_process(cropped_img_2)
            
        return cropped_img_1, cropped_img_2

class SiamMAE_VideoDataset(data.Dataset):
    def __init__(self, files, args):

        interpolation = getattr(F.InterpolationMode, args.interpolation_method.upper())

        self.transforms_manager = DoubleRandomResizedCrop(use_same_crop=True, interpolation=interpolation)
        self.args = args
        self.files = files

    def __len__(self,):
        return len(self.files)

    def __getitem__(self, idx):
        video_path = self.files[idx]
        frames = []
        for f1, f2 in sample_frames(video_path, gap_range=[4, 48], repeated_sampling=self.args.repeated_sampling_factor):
            f1, f2 = self.transforms_manager(f1, f2)
            frames.extend([f1.unsqueeze(0), f2.unsqueeze(0)])
        return torch.cat(frames, dim=0)