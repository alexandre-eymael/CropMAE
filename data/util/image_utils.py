
import math
import torch
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as T2

class RRC(T2.RandomResizedCrop):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """
    @staticmethod
    def get_params(img, scale, ratio):
        width, height = F._get_image_size(img)
        area = height * width

        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()

        return i, j, h, w

class DoubleRRC(RRC):
    """
    Perform a random resized crop on two images.
    The crops use the same parameters if `same_crop` is True, otherwise they are resampled.
    """
    def __call__(self, img1, img2, same_crop=True):
        i, j, h, w = self.get_params(img1, scale=self.scale, ratio=self.ratio)
        crop1 = F.resized_crop(img1, i, j, h, w, self.size, self.interpolation)
        if not same_crop: # Resample parameters if a different crop is desired
            i, j, h, w = self.get_params(img2, scale=self.scale, ratio=self.ratio)
        crop2 = F.resized_crop(img2, i, j, h, w, self.size, self.interpolation)
        return crop1, crop2