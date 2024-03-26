import torch
import torchvision.transforms.v2 as T2
from torchvision.transforms.v2 import functional as F
from .image_utils import RRC

class GlobalToLocal:
    """
    Reconstructs the local view from the global view.
    """

    def __init__(self, args):

        super().__init__()

        interpolation = getattr(F.InterpolationMode, args.interpolation_method.upper())

        self.crop_global = RRC(
            size=(args.input_size, args.input_size),
            scale=(args.random_area_min_global, args.random_area_max_global),
            ratio=(args.random_aspect_ratio_min_global, args.random_aspect_ratio_max_global),
            interpolation=interpolation,
            antialias=True
        )

        self.crop_local = RRC(
            size=(args.input_size, args.input_size),
            scale=(args.random_area_min_local, args.random_area_max_local),
            ratio=(args.random_aspect_ratio_min_local, args.random_aspect_ratio_max_local),
            interpolation=interpolation,
            antialias=True
        )

        self.transforms = T2.Compose([
            T2.RandomHorizontalFlip(p=args.horizontal_flip_p),
            T2.ToImage(),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        add_trans = []
        if args.use_color_jitter:
            add_trans.append(T2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
        if args.use_gaussian_blur:
            add_trans.append(T2.GaussianBlur(kernel_size=12, sigma=(0.1, 2.0)))
        if args.use_elastic_transform:
            add_trans.append(T2.ElasticTransform())
        if len(add_trans) > 0:
            self.add_trans = T2.Compose(add_trans)
        else:
            self.add_trans = None

    def _create_views(self, img):
        img_global = self.crop_global(img)
        img_local = self.crop_local(img_global)
        return img_global, img_local

    def _process_views(self, view1, view2):
        view1 = self.transforms(view1)
        view2 = self.transforms(view2)
        return view1, view2
    
    def _merge_views(self, global_view, local_view):
        return torch.cat([global_view.unsqueeze(0), local_view.unsqueeze(0)], dim=0)
    
    def __call__(self, img):
        img_global, img_local = self._create_views(img)
        img_global, img_local = self._process_views(img_global, img_local)
        if self.add_trans is not None:
            img_global = self.add_trans(img_global)
        return self._merge_views(img_global, img_local)

class LocalToGlobal(GlobalToLocal):
    """
    Reconstructs the global view from the local view.
    """
    
    def _merge_views(self, global_view, local_view):
        return super()._merge_views(local_view, global_view) # Invert order
    
class RandomViews(GlobalToLocal):
    """
    Performs two random crops and reconstruct one from the other.
    """

    def _create_views(self, img):
        img1 = self.crop_global(img)
        img2 = self.crop_global(img)
        return img1, img2

class SameViews(GlobalToLocal):
    """
    Reconstrucs the view from itself.
    """

    def _create_views(self, img):
        img1 = self.crop_global(img)
        img1 = self.transforms(img1)
        return img1, img1
    
    def _process_views(self, view1, view2):
        return view1, view2 # Processing is done is the create_views method