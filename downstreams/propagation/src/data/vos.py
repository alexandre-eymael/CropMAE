from __future__ import print_function, absolute_import

import os
import numpy as np
import math
import cv2
import torch

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    return img

def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    img = cv2.resize(img, (owidth, oheight))
    img = im_to_torch(img)
    return img

def load_image(img_path):
    # H x W x C => C x H x W
    img = cv2.imread(img_path)
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = img.copy()
    return im_to_torch(img)

def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x

######################################################################
def try_np_load(p):
    try:
        return np.load(p)
    except:
        return None

def make_lbl_set(lbls):
    lbl_set = [np.zeros(3).astype(np.uint8)]   
    
    flat_lbls_0 = lbls[0].copy().reshape(-1, lbls.shape[-1]).astype(np.uint8)
    lbl_set = np.unique(flat_lbls_0, axis=0)

    return lbl_set


class VOSDataset(torch.utils.data.Dataset):
    def __init__(self, args):

        self.filelist = args.filelist
        self.img_size = args.img_size
        self.video_len = args.video_len
        self.map_scale = args.map_scale

        f = open(self.filelist, 'r')
        self.jpgfiles = []
        self.lblfiles = []

        for line in f:
            rows = line.split()
            jpgfile = rows[0]
            lblfile = rows[1]

            self.jpgfiles.append(jpgfile)
            self.lblfiles.append(lblfile)

        f.close()

    def make_paths(self, folder_path, label_path):
        I, L = os.listdir(folder_path), os.listdir(label_path)
        L = [ll for ll in L if 'npy' not in ll]

        frame_num = len(I) + self.video_len
        I.sort(key=lambda x:int(x.split('.')[0]))
        L.sort(key=lambda x:int(x.split('.')[0]))

        I_out, L_out = [], []

        for i in range(frame_num):
            i = max(0, i - self.video_len)
            img_path = f"{folder_path}/{I[i]}"
            lbl_path = f"{label_path}/{L[i]}"

            I_out.append(img_path)
            L_out.append(lbl_path)

        return I_out, L_out


    def __getitem__(self, index):

        folder_path = self.jpgfiles[index]
        label_path = self.lblfiles[index]

        imgs = []
        imgs_orig = []
        lbls = []

        frame_num = len(os.listdir(folder_path)) + self.video_len

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        img_paths, lbl_paths = self.make_paths(folder_path, label_path)

        for i in range(frame_num):

            img_path, lbl_path = img_paths[i], lbl_paths[i]
            img = load_image(img_path)  # CxHxW
            lblimg = cv2.imread(lbl_path)

            ori_h, ori_w = img.size(1), img.size(2)
            if len(self.img_size) == 1:
                if (ori_w > ori_h):
                    new_w = self.img_size[0]
                    new_h = int(ori_h * new_w / ori_w)
                    new_h = int((new_h // 64) * 64)
                else:
                    new_h = self.img_size[0]
                    new_w = int(ori_w * new_h / ori_h)
                    new_w = int((new_w // 64) * 64)
            else:
                new_h, new_w = self.img_size

            if new_h != ori_h or new_w != ori_w:
                img = resize(img, new_w, new_h)
                lblimg = cv2.resize(lblimg, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            img_orig = img.clone()

            img = color_normalize(img, mean, std)

            imgs_orig.append(img_orig)
            imgs.append(img)
            lbls.append(lblimg.copy())
            
        # Meta info
        meta = dict(folder_path=folder_path, img_paths=img_paths, lbl_paths=lbl_paths)

        ########################################################
        # Load reshaped label information (load cached versions if possible)
        lbls = np.stack(lbls)
        prefix = '/'.join(lbl_paths[0].split('.')[:-1])

        # Get lblset
        lblset = make_lbl_set(lbls)

        if np.all((lblset[1:] - lblset[:-1]) == 1):
            lblset = lblset[:, 0:1]

        resizes = []

        rsz_h, rsz_w = math.ceil(img.size(1) / self.map_scale[0]), math.ceil(img.size(2) / self.map_scale[1])

        for i,p in enumerate(lbl_paths):
            prefix = '/'.join(p.split('.')[:-1])
            oh_path = f"{prefix}_onehot.npy"
            rz_path = f"{prefix}_size{rsz_h}x{rsz_w}.npy"

            onehot = try_np_load(oh_path) 
            if onehot is None:
                print('computing onehot lbl for', oh_path)
                onehot = np.stack([np.all(lbls[i] == ll, axis=-1) for ll in lblset], axis=-1)
                np.save(oh_path, onehot)

            resized = try_np_load(rz_path)
            if resized is None:
                print('computing resized lbl for', rz_path)
                resized = cv2.resize(np.float32(onehot), (rsz_w, rsz_h), interpolation=cv2.INTER_LINEAR)
                np.save(rz_path, resized)

            resizes.append(resized)

        ########################################################
        
        imgs = torch.stack(imgs)
        imgs_orig = torch.stack(imgs_orig)
        lbls_tensor = torch.from_numpy(np.stack(lbls))
        lbls_resize = np.stack(resizes)

        assert lbls_resize.shape[0] == len(meta['lbl_paths'])

        return imgs, imgs_orig, lbls_resize, lbls_tensor, lblset, meta

    def __len__(self):
        return len(self.jpgfiles)