from __future__ import print_function, absolute_import

import os
import numpy as np
import math

import cv2
import torch
from matplotlib import cm
from scipy import io as sio

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
    img = cv2.resize( img, (owidth, oheight) )
    img = im_to_torch(img)
    return img

def load_image(img_path):
    # H x W x C => C x H x W
    img = cv2.imread(img_path)
#     print(img_path)
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:,:,::-1]
    img = img.copy()
    return im_to_torch(img)

def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x

import time




######################################################################
def try_np_load(p):
    try:
        return np.load(p)
    except:
        return None

def make_lbl_set(lbls):
    print(lbls.shape)
    t00 = time.time()

    lbl_set = [np.zeros(3).astype(np.uint8)]
    
    flat_lbls_0 = lbls[0].copy().reshape(-1, lbls.shape[-1]).astype(np.uint8)
    lbl_set = np.unique(flat_lbls_0, axis=0)
    
    print('lbls', time.time() - t00)

    return lbl_set

class JhmdbSet(torch.utils.data.Dataset):
    def __init__(self, args, sigma=0.5):

        self.filelist = args.filelist
        self.img_size = args.img_size
        self.video_len = args.video_len
        self.map_scale = args.map_scale

        self.sigma = sigma

        f = open(self.filelist, 'r')
        self.jpgfiles = []
        self.lblfiles = []

        for line in f:
            rows = line.split()
            jpgfile = rows[1]
            lblfile = rows[0]

            self.jpgfiles.append(jpgfile)
            self.lblfiles.append(lblfile)

        f.close()
    
    def get_onehot_lbl(self, lbl_path):
        name = '/'.join(lbl_path.split('.')[:-1]) + '_onehot.npy'
        if os.path.exists(name):
            return np.load(name)
        else:
            return None
    

    def make_paths(self, folder_path):
        I = [ ll for ll in os.listdir(folder_path) if '.png' in ll ]

        frame_num = len(I) + self.video_len
        I.sort(key=lambda x:int(x.split('.')[0]))

        I_out = []

        for i in range(frame_num):
            i = max(0, i - self.video_len)
            img_path = "%s/%s" % (folder_path, I[i])

            I_out.append(img_path)

        return I_out


    def __getitem__(self, index):

        folder_path = self.jpgfiles[index]
        label_path = self.lblfiles[index]

        imgs = []
        imgs_orig = []
        lbls = []
        
        img_paths = self.make_paths(folder_path)
        frame_num = len(img_paths)

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        # frame_num = 30
        for i in range(frame_num):

            img_path = img_paths[i]
            img = load_image(img_path)  # CxHxW

            # print('loaded', i, time.time() - t00)

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

            img_orig = img.clone()
            img = color_normalize(img, mean, std)

            imgs_orig.append(img_orig)
            imgs.append(img)

        rsz_h, rsz_w = math.ceil(img.size(1) / self.map_scale[0]), math.ceil(img.size(2) /self.map_scale[1])

        lbls_mat = sio.loadmat(label_path)

        lbls_coord = lbls_mat['pos_img']
        lbls_coord = lbls_coord - 1

        lbls_coord[0, :, :] = lbls_coord[0, :, :] * float(new_w) / float(ori_w) / self.map_scale[0]
        lbls_coord[1, :, :] = lbls_coord[1, :, :] * float(new_h) / float(ori_h) / self.map_scale[1]
        lblsize =  (rsz_h, rsz_w)

        lbls = np.zeros((lbls_coord.shape[2], lblsize[0], lblsize[1], lbls_coord.shape[1]))

        for k in range(lbls_coord.shape[2]):
            lbls_coord_now = lbls_coord[:, :, k]

            for j in range(lbls_coord.shape[1]):
                if self.sigma > 0:
                    draw_labelmap_np(lbls[k, :, :, j], lbls_coord_now[:, j], self.sigma)
                else:
                    tx = int(lbls_coord_now[0, j])
                    ty = int(lbls_coord_now[1, j])
                    if tx < lblsize[1] and ty < lblsize[0] and tx >=0 and ty >=0:
                        lbls[k, ty, tx, j] = 1.0

        lbls_tensor = torch.zeros(frame_num, lblsize[0], lblsize[1], lbls_coord.shape[1])

        for j in range(frame_num):
            if j < self.video_len:
                nowlbl = lbls[0]
            else:
                if(j - self.video_len < len(lbls)):
                    nowlbl = lbls[j - self.video_len]
            lbls_tensor[j] = torch.from_numpy(nowlbl)
        
        lbls_tensor = torch.cat([(lbls_tensor.sum(-1) == 0)[..., None] *1.0, lbls_tensor], dim=-1)

        lblset = np.arange(lbls_tensor.shape[-1]-1)
        lblset = np.array([[0, 0, 0]] + [cm.Paired(i)[:3] for i in lblset]) * 255.0

        # Meta info
        meta = dict(folder_path=folder_path, img_paths=img_paths, lbl_paths=[])
        
        imgs = torch.stack(imgs)
        imgs_orig = torch.stack(imgs_orig)
        lbls_resize = lbls_tensor #np.stack(resizes)

        assert lbls_resize.shape[0] == len(meta['img_paths'])

        return imgs, imgs_orig, lbls_resize, lbls_tensor, lblset, meta

    def __len__(self):
        return len(self.jpgfiles)


def draw_labelmap_np(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img
