import os
import argparse
import numpy as np
from PIL import Image
import cv2
import sys

n_cl = 20
CLASSES = ['background', 'hat', 'hair', 'sun-glasses', 'upper-clothes', 'dress',
           'coat', 'socks', 'pants', 'gloves', 'scarf', 'skirt', 'torso-skin',
           'face', 'right-arm', 'left-arm', 'right-leg', 'left-leg', 'right-shoe', 'left-shoe']

parser = argparse.ArgumentParser()

parser.add_argument("-g", "--gt_dir", type=str,
                    help="ground truth path")
parser.add_argument("-p", "--pre_dir", type=str,
                    help="prediction path")

def main(args):
    image_paths, label_paths = init_path(args)
    hist = compute_hist(image_paths, label_paths)
    return show_result(hist)

def init_path(args):
    image_dir = args.pre_dir
    label_dir = args.gt_dir

    file_names = []
    for vid in os.listdir(image_dir):
        for img in os.listdir(os.path.join(image_dir, vid)):
            file_names.append([vid, img])
    print ("result of", image_dir)

    image_paths = []
    label_paths = []
    for file_name in file_names:
        if "blend" not in file_name[1]:
            image_paths.append(os.path.join(image_dir, file_name[0], file_name[1]))
            label_paths.append(os.path.join(label_dir, file_name[0], file_name[1].split("_")[0] + ".png"))
    return image_paths, label_paths

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(images, labels):

    hist = np.zeros((n_cl, n_cl))
    for img_path, label_path in zip(images, labels):
        label = Image.open(label_path.replace('.jpg', '.png'))
        label_array = np.array(label, dtype=np.int32)
        image = cv2.imread(img_path)
        image = Image.fromarray(image)
        image_array = np.array(image, dtype=np.int32)

        gtsz = label_array.shape

        imgsz = image_array.shape

        if image_array.max() > 20:
            print(img_path, image_array.max())
            sys.exit()


        if not gtsz == imgsz:
            image = image.resize((gtsz[1], gtsz[0]), Image.NEAREST)
            image_array = np.array(image, dtype=np.int32)

        if len(image_array.shape) == 3:
            image_array = image_array[..., -1]


        hist += fast_hist(label_array, image_array, n_cl)

    return hist

def show_result(hist):

    classes = CLASSES
    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    print ('=' * 50)

    # @evaluation 1: overall accuracy
    acc = num_cor_pix.sum() / hist.sum()
    print ('>>>', 'overall accuracy', acc)
    print ('-' * 50)

    # @evaluation 2: mean accuracy & per-class accuracy
    print ('Accuracy for each class (pixel accuracy):')
    for i in range(n_cl):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / num_gt_pix[i]))
    acc = num_cor_pix / num_gt_pix
    print ('>>>', 'mean accuracy', np.nanmean(acc))
    print ('-' * 50)

    # @evaluation 3: mean IU & per-class IU
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    for i in range(n_cl):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    print ('>>>', 'mean IU', np.nanmean(iu))
    print ('-' * 50)

    # @evaluation 4: frequency weighted IU
    freq = num_gt_pix / hist.sum()
    print ('>>>', 'fwavacc', (freq[freq > 0] * iu[freq > 0]).sum())
    print ('=' * 50)

    return np.nanmean(iu)



if __name__ == '__main__':
    args = parser.parse_args()
    iu = main(args)