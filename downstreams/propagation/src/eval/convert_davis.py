import os
import numpy as np
import cv2

from PIL import Image

from .palette import tensor as palette_tensor
import argparse
import multiprocessing as mp
from itertools import repeat

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_folder', type=str)
    parser.add_argument('-i', '--in_folder', type=str)
    parser.add_argument('-d', '--dataset', type=str)

    args = parser.parse_args()
    return args

def create_directories(dataset, out_folder, in_folder, subset='val'):
    
    jpglist = []

    annotations_folder = dataset + '/Annotations/480p/'
    f1 = open(dataset + f'/ImageSets/2017/{subset}.txt', 'r')
    for line in f1:
        line = line[:-1]
        jpglist.append(line)
    f1.close()

    out_folder = out_folder
    current_folder = in_folder

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    return annotations_folder, out_folder, current_folder, jpglist

palette = palette_tensor.astype(np.uint8)

def color2id(c):
    return np.arange(0, palette.shape[0])[np.all(palette == c, axis=-1)]

def convert_dir(i, annotations_folder, out_folder, current_folder, jpglist):

    print(f"Processing {i} / {len(jpglist)}")

    fname = jpglist[i]
    gtfolder = annotations_folder + fname + '/'
    outfolder = out_folder + fname + '/'

    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    files = [_ for _ in os.listdir(gtfolder) if _[-4:] == '.png']

    lblimg  = cv2.imread(gtfolder + "{:05d}.png".format(0))
    height = lblimg.shape[0]
    width  = lblimg.shape[1]

    for j in range(len(files)):
        outname = outfolder + "{:05d}.png".format(j)
        inname  = current_folder + str(i) + '_' + str(j) + '_mask.png'

        lblimg  = cv2.imread(inname)
        flat_lblimg = lblimg.reshape(-1, 3)
        lblidx  = np.zeros((lblimg.shape[0], lblimg.shape[1]))
        lblidx2  = np.zeros((lblimg.shape[0], lblimg.shape[1]))

        colors = np.unique(flat_lblimg, axis=0)

        for c in colors:
            cid = color2id(c)
            if len(cid) > 0:
                lblidx2[np.all(lblimg == c, axis=-1)] = cid

        lblidx = lblidx2

        lblidx = lblidx.astype(np.uint8)
        lblidx = cv2.resize(lblidx, (width, height), interpolation=cv2.INTER_NEAREST)
        lblidx = lblidx.astype(np.uint8)

        im = Image.fromarray(lblidx)
        im.putpalette(palette.ravel())
        im.save(outname, format='PNG')

def main(args):

    annotations_folder, out_folder, current_folder, jpglist = create_directories(args.dataset, args.out_folder, args.in_folder, args.set)
    print(f"TOTAL NUM IS: {len(jpglist)}")

    pool = mp.Pool(8)
    results = pool.starmap(convert_dir, 
                           zip(
                                range(len(jpglist)),
                                repeat(annotations_folder),
                                repeat(out_folder),
                                repeat(current_folder),
                                repeat(jpglist)
                            )
                        )

if __name__ == "__main__":
    args = parse_args()
    main(args)