from math import floor
from PIL import Image
import random
import glob
import os
import sys
import numpy as np

import matplotlib.pyplot as plt

# check if python3 is used
if sys.version_info < (3, 0):
    print("This programs need python3x to run. Aborting.")
    sys.exit(1)


def initialize(backgrounds_dir, classes_dir):
    # Loading data
    backgrounds = os.listdir(backgrounds_dir)
    class_names = os.listdir(classes_dir)
    class_objects = [os.listdir(os.path.join(classes_dir, c))
                     for c in class_names]

    # ignore scaling for now: .resize(resolution, Image.BICUBIC)
    bgs = [Image.open(os.path.join(backgrounds_dir, b))
           for b in backgrounds if b.endswith(".jpg")]

    objs = []
    for i, c in enumerate(class_objects):
        print('loading '+class_names[i]+' as '+str(i))
        objs = objs + [[Image.open(os.path.join(classes_dir, class_names[i], o))
                        for o in c if o.endswith('.png')]]

    return objs, class_names, bgs


def mkimage(filename, objs, names, bgs, maxobjs, output_dir="images_out", single=False):
    """
    Simulate an image
    """
    log = []
    im = bgs[random.randint(0, len(bgs)-1)].copy()
    seg_ims = [Image.fromarray(np.zeros(
        (im.height, im.width), dtype=np.uint8)).convert("L") for _ in range(len(objs))]
    cls0 = random.randint(0, len(objs)-1)
    for _ in range(0, random.randint(1, maxobjs)):
        if single:
            cls = cls0
        else:
            cls = random.randint(0, len(objs)-1)
        obj = random.choice(objs[cls])
        sizex, sizey = obj.size
        imx, imy = im.size
        posx = random.randint(-floor(sizex/2), imx-floor(sizex/2))
        posy = random.randint(-floor(sizey/2), imy-floor(sizey/2))
        im.paste(obj, (posx, posy), obj)

        obj_array = np.array(obj)
        obj_mask = (obj_array[:, :, 3] != 0)
        obj_mask = np.uint8(obj_mask)*255
        obj_seg = Image.fromarray(obj_mask).convert("L")
        seg_ims[cls].paste(obj_seg, (posx, posy), obj)

        log = log + \
            ['{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                names[cls], cls, posy, posx, posy+sizey, posx+sizex)]
    im.save(os.path.join(output_dir, filename+'.png'))
    for i, seg_im in enumerate(seg_ims):
        seg_im.save(os.path.join(output_dir, f'{filename}_segment{i}.png'))

    with open(os.path.join(output_dir, filename+'.txt'), 'w') as f:
        [f.write(l) for l in log]


def test():
    """
    Testing
    """
    objects, names, backgrounds = initialize(
        backgrounds_dir="backgrounds", classes_dir="crops")
    mkimage('test', objects, names, backgrounds,
            output_dir="images_out", maxobjs=6)
