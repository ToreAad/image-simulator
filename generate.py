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


mask_counter = 0
def mkimage(filename, objs, names, bgs, maxobjs, output_dir="images_out", single=False):
    """
    Simulate an image
    """
    global mask_counter
    log = []
    im = bgs[random.randint(0, len(bgs)-1)].copy()
    cls0 = random.randint(0, len(objs)-1)

    rgb_folder = os.path.join(output_dir, "rgb")
    mask_folder = os.path.join(output_dir, "masks")

    for d in [rgb_folder, mask_folder]:
        if not os.path.exists(d):
            os.makedirs(d)

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
        mask_path = os.path.join(mask_folder, f"mask_{mask_counter}.png") 
        mask_counter += 1
        obj_seg.resize((28,28)).save(mask_path)

        log.append(f'{names[cls]}\t{cls}\t{posy}\t{posx}\t{posy+sizey}\t{posx+sizex}\t{mask_path}\n')

    im.resize((256,256)).save(os.path.join(rgb_folder, f'{filename}.png'))

    logfile_path = os.path.join(output_dir, f'{filename}.txt')
    with open(logfile_path, 'w') as f:
        [f.write(l) for l in log]

    csv_file = os.path.join(output_dir, "contents.csv")
    
    if not os.path.exists(csv_file):
        with open(csv_file, 'w+') as f:
            f.write('log_file\n')

    with open(csv_file, 'a') as f:
        f.write(f"{logfile_path}\n")
    


def test():
    """
    Testing
    """
    objects, names, backgrounds = initialize(
        backgrounds_dir="backgrounds", classes_dir="crops")
    mkimage('test', objects, names, backgrounds,
            output_dir="images_out", maxobjs=6)
