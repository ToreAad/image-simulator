"""
Adapted from code of Pierre Gillot
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import os
import sys
import numpy as np
import math
import tensorflow as tf

from PIL import Image
from tqdm import tqdm

from typing import Dict, List

def _int64_feature(value : int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value : bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def partition(l : List, n : int) -> None:
    for i in range(0, len(l), n):
        yield l[i:min(i + n, len(l))]

def main(output_dir, input_dir, shards):
    csv_file = os.path.join(input_dir, 'contents.csv')
    assert os.path.exists(csv_file), f"Couldn't find the dataset at {input_dir}"

    with open(csv_file, 'r') as f:
        data = f.readlines()

    labels = data[1].split()
    features = []
    for feature in data[1:]:
        example = {}
        for lbl, f_val in zip(labels, feature.split(',')):
            example[lbl] = f_val
        features.append(example)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    n = int(len(features)/shards)
    for i, chunk in enumerate(partition(features, n)):
        tf_record_path = os.path.join(output_dir, f"shard{i}.tfrecords")
        with tf.python_io.TFRecordWriter(tf_record_path) as TFrecords_writer:
            for feature in tqdm(chunk):
                tf_feature = {}
                for lbl, val in feature.items():
                    with Image.open(val) as im:
                        np_raw = np.array(im).tostring()
                        tf_feature[lbl] = _bytes_feature(tf.compat.as_bytes(np_raw))
                example = tf.train.Example(features=tf.train.Features(feature=tf_feature))
                TFrecords_writer.write(example.SerializeToString())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--shards')
    args = parser.parse_args()
    main(output_dir = args.output_dir, input_dir = args.input_dir, shards = int(args.shards))