#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cPickle as pickle
import gzip
import os


def pickle_dump(path, obj, compress=False):
    if compress:
        with gzip.open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(path, compressed=False):
    if compressed:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)


def iter_sub_dirs(root):
    for item in os.listdir(root):
        item = os.path.join(root, item)
        if os.path.isdir(item):
            yield item
