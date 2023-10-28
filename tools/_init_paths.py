# -*- encoding: utf-8 -*-
#Time        :2021/03/03 13:15:10
#Author      :Chen
#FileName    :_init_paths.py
#Version     :1.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, '/home/cyang/EndoCV')
add_path(lib_path)