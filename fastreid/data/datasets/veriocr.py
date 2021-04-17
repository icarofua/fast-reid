# encoding: utf-8
"""
@author:  Icaro Oliveira
@contact: icarofua@gmail.com
"""

import glob
import os.path as osp
import re

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VERIOCR(ImageDataset):
    """VERIOCR.

    Reference:
        de Oliveira, Icaro O., et al. "Vehicle re-identification: exploring feature fusion using multi-stream convolutional networks." arXiv preprint arXiv:1911.05541 (2019).

    URL: `<https://github.com/icarofua/vehicle-ReId>`_

    Dataset statistics:
        - identities: 2654.
        - images: 15860 (train) + 3115 (query) + 7188 (gallery).
    """
    dataset_dir = "veriocr"
    dataset_name = "veriocr"

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        required_files = [
            self.dataset_dir
        ]
        self.check_before_run(required_files)
        self.train_txt = f"{self.dataset_dir}/images_train.txt"
        self.query_txt = f"{self.dataset_dir}/images_query.txt"
        self.gallery_txt = f"{self.dataset_dir}/images_gallery.txt"

        train = self.process_dir(self.train_txt)
        query = self.process_dir(self.query_txt, is_train=False)
        gallery = self.process_dir(self.gallery_txt, is_train=False)

        super(VERIOCR, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, txt_file, is_train=True):
        data = []
        img_list = open(txt_file).readlines()
        for img_path in img_list:
            pid, camid, path = img_path.split(",")
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((osp.join(self.dataset_dir,path[:-1]), pid, camid))

        return data
