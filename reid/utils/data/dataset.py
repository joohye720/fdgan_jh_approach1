from __future__ import print_function
import os.path as osp

import numpy as np

from ..serialization import read_json

def _pluck(identities, indices, relabel=False):
    ret = []
    query = {}
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        if relabel:
            if index not in query.keys():
                query[index] = []
        else:
            if pid not in query.keys():
                query[pid] = []
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                name = osp.splitext(fname)[0]
                #print(name) #.jpg removing
                x, y, _ = map(int, name.split('_')) # change to integer
                #print("%d %d, %d %d"%(pid, x, camid, y))
                assert pid == x and camid == y
                if relabel:
                    ret.append((fname, index, camid))
                    query[index].append(fname)
                else:
                    ret.append((fname, pid, camid))
                    query[pid].append(fname)
    return ret, query

class Dataset(object):
    def __init__(self, root, split_id=0):
        self.root = root
        self.split_id = split_id
        self.meta = None
        self.split = None
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    def __len__(self):
        return

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    @property
    def poses_dir(self):
        return osp.join(self.root, 'poses')

    def load(self, num_val=0.1, verbose=True):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]  # list to dictionary type

        trainval_pids = sorted(np.asarray(self.split['trainval']))
        num = len(trainval_pids) #training identity number
        if isinstance(num_val, float):
            num_val = int(round(num * num_val))
        if num_val >= num or num_val < 0:
            raise ValueError("num_val exceeds total identities {}"
                             .format(num))
        train_pids = sorted(trainval_pids[:-num_val])
        val_pids = sorted(trainval_pids[-num_val:])
        # train_pids and val_pids divided with reference value 100. -->  if train_pids identitiy is 267 --> 167/100 (train_pids and val_pids)
        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']
        #identities, list type ,len = 533 ,identity number , identities[0] --> list type , len = 6 , camera number , identities[0][0] --> list type , len = number of images with specific identity camera
        self.train, self.train_query = _pluck(identities, train_pids, relabel=True)
        self.val, self.val_query = _pluck(identities, val_pids, relabel=True)
        self.trainval, self.trainval_query = _pluck(identities, trainval_pids, relabel=True)
        self.query, self.query_query = _pluck(identities, self.split['query'])
        self.gallery, self.gallery_query = _pluck(identities, self.split['gallery'])
        self.num_train_ids = len(train_pids)
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = len(trainval_pids)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  val      | {:5d} | {:8d}"
                  .format(self.num_val_ids, len(self.val)))
            print("  trainval | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(self.split['query']), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(self.split['gallery']), len(self.gallery)))

    def _check_integrity(self):

        print(osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json')) and \
               osp.isdir(osp.join(self.root, 'poses')))
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json')) and \
               osp.isdir(osp.join(self.root, 'poses'))
