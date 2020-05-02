from __future__ import absolute_import
from collections import defaultdict
import random
import numpy as np
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)

def _choose_from(start, end, excluded_range=None, size=1, replace=False):
    num = end - start + 1
    if excluded_range is None:
        return np.random.choice(num, size=size, replace=replace) + start
    ex_start, ex_end = excluded_range
    num_ex = ex_end - ex_start + 1
    num -= num_ex
    inds = np.random.choice(num, size=size, replace=replace) + start
    inds += (inds >= ex_start) * num_ex
    return inds

def data_source_split(data_source):
    data_source_rgb = []
    data_source_ir = []
    for i in range(len(data_source)):
        tuple_tmp = data_source[i]
        if tuple_tmp[2] == 0 or tuple_tmp[2] == 1 or tuple_tmp[2] == 3 or tuple_tmp[2] == 4 :
            data_source_rgb.append(tuple_tmp)
        else :
            data_source_ir.append(tuple_tmp)
    return data_source_rgb , data_source_ir

class RandomPairSampler(Sampler):
    def __init__(self, data_source, neg_pos_ratio=1):
        super(RandomPairSampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.neg_pos_ratio = neg_pos_ratio
        # Sort by pid
        ############################################
        self.data_source_rgb , self.data_source_ir= data_source_split(self.data_source)
        ############################################
        self.num_samples_rgb = len(self.data_source_rgb)
        self.num_samples_ir = len(self.data_source_ir)
        indices_rgb = np.argsort(np.asarray(self.data_source_rgb)[:, 1])
        self.index_map_rgb = dict(zip(np.arange(self.num_samples_rgb), indices_rgb))
        self.index_range_rgb = defaultdict(lambda: [self.num_samples_rgb, -1])
        for i, j in enumerate(indices_rgb):
            _, pid_rgb, _ = self.data_source_rgb[j]
            self.index_range_rgb[pid_rgb][0] = min(self.index_range_rgb[pid_rgb][0], i)
            self.index_range_rgb[pid_rgb][1] = max(self.index_range_rgb[pid_rgb][1], i)
        ############################################
        indices_ir = np.argsort(np.asarray(self.data_source_ir)[:, 1])
        self.index_map_ir = dict(zip(np.arange(self.num_samples_ir), indices_ir))
        self.index_range_ir = defaultdict(lambda: [self.num_samples_ir, -1])
        for i, j in enumerate(indices_ir):
            _, pid_ir, _ = self.data_source_ir[j]
            self.index_range_ir[pid_ir][0] = min(self.index_range_ir[pid_ir][0], i)
            self.index_range_ir[pid_ir][1] = max(self.index_range_ir[pid_ir][1], i)
        ############################################
        indices = np.argsort(np.asarray(data_source)[:, 1])
        self.index_map = dict(zip(np.arange(self.num_samples), indices))

    def __iter__(self):
        indices = np.random.permutation(self.num_samples)
        for i in indices:
            # anchor sample
            anchor_index = self.index_map[i]
#o            _, pid, _ = self.data_source[anchor_index]

            _, pid, anchor_cam_id = self.data_source[anchor_index]
            while self.index_range_ir[pid] == [len(self.index_map_ir),-1] or self.index_range_rgb[pid] == [len(self.index_map_rgb),-1]:
                new_i = random.randrange(50,self.num_samples-1)
                anchor_index = self.index_map[new_i]
                _, pid, anchor_cam_id = self.data_source[anchor_index]
            ##j####################################################################################
            if anchor_cam_id == 0 or anchor_cam_id == 1 or anchor_cam_id == 3 or anchor_cam_id == 4 :
                start_ir, end_ir = self.index_range_ir[pid]
                pos_index_ir = _choose_from(start_ir, end_ir, excluded_range = None)[0]
                (imagename_pos, pid_pos, camid_pos) = self.data_source_ir[self.index_map_ir[pos_index_ir]]
                positive_index = self.data_source.index((imagename_pos, pid_pos, camid_pos))
 #               yield anchor_index, self.index_map_ir[pos_index_ir]
                yield anchor_index, positive_index
                # negative samples
                neg_indices_ir= _choose_from(0, self.num_samples_ir - 1, excluded_range=(start_ir, end_ir), size=self.neg_pos_ratio)

                for neg_index_ir in neg_indices_ir:
                    (imagename_neg, pid_neg, camid_neg) = self.data_source_ir[self.index_map_ir[neg_index_ir]]
                    negative_index = self.data_source.index((imagename_neg, pid_neg, camid_neg))
                    yield anchor_index, negative_index
            else :
                start_rgb, end_rgb = self.index_range_rgb[pid]

                pos_index_rgb = _choose_from(start_rgb, end_rgb, excluded_range=None)[0]
                (imagename_pos, pid_pos, camid_pos) = self.data_source_rgb[self.index_map_rgb[pos_index_rgb]]
                positive_index = self.data_source.index((imagename_pos, pid_pos, camid_pos))
                #               yield anchor_index, self.index_map_ir[pos_index_ir]
                yield anchor_index, positive_index
                # negative samples
                neg_indices_rgb = _choose_from(0, self.num_samples_rgb - 1, excluded_range=(start_rgb, end_rgb),
                                              size=self.neg_pos_ratio)

                for neg_index_rgb in neg_indices_rgb:
                    (imagename_neg, pid_neg, camid_neg) = self.data_source_rgb[self.index_map_rgb[neg_index_rgb]]
                    negative_index = self.data_source.index((imagename_neg, pid_neg, camid_neg))
                    yield anchor_index, negative_index

    def __len__(self):
        return self.num_samples * (1 + self.neg_pos_ratio)
'''
#################################################################
        # Get the range of indices for each pid
   #     self.index_range = defaultdict(lambda: [self.num_samples, -1])
   #     for i, j in enumerate(indices):
   ##         _, pid, _ = data_source[j]
   #         self.index_range[pid][0] = min(self.index_range[pid][0], i)
   #         self.index_range[pid][1] = max(self.index_range[pid][1], i)
##################################################################

            # positive sample
         #####################################################################################
   #         start, end = self.index_range[pid]
    #        pos_index = _choose_from(start, end,  excluded_range=(i, i) )[0]
    #        yield anchor_index, self.index_map[pos_index]

            # negative samples
     #       neg_indices = _choose_from(0, self.num_samples - 1, excluded_range=(start, end),size=self.neg_pos_ratio)
      #      for neg_index in neg_indices:
       #         yield anchor_index, self.index_map[neg_index]
         #####################################################################################
                #####################################################################################
                #            pos_index = _choose_from_jhversion_positive(start, end,  anchor_cam_id ,self.data_source,  self.index_map ,size=1, replace=False,  excluded_range=(i, i))[0] #j
                # j           positive_index = self.index_map[pos_index]
                # j          _, positive_pid, positive_cam_id = self.data_source[positive_index ]


    def _choose_from_jhversion_positive(start, end, anchor_cam_id=None, data_source=None, index_map=None, size=1,
                                        replace=False, excluded_range=None):
        num = end - start + 1
        if excluded_range is None:
            return np.random.choice(num, size=size, replace=replace) + start
        ex_start, ex_end = excluded_range
        num_ex = ex_end - ex_start + 1
        num -= num_ex
        if anchor_cam_id == 0 or anchor_cam_id == 1 or anchor_cam_id == 3 or anchor_cam_id == 4:
            a = True
            while a:
                inds = np.random.choice(num, size=size, replace=replace) + start
                inds += (inds >= ex_start) * num_ex
                positive_index = index_map[inds[0]]
                _, positive_pid, positive_cam_id = data_source[positive_index]
                if positive_cam_id == 2 or positive_cam_id == 5:
                    a = False
                else:
                    a = True
        else:
            a = True
            while a:
                inds = np.random.choice(num, size=size, replace=replace) + start
                inds += (inds >= ex_start) * num_ex
                positive_index = index_map[inds[0]]
                _, positive_pid, positive_cam_id = data_source[positive_index]
                if positive_cam_id == 0 or positive_cam_id == 1 or positive_cam_id == 3 or positive_cam_id == 4:
                    a = False
                else:
                    a = True
        return inds
'''