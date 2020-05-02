#ir check
from __future__ import absolute_import
import os.path as osp
import random
import numpy as np
import torch
from scipy import ndimage
from PIL import Image

from reid.utils.data import transforms

def data_source_split(data_source):
    data_source_rgb = []
    data_source_ir = []
    for i in range(len(data_source)):
        tuple_tmp = data_source[i]
        a = tuple_tmp[10]
        if int(tuple_tmp[10]) == 0 or int(tuple_tmp[10]) == 1 or int(tuple_tmp[10]) == 3 or int(tuple_tmp[10]) == 4 :
            data_source_rgb.append(tuple_tmp)
        else :
            data_source_ir.append(tuple_tmp)
    return data_source_rgb , data_source_ir

class Preprocessor(object):
    def __init__(self, dataset, root=None, with_pose=False, pose_root=None, pid_imgs=None, height=256, width=128, pose_aug='no', transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.with_pose = with_pose
        self.pose_root = pose_root
        self.pid_imgs = pid_imgs
        self.height = height
        self.width = width
        self.pose_aug = pose_aug

        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if transform==None:
            self.transform = transforms.Compose([
                                 transforms.RectScale(height, width),
                                 transforms.RandomSizedEarser(),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalizer,
                             ])
        else:
            self.transform = transform
        self.transform_p = transforms.Compose([
                                 transforms.RectScale(height, width),
                                 transforms.ToTensor(),
                                 normalizer,
                             ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            if not self.with_pose:
                return [self._get_single_item(index) for index in indices]
            else:
                return [self._get_single_item_with_pose(index) for index in indices]
        if not self.with_pose:
            return self._get_single_item(indices)
        else:
            return self._get_single_item_with_pose(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)
        return img, fname, pid, camid

    def _get_single_item_with_pose(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)

        pid_query = list(self.pid_imgs[pid])
        if fname in pid_query and len(pid_query)>1:
            pid_query.remove(fname)
        pid_query_rgb, pid_query_ir = data_source_split(pid_query)

        pname = osp.splitext(random.choice(pid_query_ir))[0] # pname = '00000098_05_0013'
      #  print(pname)
            ################################################
        #Pose tansfer dataloader
        ################################################
        ppath = pname+'.txt'
        if self.pose_root is not None:
            ppath = osp.join(self.pose_root, ppath)
        gtpath = pname+'.jpg'
        if self.root is not None:
            gtpath = osp.join(self.root, gtpath)

        gt_img = Image.open(gtpath).convert('RGB')
      #  print(gt_img.size[1],gt_img.size[0])
        landmark = self._load_landmark(ppath, self.height/gt_img.size[1], self.width/gt_img.size[0])
        maps = self._generate_pose_map(landmark)

        flip_flag = random.choice([True, False])
        if flip_flag:
            gt_img = gt_img.transpose(Image.FLIP_LEFT_RIGHT)
            maps = np.flip(maps,2)

        maps = torch.from_numpy(maps.copy()).float()
        gt_img = self.transform_p(gt_img)

        return {'origin': img,
                'target': gt_img,
                'posemap': maps,
                'pid': torch.LongTensor([pid])}

    def _load_landmark(self, path, scale_h, scale_w):
        landmark = []
        with open(path,'r') as f:
            landmark_file = f.readlines()
        for line in landmark_file:
            line1 = line.strip()
            h0 = int(float(line1.split(' ')[0]) * scale_h)
            w0 = int(float(line1.split(' ')[1]) * scale_w)
            if h0<0: h0=-1
            if w0<0: w0=-1
            landmark.append(torch.Tensor([[h0,w0]]))
        landmark = torch.cat(landmark).long()
        return landmark

    def _generate_pose_map(self, landmark, gauss_sigma=5):
        maps = []
        randnum = landmark.size(0)+1
        if self.pose_aug=='erase':
            randnum = random.randrange(landmark.size(0))
        elif self.pose_aug=='gauss':
            gauss_sigma = random.randint(gauss_sigma-1,gauss_sigma+1)
        elif self.pose_aug!='no':
            assert ('Unknown landmark augmentation method, choose from [no|erase|gauss]')
        for i in range(landmark.size(0)):
            map = np.zeros([self.height,self.width])
            if landmark[i,0]!=-1 and landmark[i,1]!=-1 and i!=randnum:

                #print(landmark[i,0],landmark[i,1])
                map[landmark[i,0],landmark[i,1]]=1
                map = ndimage.filters.gaussian_filter(map,sigma = gauss_sigma)
                map = map/map.max()
            maps.append(map)
        maps = np.stack(maps, axis=0)
      #  print("-----------")
        return maps
