from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import DataLoader

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
from .utils import to_numpy
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
cudnn.enabled = True
cudnn.benchmark = True

torch.multiprocessing.set_sharing_strategy('file_system')

def gallery_set(gallery,k):
    test_id = [6,10,17,21,24,25,27,28,31,34,36,37,40,41,42,43,44,45,49,50,51,54,63,69,75,80,81,82,83,84,85,86,87,88,89,90,93,102,104,105,106,108,112,116,117,122,125,129,130,134,138,139,150,152,162,166,167,170,172,176,185,190,192,202,204,207,210,215,223,229,232,237,252,253,257,259,263,266,269,272,273,274,275,282,285,291,300,301,302,303,307,312,315,318,331,333]
    test_id[:] = [x-1 for x in test_id]
    blank_list = [0]*len(gallery)
    blank_list2 = []
    new_gallery_set = []
    for i in range(len(gallery)):
        blank_list[i] = gallery[i][1] 
    for i in range(len(test_id)):
        start_index = blank_list.index(test_id[i])
        if i == len(test_id)-1:
            finish_index = len(gallery)-1
        else :
            finish_index = blank_list.index(test_id[i+1])
        if k == 1:
            blank_list2.append(random.randint(start_index,finish_index))
        if k == 10:
            for i in range(10):
                blank_list2.append(random.randint(start_index, finish_index))
    for i in range(len(blank_list2)):
        new_gallery_set.append(gallery[blank_list2[i]])
        
    return new_gallery_set
        
def data_source_split(data_source):
    data_source_rgb = []
    data_source_ir = []
    for i in range(len(data_source)):
        tuple_tmp = data_source[i][2]
        if int(tuple_tmp) == 0 or int(tuple_tmp) == 1 or int(tuple_tmp) == 3 or int(tuple_tmp) == 4 : # All search
     #   if int(tuple_tmp) == 0 or int(tuple_tmp) == 1 : #indoor search
            data_source_rgb.append(data_source[i])
        else :
            data_source_ir.append(data_source[i])
    return data_source_rgb , data_source_ir

def extract_embeddings(model, features, alpha, query=None, topk_gallery=None, rerank_topk=0, print_freq=500):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    pairwise_score = Variable(torch.zeros(len(query), rerank_topk, 2).cuda())
    probe_feature = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)

    with torch.no_grad():
      for i in range(len(query)):
        gallery_feature = torch.cat([features[f].unsqueeze(0) for f, _, _ in topk_gallery[i]], 0)
        pairwise_score[i, :, :] = model(Variable(probe_feature[i].view(1, -1).cuda(), volatile=True),
                                        Variable(gallery_feature.cuda(), volatile=True))
        batch_time.update(time.time() - end)
        end = time.time()

 #       if (i + 1) % print_freq == 0:
 #        print('Extract Embedding: [{}/{}]\t'
 ##              'Time {:.3f} ({:.3f})\t'
  #             'Data {:.3f} ({:.3f})\t'.format(
  #             i + 1, len(query),
   #            batch_time.val, batch_time.avg,
    #           data_time.val, data_time.avg))

    return pairwise_score.view(-1,2)


def extract_features(model, data_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    image_tensor = OrderedDict()

    end = time.time()

    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid , imgs in zip(fnames, outputs, pids , imgs):
            features[fname] = output
            labels[fname] = pid
            image_tensor[fname] = imgs
        batch_time.update(time.time() - end)
        end = time.time()

  #      if (i + 1) % print_freq == 0:
 #           print('Extract Features: [{}/{}]\t'
 #                 'Time {:.3f} ({:.3f})\t'
 #                 'Data {:.3f} ({:.3f})\t'
   #               .format(i + 1, len(data_loader),
    #                      batch_time.val, batch_time.avg,
     #                     data_time.val, data_time.avg))
    return features, labels,image_tensor
def change_output( outputs, outputs_ir, camid):
    for i in range(len(camid)):
        if camid[i].item() == 2 or camid[i].item() == 5:
   #         print(outputs[i])
   #         print(outputs_ir[i])
            outputs[i] = outputs_ir[i]
   #         print("------------")
    #        print(outputs[i])
     #       print(outputs_ir[i])
    return outputs
def extract_features_jh(model_ir,model_rgb, data_loader, print_freq=1, metric=None):
    model_ir.eval()
    model_rgb.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    image_tensor = OrderedDict()

    end = time.time()

    for i, (imgs, fnames, pids, camid) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs_ir = extract_cnn_feature(model_ir,imgs)
        outputs = extract_cnn_feature(model_rgb, imgs)
        outputs = change_output( outputs, outputs_ir, camid)
        for fname, output, pid , imgs in zip(fnames, outputs, pids , imgs):
            features[fname] = output
            labels[fname] = pid
            image_tensor[fname] = imgs
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))
    return features, labels,image_tensor




def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


#def evaluate_all(distmat, query=None, gallery=None,
 #                query_ids=None, gallery_ids=None,
#                 query_cams=None, gallery_cams=None,
 #                cmc_topk=(1, 5, 10), dataset=None, top1=True):
#def evaluate_all(distmat, query=None, gallery=None,
#                     query_ids=None, gallery_ids=None,
#                     query_cams=None, gallery_cams=None,
#                     cmc_topk=(1, 5, 10), dataset=None, top1=True):
def evaluate_all(distmat, query=None, gallery=None,
                     query_ids=None, gallery_ids=None,
                     query_cams=None, gallery_cams=None,
                     cmc_topk=(1, 10, 20), dataset=None, top1=True):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if top1:
      # Compute all kinds of CMC scores
      if not dataset:
        cmc_configs = {
            'allshots': dict(separate_camera_set=False,
                             single_gallery_shot=False,
                             first_match_break=False),
            'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False),
            'market1501': dict(separate_camera_set=False,
                               single_gallery_shot=False,
                               first_match_break=True)}
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                query_cams, gallery_cams, **params)
                      for name, params in cmc_configs.items()}

        print('CMC Scores{:>12}{:>12}{:>12}'
              .format('allshots', 'cuhk03', 'market1501'))
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
                  .format(k, cmc_scores['allshots'][k - 1],
                          cmc_scores['cuhk03'][k - 1],
                          cmc_scores['market1501'][k - 1]))

        # Use the allshots cmc top-1 score for validation criterion
        return cmc_scores['allshots'][0]
      else:

        if (dataset == 'cuhk03'):
          cmc_configs = {
              'cuhk03': dict(separate_camera_set=True,
                                single_gallery_shot=True,
                                first_match_break=False),
              }
          cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                  query_cams, gallery_cams, **params)
                        for name, params in cmc_configs.items()}

          print('CMC Scores{:>12}'.format('cuhk03'))
          for k in cmc_topk:
              print('  top-{:<4}{:12.1%}'
                    .format(k,
                            cmc_scores['cuhk03'][k - 1]))
          # Use the allshots cmc top-1 score for validation criterion
          return cmc_scores['cuhk03'][0], mAP
        elif (dataset =='sysumm01'):
          cmc_configs = {
              'sysumm01': dict(separate_camera_set=False,
                                 single_gallery_shot=False,
                                 first_match_break=True)
                      }
          cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                  query_cams, gallery_cams, **params)
                        for name, params in cmc_configs.items()}

          print('CMC Scores{:>12}'.format('sysumm01'))
          for k in cmc_topk:
              print('  top-{:<4}{:12.1%}'
                    .format(k,
                            cmc_scores['sysumm01'][k-1]))
          return cmc_scores['sysumm01'][0], mAP
        else:
          cmc_configs = {
              'market1501': dict(separate_camera_set=False,
                                 single_gallery_shot=False,
                                 first_match_break=True)
                      }
          cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                  query_cams, gallery_cams, **params)
                        for name, params in cmc_configs.items()}

          print('CMC Scores{:>12}'.format('market1501'))
          for k in cmc_topk:
              print('  top-{:<4}{:12.1%}'
                    .format(k,
                            cmc_scores['market1501'][k-1]))
          return cmc_scores['market1501'][0], mAP
    else:
      return mAP

class CascadeEvaluator(object):
    def __init__(self, base_model, embed_model, embed_dist_fn=None):
        super(CascadeEvaluator, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model
        self.embed_dist_fn = embed_dist_fn

    def evaluate(self, data_loader, query, gallery, alpha=0, cache_file=None,
                 rerank_topk=75, second_stage=True, dataset=None, top1=True):
        # Extract features image by image
        features, _ ,img= extract_features(self.base_model, data_loader)
        gallery ,query =  data_source_split(query)
     #   #######################################
     #   gallery = gallery_set(gallery,1)
     #   #######################################
        # Compute pairwise distance and evaluate for the first stage
        distmat = pairwise_distance(features, query, gallery)
        print("First stage evaluation:")
        if second_stage:
            evaluate_all(distmat, query=query, gallery=gallery, dataset=dataset, top1=top1)

            # Sort according to the first stage distance
            distmat = to_numpy(distmat)
            rank_indices = np.argsort(distmat, axis=1)

            # Build a data loader for topk predictions for each query
            topk_gallery = [[] for i in range(len(query))]
            for i, indices in enumerate(rank_indices):
                for j in indices[:rerank_topk]:
                    gallery_fname_id_pid = gallery[j]
                    topk_gallery[i].append(gallery_fname_id_pid)

            embeddings = extract_embeddings(self.embed_model, features, alpha,
                                    query=query, topk_gallery=topk_gallery, rerank_topk=rerank_topk)

            if self.embed_dist_fn is not None:
                embeddings = self.embed_dist_fn(embeddings.data)

            # Merge two-stage distances
            for k, embed in enumerate(embeddings):
                i, j = k // rerank_topk, k % rerank_topk
                distmat[i, rank_indices[i, j]] = embed
            for i, indices in enumerate(rank_indices):
                bar = max(distmat[i][indices[:rerank_topk]])
                gap = max(bar + 1. - distmat[i, indices[rerank_topk]], 0)
                if gap > 0:
                    distmat[i][indices[rerank_topk:]] += gap
            print("Second stage evaluation:")
        return evaluate_all(distmat, query, gallery, dataset=dataset, top1=top1)
