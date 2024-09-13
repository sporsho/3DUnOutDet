import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import yaml
import struct
import torch
from sklearn.metrics import multilabel_confusion_matrix
from torch.utils import data
import os
import glob
# from sklearn.neighbors import KDTree, NearestNeighbors
import pickle
import cupy as cp
from cuml.neighbors import NearestNeighbors
import random

from tqdm.auto import tqdm
import open3d as o3d
import pandas as pd
def configure_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def evaluate_cm(cm, class_name):
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    iou1 = tp / (tp + fp + fn)
    f1 = 2 * recall * precision / (precision + recall)
    print(f'Class: {class_name}, Precision:{precision}, Recall: {recall}, IOU: {iou1}, F1: {f1}')
    return iou1
def estimate_plane(pcd):
    tpc = o3d.geometry.PointCloud()
    tpc.points = o3d.utility.Vector3dVector(pcd.copy())
    plane_model, inliers = tpc.segment_plane(distance_threshold=0.1, ransac_n=9, num_iterations=50000)
    return plane_model

def collate_fn_cp(data):
    data2stack = np.stack([d['data'] for d in data]).astype(np.float32)
    dist2stack = np.stack([d['dist'] for d in data]).astype(np.float32)
    ind2stack = np.stack([d['ind'] for d in data]).astype(np.int64)
    label2stack = np.stack([d['label'] for d in data])

    return {'data': torch.from_numpy(data2stack, ), 'label': torch.from_numpy(label2stack),
            'dist': torch.from_numpy(dist2stack), 'ind': torch.from_numpy(ind2stack)}


def segment_plane(plane_eqn, pcd, threshold):
    a, b, c, d = plane_eqn
    x = pcd[:, 0]
    y = pcd[:, 1]
    z = pcd[:, 2]
    normalizer = np.sqrt(a * a + b * b + c * c)
    pq = np.abs(a * x + b * y + c * z + d)
    D = pq / normalizer
    inliers = np.where(D <= threshold)[0]
    return inliers

def collate_fn_cp_inference(data):
    data2stack = np.stack([d['data'] for d in data]).astype(np.float32)
    dist2stack = np.stack([d['dist'] for d in data]).astype(np.float32)
    ind2stack = np.stack([d['ind'] for d in data]).astype(np.int64)
    # we do not need label during inference, but it does not hurt to provide one
    label2stack = np.stack([d['label'] for d in data])

    return {'data': torch.from_numpy(data2stack, ), 'label': torch.from_numpy(label2stack),
            'dist': torch.as_tensor(dist2stack), 'ind': torch.as_tensor(ind2stack)}


def get_files(folder, ext):
    files = glob.glob(os.path.join(folder, f"*.{ext}"))
    return files
class WadsPointCloudDataset(data.Dataset):
    def __init__(self, device, data_path, imageset='train', label_conf='wads.yaml', k=121, leaf_size=100, mean=[0.3420934,  -0.01516175 ,-0.5889243 ,  9.875928  ], std=[25.845459,  18.93466,    1.5863657, 14.734034 ],
                 shuffle_indices=False, save_ind=True, recalculate=False, desnow_root=None, pred_folder=None,
                                         snow_label=None, single_shot=False):
        self.device = device
        self.recalculate = recalculate
        self.k = k
        self.leaf_size = leaf_size
        self.save_ind = save_ind
        self.mean = np.array(mean)
        self.std = np.array(std)
        with open(label_conf, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.shuffle_indices = shuffle_indices
        self.desnow_root = desnow_root
        self.pred_folder = pred_folder
        self.snow_label = snow_label
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        elif imageset == 'all':
            split = semkittiyaml['split']['train'] + semkittiyaml['split']['valid'] + semkittiyaml['split']['test']
        elif imageset == 'bug':
            split = ["11"]
        elif imageset == 'pug':
            split = ["12"]
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.pred_idx = list()
        for i_folder in split:
            self.im_idx += get_files('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']), 'bin')
            if desnow_root is not None:
                assert os.path.exists(desnow_root)
                self.pred_idx += get_files('/'.join([desnow_root, str(i_folder).zfill(2), pred_folder]), 'label')

        self.im_idx.sort()
        if single_shot:
            self.im_idx = self.im_idx[:1]
        if desnow_root is not None:
            self.pred_idx.sort()


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                     dtype=np.int32).reshape((-1, 1))
        annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
        if self.imageset == 'train':
            if np.random.random() > 0.5:
                rotate_rad = np.deg2rad(np.random.random()*360)
                cos, sine = np.cos(rotate_rad), np.sin(rotate_rad)
                rot_mat = np.matrix([[cos, sine], [-sine, cos]])
                # rotate x and y
                data[:, :2] = np.dot(data[:, :2], rot_mat)
            # if np.random.uniform(high=1.0, low=0.0) >= 0.5:
            #     data[:, 0] *= -1
            # if np.random.uniform(high=1.0, low=0.0) >= 0.5:
            #     data[:, 1] *= -1
            #
            # if np.random.uniform(high=1.0, low=0.0) >= 0.5:
            #     data[:, 2] *= -1
        # data = raw_data

        if self.desnow_root is not None:
            if self.pred_idx[index].endswith('.pred'):
                preds = np.fromfile(self.pred_idx[index], dtype=np.int64)
            else:
                preds = np.fromfile(self.pred_idx[index], dtype=np.int32)
            preds = preds.reshape(-1)
            snow_indices = np.where(preds == self.snow_label)[0]

            data = np.delete(data, obj=snow_indices.tolist(), axis=0)
            annotated_data = np.delete(annotated_data, obj=snow_indices.tolist(), axis=0)
        kd_path = self.im_idx[index].replace('velodyne', 'knn')[:-3] + 'pkl'
        # err = True
        if os.path.exists(kd_path) and not self.recalculate:
            with open(kd_path, 'rb') as f:
                try:
                    ind = pickle.load(f)
                    dist = pickle.load(f)
                    # if ind.shape[1] > self.k:
                    #     ind = ind[:, :self.k]
                    #     dist = dist[:, :self.k]
                    err = False
                except EOFError:
                    err = True
        else:
            err = True
        if err or self.recalculate:
            p1 = torch.from_numpy(data[:, :3]).to(self.device)
            p1 = cp.asarray(p1)
            # metric: string(default='euclidean').
            # Supported
            # distances
            # are['l1, '
            # cityblock
            # ',
            # 'taxicab', 'manhattan', 'euclidean', 'l2', 'braycurtis', 'canberra',
            # 'minkowski', 'chebyshev', 'jensenshannon', 'cosine', 'correlation']
            self.nn = NearestNeighbors()
            while True:

                try:
                    self.nn.fit(p1)
                    break
                except Exception:
                    print("caught it")

            dist, ind = self.nn.kneighbors(p1, self.k)
            # ['euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity']
            # tree = KDTree(data[:, :3], leaf_size=self.leaf_size, metric='cityblock')


            # ind, dist = tree.query_radius(data[:,:3], r=0.5, return_distance=True)
            # process radius and ind for dist query
            # dist = uneven_stack(dist, limit=self.k)
            # ind = uneven_stack(ind, limit=self.k)
            # dist, ind = tree.query(data[:, :3], k=self.k)
            ind = cp.asnumpy(ind)
            dist = cp.asnumpy(dist)
            ind = ind.astype(np.int64)
            # dist = dist.reshape(data.shape[0], -1)
            if self.save_ind:
                parent = os.path.dirname(kd_path)
                os.makedirs(parent, exist_ok=True)
                with open(kd_path, 'wb') as f:
                    pickle.dump(ind, f)
                    pickle.dump(dist, f)
        dist = dist + 1.0
        # normalize the distance
        # d_mean = np.mean(dist, axis=1, keepdims=True)
        # d_std = np.std(dist, axis=1, keepdims=True)
        # dist = (dist - d_mean) / d_std
        if self.shuffle_indices:
            s_ind = np.random.rand(*ind.shape).argsort(axis=1)
            ind = np.take_along_axis(ind, s_ind, axis=1)
            dist = np.take_along_axis(dist, s_ind, axis=1)


        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data).reshape(-1)
        # data = (data - self.mean) / self.std

        if self.imageset == 'train':
            if np.random.random() > 0.5:
                data[:, :3] += np.random.normal(size=(data.shape[0], 3), loc=0, scale=0.1)
        out_dict = {'data': data.astype(np.float32), 'dist': dist.astype(np.float32), 'ind': ind, 'label': annotated_data.astype(np.uint8)}
        return out_dict
def main(args):
    device = torch.device(args.device)
    # gpu warmup
    for i in range(100):
        _ = torch.zeros(size=(100,100,100), dtype=torch.float32, device=device)


    data_path = args.data_dir

    with open(args.label_config, 'r') as stream:
        config = yaml.safe_load(stream)
    im_idx = []
    split = config['split']['test']
    for i_folder in split:
        im_idx += get_files('/'.join([data_path, 'sequences', str(i_folder).zfill(2), 'velodyne']), 'bin')

    # th = 2.5 # intensity threshold
    th = 80 # distance threshold
    pre_time = list()
    ex_time = list()
    pbar = tqdm(total=len(im_idx))

    pcms = np.zeros(shape=(2, 2, 2), dtype=np.int64)
    with torch.no_grad():
        for pcd_file in im_idx:
            data = np.fromfile(pcd_file, dtype=np.float32).reshape((-1, 4))
            label = np.fromfile(pcd_file.replace('velodyne', 'labels')[:-3] + 'label',
                                            dtype=np.int32)
            label = np.where(label == 110, 1, 0)
            snow_ind = np.where(label == 1)
            non_snow_ind = np.where(label == 0)
            p1 = torch.from_numpy(data[:, :3]).to(device)
            d = torch.sqrt(torch.sum(p1 * p1, dim=1))
            rem = torch.from_numpy(data[:, 3]).to(device)
            p1 = cp.asarray(p1)
            nn = NearestNeighbors()
            while True:

                try:
                    nn.fit(p1)
                    break
                except Exception:
                    print("caught it")

            dist, ind = nn.kneighbors(p1, 9)
            dist = torch.as_tensor(dist + 1.0, device=device)
            ind = torch.as_tensor(ind, device=device)
            # nn_rem = rem[ind]
            nn_rem = d[ind]
            # nn_rem2 = nn_rem * dist * dist
            nn_rem2 = nn_rem
            nn_fft = torch.fft.fft(nn_rem2, dim=1, norm='backward')
            norm_fft = nn_fft.norm(p=2, dim=1)
            # norm_fft = nn_rem.norm(p=2, dim=1)


            pred = torch.where(norm_fft >= th, 0, 1)
            pcm = multilabel_confusion_matrix(y_true=label, y_pred=pred.cpu().numpy(),
                                              labels=[i for i in range(2)])

            pbar.update()

            # evaluate_cm(pcm[1], f'snow}')
            pcms += pcm
    print('*' * 80)
    print('Evaluation using  multilabel confusion matrix')
    print('*' * 80)
    IOUs = list()
    ordered_class_names = ['background', 'snow']
    for i in range(0, 2):
        iou = evaluate_cm(pcms[i], ordered_class_names[i])
        print(pcms[i])
        IOUs.append(iou)
    class_jaccard = torch.tensor(np.array(IOUs))
    m_jaccard = class_jaccard.mean().item()

    for i, jacc in enumerate(class_jaccard):
        sys.stdout.write('{jacc:.2f} &'.format(jacc=jacc.item() * 100))
        # sys.stdout.write('{jacc:.2f}\\% &'.format(jacc=jacc.item() * 100))
        # sys.stdout.write(",")
    sys.stdout.write('{jacc:.2f}'.format(jacc=m_jaccard * 100))
    # sys.stdout.write(",")
    # sys.stdout.write('{acc:.2f}'.format(acc=m_accuracy.item()))
    sys.stdout.write('\n')
    for i in range(1, len(class_jaccard)):
        sys.stdout.write('\\bfseries{{ {name} }} &'.format(name=ordered_class_names[i]))
    sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':
    # Testing settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default='/var/local/home/aburai/DATA/WADS2')
    parser.add_argument("-label_config", type=str, default='../configs/binary_desnow_wads.yaml')
    parser.add_argument('-p', '--model_save_path',
                        default='/var/local/home/aburai/DATA/exp_2024/bin_seg/OutDet/outdet.pt')
    parser.add_argument('-o', '--test_output_path',
                        default='/var/local/home/aburai/DATA/exp_2024/bin_seg/OutDet/outputs')
    parser.add_argument('-m', '--model', choices=['polar', 'traditional'], default='polar',
                        help='training model: polar or traditional (default: polar)')
    parser.add_argument('--device', type=str, default='cuda:0', help='validation interval (default: 4000)')
    parser.add_argument('--K', type=int, default=3, help='batch size for training (default: 2)')

    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for training (default: 1)')

    parser.add_argument(
        '--desnow_root', '-dr',
        type=str,
        default=None,
        help='Set this if you want to use the Uncertainty Version'
    )
    parser.add_argument("--pred_folder",
                        type=str,
                        default=None)
    parser.add_argument('--snow_label',
                        type=int,
                        default=None)

    parser.add_argument('--save_pred', type=bool, default=False)
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    configure_randomness(12345)
    main(args)
