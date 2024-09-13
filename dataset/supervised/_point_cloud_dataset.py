import torch
from torch.utils import data
import yaml
import os
import glob
import numpy as np
import pickle
import cupy as cp
from cuml.neighbors import NearestNeighbors
from cuml.common.device_selection import set_global_device_type

from dataset.utils.utils import read_bin_pcd, read_bin_label, apply_random_rotation_to_pcd, add_random_noise_to_pcd

set_global_device_type('gpu')


def get_files(folder, ext):
    files = glob.glob(os.path.join(folder, f"*.{ext}"))
    return files


class WadsPointCloudDataset(data.Dataset):
    def __init__(self, device, data_path, imageset='train', label_conf='wads.yaml', k=121, leaf_size=100, mean=[0.3420934,  -0.01516175 ,-0.5889243 ,  9.875928  ], std=[25.845459,  18.93466,    1.5863657, 14.734034 ],
                 shuffle_indices=False, save_ind=True, recalculate=False, desnow_root=None, pred_folder=None,
                                         snow_label=None):
        self.device = device
        self.recalculate = recalculate
        self.k = k
        self.leaf_size = leaf_size
        self.save_ind = save_ind
        self.mean = np.array(mean)
        self.std = np.array(std)



        self.imageset = imageset
        self.shuffle_indices = shuffle_indices
        self.desnow_root = desnow_root
        self.pred_folder = pred_folder
        self.snow_label = snow_label


        config = self.get_config(label_conf)
        if config is not None:
            self.learning_map = config['learning_map']
        else:
            self.learning_map = None
        split = self.get_split(imageset, config)

        self.im_idx = []
        self.pred_idx = list()

        self.gather_files(data_path, split)
        self.im_idx.sort()
        if desnow_root is not None:
            self.pred_idx.sort()
    def get_config(self, label_config):
        if label_config is None:
            return None
        else:
            with open(label_config, 'r') as stream:
                config = yaml.safe_load(stream)
            return config
    def gather_files(self, data_path, split):
        for i_folder in split:
            self.im_idx += get_files('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']), 'bin')
            if self.desnow_root is not None:
                assert os.path.exists(self.desnow_root)
                self.pred_idx += get_files('/'.join([self.desnow_root, str(i_folder).zfill(2), self.pred_folder]), 'label')

    def get_split(self, imageset, semkittiyaml):
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
        return split

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def read_data_label(self, index):
        data = read_bin_pcd(self.im_idx[index], ftype=np.float32, n_attributes=4)
        annotated_data = read_bin_label(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                        ftype=np.int32)
        return data, annotated_data


    def desnow_data_label(self, index, data, annotated_data):
        if self.pred_idx[index].endswith('.pred'):
            preds = np.fromfile(self.pred_idx[index], dtype=np.int64)
        else:
            preds = np.fromfile(self.pred_idx[index], dtype=np.int32)
        preds = preds.reshape(-1)
        snow_indices = np.where(preds == self.snow_label)[0]

        data = np.delete(data, obj=snow_indices.tolist(), axis=0)
        annotated_data = np.delete(annotated_data, obj=snow_indices.tolist(), axis=0)
        return data, annotated_data

    def __getitem__(self, index):
        data, annotated_data = self.read_data_label(index)
        if self.desnow_root is not None:
            data, annotated_data = self.desnow_data_label(index, data, annotated_data)

        kd_path = self.im_idx[index].replace('velodyne', 'knn')[:-3] + 'pkl'
        # err = True
        if os.path.exists(kd_path) and not self.recalculate:
            with open(kd_path, 'rb') as f:
                try:
                    ind = pickle.load(f)
                    dist = pickle.load(f)
                    err = False
                except EOFError:
                    err = True
        else:
            err = True
        if err or self.recalculate:
            p1 = torch.from_numpy(data[:, :3]).to(self.device)
            p1 = cp.asarray(p1)
            self.nn = NearestNeighbors()
            while True:

                try:
                    self.nn.fit(p1)
                    break
                except Exception:
                    print("caught it")

            dist, ind = self.nn.kneighbors(p1, self.k)
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
        # shuffle indices of the neighbour
        if self.shuffle_indices:
            s_ind = np.random.rand(*ind.shape).argsort(axis=1)
            ind = np.take_along_axis(ind, s_ind, axis=1)
            dist = np.take_along_axis(dist, s_ind, axis=1)
        if self.learning_map is not None:
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data).reshape(-1)
        data = (data - self.mean) / self.std
        if self.imageset == 'train':
            data = apply_random_rotation_to_pcd(data)
            data = add_random_noise_to_pcd(data)
        out_dict = {'data': data.astype(np.float32), 'dist': dist.astype(np.float32), 'ind': ind, 'label': annotated_data.astype(np.uint8)}
        return out_dict


class PointCloudDataset(data.Dataset):
    def __init__(self, data_path, imageset='train', label_conf='wads.yaml'):
        self.mean = np.array([0.43,0.29,-0.67,10.8])
        self.std = np.array([1.17,1.40,0.05,0.97])
        with open(label_conf, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        elif imageset == 'all':
            split = semkittiyaml['split']['train'] + semkittiyaml['split']['valid'] + semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.pred_idx = list()
        for i_folder in split:
            self.im_idx += get_files('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']), 'bin')
        self.im_idx.sort()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        data = (raw_data - self.mean) / self.std
        annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                     dtype=np.int32).reshape((-1, 1))
        annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary

        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data).reshape(-1)

        out_dict = {'data': data.astype(np.float32), 'label': annotated_data.astype(np.uint8)}
        return out_dict
