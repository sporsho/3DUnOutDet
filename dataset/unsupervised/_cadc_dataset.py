import torch
from torch.utils import data
import glob
from dataset import CADC
from dataset.supervised import WadsPointCloudDataset
import os

from dataset.utils.utils import read_bin_pcd
import numpy as np

class CADCPointCloudDataset(WadsPointCloudDataset):
    def __init__(self,device, data_path, imageset='train', k=121, leaf_size=100,
                 mean=[ 0.2759826, 1.4695336, -0.20206092, 0.01739084] , std=[23.292917, 20.201435, 1.6440893, 0.07074773],
                 shuffle_indices=False, save_ind=True, recalculate=False, desnow_root=None, pred_folder=None,
                                         snow_label=None):
        super().__init__(device=device, data_path=data_path, imageset=imageset, label_conf=None, k=k, leaf_size=leaf_size, mean=mean,
                         std=std, shuffle_indices=shuffle_indices, save_ind=save_ind, recalculate=recalculate,
                         desnow_root=desnow_root, pred_folder=pred_folder, snow_label=snow_label)

    def get_split(self, imageset, semkittiyaml):
        assert imageset in ['train', 'val', 'test', 'human_annotations', 'all']
        if imageset == 'all':
            split = list()
            split.extend(CADC['train'])
            split.extend(CADC['val'])
            split.extend(CADC['test'])
        elif imageset == 'train':
            split = CADC['train']
        elif imageset == 'val':
            split = CADC['val']
        elif imageset == 'test':
            split = CADC['test']
        elif imageset == 'human_annotations':
            split = ['2019_02_27_0043', '2019_02_27_0075']
        else:
            raise ValueError()

        return split

    def gather_files(self, data_path, split):
        for date_seq in split:
            self.im_idx += glob.glob(os.path.join(data_path, date_seq, 'velodyne', '*.bin'))
            if self.desnow_root is not None:
                raise ValueError('not implemented yet')

    def read_data_label(self, index):
        data = read_bin_pcd(self.im_idx[index], ftype=np.float32, n_attributes=4)
        anno = np.zeros(shape=(data.shape[0]), dtype=np.int32)
        return data, anno

    def __getitem__(self, index):
        out_dict = super().__getitem__(index)
        del out_dict['label']
        out_dict['file'] = self.im_idx[index]
        return out_dict


if __name__ == '__main__':
    device = torch.device('cuda:0')
    data_root = '/var/local/home/aburai/DATA/CADC_V2'
    ds = CADCPointCloudDataset(device=device, data_path=data_root, imageset='all')

    print(len(ds))
