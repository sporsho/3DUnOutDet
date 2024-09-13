from dataset.supervised import WadsPointCloudDataset

class UnSupervisedWADS(WadsPointCloudDataset):
    def __init__(self, device, data_path, imageset='train', label_conf='wads.yaml', k=121, leaf_size=100,
                 mean=[0.3420934, -0.01516175, -0.5889243, 9.875928], std=[25.845459, 18.93466, 1.5863657, 14.734034],
                 shuffle_indices=False, save_ind=True, recalculate=False, desnow_root=None, pred_folder=None,
                 snow_label=None, single_shot=False):
        super().__init__(device, data_path, imageset, label_conf, k, leaf_size,
                 mean, std,
                 shuffle_indices, save_ind, recalculate, desnow_root, pred_folder,
                 snow_label)
        if single_shot:
            self.im_idx = self.im_idx[:1]
            if desnow_root is not None:
                self.pred_idx = self.pred_idx[:1]

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        del data_dict['label'] # remove labels to make it truely unsupervised
        return data_dict


