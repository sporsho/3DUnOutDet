#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import sys
import numpy as np
import torch
import yaml
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import multilabel_confusion_matrix

from deterministic import configure_randomness
from modules import OutDet
from dataset.utils.collate import collate_fn_cp, collate_fn_cp_inference
from dataset.supervised import WadsPointCloudDataset
import warnings
import glob

warnings.filterwarnings("ignore")





def get_seq_name_from_path(path):
    tmps = path.split(os.path.sep)
    seq = tmps[-3]
    name = tmps[-1]
    tmps2 = name.split(".")
    name = tmps2[0]
    return seq, name


def main(args):
    data_root = '/var/local/home/aburai/DATA/CADC_V2'
    human_annotated_seq = ['2019_02_27_0043', '2019_02_27_0075']

    pcms = np.zeros(shape=(2, 2, 2), dtype=np.int64)
    for seq in human_annotated_seq:
        label_files = glob.glob(os.path.join(data_root, 'sequences', seq, 'human_annotations', "*.label"))
        for lf in label_files:
            pf = lf.replace('human_annotations', 'predictions')
            pred = np.fromfile(pf, dtype=np.int32).reshape(-1)
            label = np.fromfile(lf, dtype=np.int32).reshape(-1)
            pcm = multilabel_confusion_matrix(y_true=label, y_pred=pred,
                                        labels=[i for i in range(2)])

            pcms += pcm
    evaluate_cm(pcms[0], 'not-snow')
    evaluate_cm(pcms[1], 'snow')


def evaluate_cm(cm, class_name):
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    iou1 = tp / (tp + fp + fn)
    f1 = 2 * recall * precision / (precision + recall)
    print(f'Class: {class_name}, Precision:{precision}, Recall: {recall}, IOU: {iou1}, F1: {f1}')
    return iou1


if __name__ == '__main__':
    # Testing settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default='/var/local/home/aburai/DATA/WADS2')
    parser.add_argument("-label_config", type=str, default='../configs/binary_desnow_wads.yaml')
    parser.add_argument('-p', '--model_save_path',
                        default='/var/local/home/aburai/DATA/exp_2024/bin_seg/UnOutDet/outdet.pt')
    parser.add_argument('-o', '--test_output_path',
                        default='/var/local/home/aburai/DATA/exp_2024/bin_seg/UnOutDet/outputs')
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
