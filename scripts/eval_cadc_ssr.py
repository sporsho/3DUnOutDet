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

from dataset.unsupervised._cadc_dataset import CADCPointCloudDataset
from dataset.unsupervised._supervisor import CADCSupervisor
from deterministic import configure_randomness
from modules import OutDet
from dataset.utils.collate import collate_fn_cp, collate_fn_cp_inference
from dataset.supervised import WadsPointCloudDataset
import warnings


warnings.filterwarnings("ignore")





def get_seq_name_from_path(path):
    tmps = path.split(os.path.sep)
    seq = tmps[-3]
    name = tmps[-1]
    tmps2 = name.split(".")
    name = tmps2[0]
    return seq, name


def main(args):
    data_path = args.data_dir
    test_batch_size = args.test_batch_size
    model_save_path = args.model_save_path
    device = torch.device(args.device)
    dilate = 1

    with open(args.label_config, 'r') as stream:
        config = yaml.safe_load(stream)

    class_strings = config["labels"]
    class_inv_remap = config["learning_map_inv"]
    num_classes = len(class_inv_remap)

    keys = class_inv_remap.keys()
    max_key = max(keys)
    look_up_table = np.zeros((max_key + 1), dtype=np.int32)
    for k, v in class_inv_remap.items():
        look_up_table[k] = v

    ordered_class_names = [class_strings[class_inv_remap[i]] for i in range(num_classes)]
    # prepare model
    model = OutDet(num_classes=num_classes, kernel_size=args.K, depth=1, dilate=dilate)
    model = model.to(device)
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    else:
        raise ValueError()

    # prepare dataset
    tree_k = int(np.round(args.K * args.K))
    test_dataset = CADCPointCloudDataset(device, data_path + '/sequences/', imageset='all', k=tree_k,
                                         desnow_root=args.desnow_root, pred_folder=args.pred_folder,
                                         snow_label=args.snow_label, recalculate=False, save_ind=False)
    if test_dataset.save_ind:
        collate_fn = collate_fn_cp
    else:
        collate_fn = collate_fn_cp_inference
    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=test_batch_size,
                                                      shuffle=False,
                                                      num_workers=0,
                                                      collate_fn=collate_fn)

    # validation
    print('*' * 80)
    print('Test network performance on validation split')
    print('*' * 80)
    pbar = tqdm(total=len(test_dataset_loader))
    model.eval()

    pcms = np.zeros(shape=(num_classes, 2, 2), dtype=np.int64)
    with torch.no_grad():
        for i_iter_val, batch in enumerate(
                test_dataset_loader):
            data = batch['data'][0].to(device)
            ind = batch['ind'][0]
            dist = batch['dist'][0].to(device)
            fname = batch['file'][0]
            supervisor = CADCSupervisor(th=1e-9, fname=fname)
            supervision = supervisor(data, dist, ind)


            pred_np = supervision.cpu().numpy().reshape(-1)
            inlier = np.where(pred_np == 0)[0]
            pcd = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
            pcd = pcd[inlier]
            pcd = pcd.reshape(-1).astype(np.float32)
            out_file = fname.replace('CADC_V2', 'CADC_SSR')
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            pcd.tofile(out_file)
            pbar.update()
    pbar.close()


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
    parser.add_argument('-d', '--data_dir', default='/var/local/home/aburai/DATA/CADC_V2')
    parser.add_argument("-label_config", type=str, default='../configs/binary_desnow_wads.yaml')
    parser.add_argument('-p', '--model_save_path',
                        default='/var/local/home/aburai/DATA/exp_2024/bin_seg/UnOutDetCADC/outdet.pt')
    parser.add_argument('-o', '--test_output_path',
                        default='/var/local/home/aburai/DATA/CADC_OUTDET')
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

    parser.add_argument('--save_pred', type=bool, default=True)
    args = parser.parse_args()


    print(' '.join(sys.argv))
    print(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    configure_randomness(12345)
    main(args)
