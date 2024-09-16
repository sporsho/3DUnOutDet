import numpy as np
import torch


def collate_fn_cp(data):
    data2stack = np.stack([d['data'] for d in data]).astype(np.float32)
    dist2stack = np.stack([d['dist'] for d in data]).astype(np.float32)
    ind2stack = np.stack([d['ind'] for d in data]).astype(np.int64)

    out_dict = dict()
    out_dict['data'] = torch.from_numpy(data2stack)
    out_dict['dist'] = torch.from_numpy(dist2stack)
    out_dict['ind'] = torch.from_numpy(ind2stack)
    if 'file' in data[0].keys():
        files = [d['file'] for d in data]
        out_dict['file'] = files
    if 'label' in data[0].keys():
        label2stack = np.stack([d['label'] for d in data])
        out_dict['label'] = torch.from_numpy(label2stack)
    if 'supervision' in data[0].keys():
        sup2stack = np.stack([d['supervision'] for d in data])
        out_dict['supervision'] = torch.from_numpy(sup2stack)
    return out_dict


def collate_fn_cp_inference(data):
    data2stack = np.stack([d['data'] for d in data]).astype(np.float32)
    dist2stack = np.stack([d['dist'] for d in data]).astype(np.float32)
    ind2stack = np.stack([d['ind'] for d in data]).astype(np.int64)
    # we do not need label during inference, but it does not hurt to provide one
    out_dict = dict()
    out_dict['data'] = torch.from_numpy(data2stack)
    out_dict['dist'] = torch.from_numpy(dist2stack)
    out_dict['ind'] = torch.from_numpy(ind2stack)
    if 'file' in data[0].keys():
        files = [d['file'] for d in data]
        out_dict['file'] = files
    if 'label' in data[0].keys():
        label2stack = np.stack([d['label'] for d in data])
        out_dict['label'] = torch.from_numpy(label2stack)
    if 'supervision' in data[0].keys():
        sup2stack = np.stack([d['supervision'] for d in data])
        out_dict['supervision'] = torch.from_numpy(sup2stack)
    return out_dict

