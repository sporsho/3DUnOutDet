import numpy as np
import glob
import os
import json

from scripts.deterministic import configure_randomness


def get_date_from_fname(path):
    tmp = path.split(os.path.sep)
    return tmp[-3], tmp[-2]

def get_unique_labels(ann):
    N = len(ann)
    objects = list()
    for i in range(N):
        frame = ann[i]['cuboids']
        M = len(frame)
        for j in range(M):
            label = frame[j]['label']
            objects.append(label)
    un = set(objects)
    return un
def save_split_as_dict(train_split, val_split, test_split):
    CADC = dict()
    CADC['train'] = train_split
    CADC['val'] = val_split
    CADC['test'] = test_split
    with open('_cadc_split.py', 'w') as f:
        f.write(f'CADC = {CADC}')

def save_split(split, fname):
    with open(fname, 'w') as f:
        for s in split:
            tmp = s.split("_")
            seq = tmp[-1]
            date = "_".join(tmp[:-1])
            f.write(f'{date} {seq}\n')
if __name__ == "__main__":
    configure_randomness(12345)
    cadc_root = '/var/local/home/aburai/DATA/CADC'
    files = glob.glob(os.path.join(cadc_root, "*", "*", "*", "*", '3d_ann.json'))
    print(len(files))
    label_dict = dict()
    all_labels = list()
    for f in files:
        date_, seq = get_date_from_fname(f)
        joint_seq_name = f'{date_}_{seq}'
        # print(date_, seq)
        ann = json.load(open(f, 'r'))
        labels = get_unique_labels(ann)
        for l in labels:
            if l in label_dict.keys():
                label_dict[l].append(joint_seq_name)
            else:
                label_dict[l] = [joint_seq_name]
    print(label_dict)
    keys = label_dict.keys()
    for k in keys:
        label_dict[k] = sorted(label_dict[k])
        seq = label_dict[k]
        print(f'{k}: {len(seq)}')

    selected_labels = ['Truck', 'Traffic_Guidance_Objects', 'Bicycle', 'Bus', 'Pedestrian', 'Garbage_Containers_on_Wheels', 'Pedestrian_With_Object', 'Car']
    # create validation set:
    val_split = list()
    for sl in selected_labels:
        seq = label_dict[sl].pop(0)
        for tmp in selected_labels:
            if seq in label_dict[tmp]:
                label_dict[tmp].remove(seq)
        val_split.append(seq)
    val_split = list(set(val_split))

    train_split = list()
    test_split = list()
    to_be_removed = list()
    for sl in selected_labels:
        train = True
        while len(label_dict[sl]) != 0:
            seq = label_dict[sl].pop()
            # remove this seq from the dict
            for tmp in selected_labels:
                if seq in label_dict[tmp]:
                    label_dict[tmp].remove(seq)
            if train:
                train_split.append(seq)
                train = False
            else:
                test_split.append(seq)
                train = True

    train_split = list(set(train_split))
    test_split = list(set(test_split))
    print(train_split)
    print(test_split)
    print(val_split)
    print(len(train_split) + len(test_split) + len(val_split))
    save_split(train_split, 'train.txt')
    save_split(val_split, 'val.txt')
    save_split(test_split, 'test.txt')
    save_split_as_dict(train_split, val_split, test_split)




# {'Animals', 'Truck', 'Traffic_Guidance_Objects', 'Bicycle', 'Bus', 'Pedestrian', 'Garbage_Containers_on_Wheels', 'Pedestrian_With_Object', 'Car', 'Horse_and_Buggy'}
# Bus: 42
# Car: 75
# Truck: 68
# Traffic_Guidance_Objects: 12
# Pedestrian: 58
# Garbage_Containers_on_Wheels: 33
# Bicycle: 4
# Pedestrian_With_Object: 9
# Animals: 1
# Horse_and_Buggy: 1