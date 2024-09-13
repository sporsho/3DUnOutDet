import numpy as np
import glob
import os

if __name__ == "__main__":
    cadc_root = '/var/local/home/aburai/DATA/CADC_V2'
    files = glob.glob(os.path.join(cadc_root, 'sequences',  '*', 'velodyne','*.bin'))
    print(len(files))
    pcd_list = list()
    for f in files:
        data = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
        pcd_list.append(data)
    all_data = np.concatenate(pcd_list, axis=0)
    ranges = np.sqrt(np.sum(all_data[:, :3] * all_data[:, :3], axis=1))
    print(np.mean(all_data, axis=0), np.std(all_data, axis=0))

    print(f'mean range: {np.mean(ranges)}, std: range {np.std(ranges)}')
    # [0.28444412  1.629835 - 0.2020733   0.01733035][23.441633
    # 20.302063
    # 1.6583984
    # 0.07110535]
    # mean
    # range: 31.238571166992188, std: range
    # 26.445486068725586