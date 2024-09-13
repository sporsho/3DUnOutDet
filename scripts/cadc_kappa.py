import glob
import os

import numpy as np
from sklearn.metrics import cohen_kappa_score

if __name__ == "__main__":
    data_root = '/var/local/home/aburai/DATA/CADC_V2'
    human_annotated_seq = ['2019_02_27_0043', '2019_02_27_0075']
    ma = list()
    ha = list()
    for seq in human_annotated_seq:
        ma_files = glob.glob(os.path.join(data_root, 'sequences', seq, 'machine_annotations', "*.label"))
        for maf in ma_files:
            haf = maf.replace('machine_annotations', 'human_annotations')
            ann_h = np.fromfile(haf, dtype=np.int32).reshape(-1, 1)
            ann_m = np.fromfile(maf, dtype=np.int32).reshape(-1, 1)
            ma.append(ann_m)
            ha.append(ann_h)

    all_ma = np.concatenate(ma, axis=0)
    all_ha = np.concatenate(ha, axis=0)
    kappa = cohen_kappa_score(y1=all_ha.reshape(-1), y2=all_ma.reshape(-1), weights='quadratic')
    print(kappa)