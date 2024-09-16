import numpy as np

def read_bin_pcd(path, ftype, n_attributes):
    data = np.fromfile(path, dtype=ftype)
    data = data.reshape(-1, n_attributes)
    return data

def read_bin_label(path, ftype):
    label = np.fromfile(path, dtype=ftype).reshape((-1, 1))
    label = label & 0xFFFF  # delete high 16 digits binary
    return label

def apply_random_rotation_to_pcd(data):

    if np.random.random() > 0.5:
        rotate_rad = np.deg2rad(np.random.random() * 360)
        cos, sine = np.cos(rotate_rad), np.sin(rotate_rad)
        rot_mat = np.matrix([[cos, sine], [-sine, cos]])
        # rotate x and y
        data[:, :2] = np.dot(data[:, :2], rot_mat)
    return data

def add_random_noise_to_pcd(data):
    if np.random.random() > 0.5:
        data[:, :3] += np.random.normal(size=(data.shape[0], 3), loc=0, scale=0.1)
    return data