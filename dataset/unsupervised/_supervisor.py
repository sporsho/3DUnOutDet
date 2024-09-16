import os

import scipy
import torch.fft
import torch.nn as nn
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree

def estimate_plane(pcd):
    tpc = o3d.geometry.PointCloud()
    tpc.points = o3d.utility.Vector3dVector(pcd.copy())
    plane_model, inliers = tpc.segment_plane(distance_threshold=0.1, ransac_n=9, num_iterations=5000000)
    return plane_model


def segment_plane(plane_eqn, pcd, threshold):
    a, b, c, d = plane_eqn
    x = pcd[:, 0]
    y = pcd[:, 1]
    z = pcd[:, 2]
    normalizer = np.sqrt(a * a + b * b + c * c)
    pq = np.abs(a * x + b * y + c * z + d)
    D = pq / normalizer
    inliers = np.where(D <= threshold)[0]
    return inliers

class Supervisor(nn.Module):
    def __init__(self, th, mean=9.875928, std=14.734034):
        super().__init__()
        self.th = th
        self.mean = mean
        self.std = std

    def forward(self, data, dist, ind):
        intensity = data[:, 3] * self.std + self.mean
        # get neighbours intensity
        ki = intensity[ind]
        standard_ki = ki * dist * dist
        fft = torch.fft.fft(standard_ki, dim=1, norm='backward')
        fft_norm = fft.norm(p=2, dim=1)
        sup = torch.where(fft_norm >= self.th, 0, 1)
        return sup.reshape(-1)

class CADCSupervisor(nn.Module):
    def __init__(self, th, fname,mean=0.01739084, std= 0.07074773):
        super().__init__()
        self.fname = fname
        self.data = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
        self.th = th
        self.mean = mean
        self.std = std

    def forward(self, data, dist, ind):
        self.data = torch.from_numpy(self.data).to(data.device)
        intensity = self.data[:, 3]
        sq_dist = torch.sum(self.data[:, :3] * self.data[:, :3], dim=1)
        intensity /= sq_dist
        # intensity *= 255.0
        # get neighbours intensity
        ki = intensity[ind[:, :5]]
        standard_ki = ki * dist[:, :5] * dist[:, :5]
        fft = torch.fft.fft(standard_ki, dim=1, norm='backward')
        fft_norm = fft.norm(p=2, dim=1)
        sup = torch.where(fft_norm >= self.th, 0, 1)
        return sup.reshape(-1)

class KITTISupervisor(Supervisor):
    def forward(self, data, dist, ind):
        intensity = data[:, 3] * self.std + self.mean
        sq_dist = torch.sum(data[:, :3] * data[:, :3], dim=1)
        intensity /= sq_dist
        ki = intensity[ind]
        standard_ki = ki * dist * dist
        fft = torch.fft.fft(standard_ki, dim=1, norm='backward')
        fft_norm = fft.norm(p=2, dim=1)
        sup = torch.where(fft_norm >= self.th, 0, 1)
        return sup.reshape(-1)
class SpraySupervisor(object):
    def __init__(self, pcd_file):
        self.velo_pcd = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 5)
        self.pcd_file = pcd_file
        self.radar_file = pcd_file.replace("velodyne", "delphi_radar")
        self.front_lidar_file = pcd_file.replace("velodyne", "ibeo_front")
        self.radar_pcd = np.fromfile(self.radar_file, dtype=np.float32).reshape(-1, 4)
        self.front_pcd = np.fromfile(self.front_lidar_file, dtype=np.float32).reshape(-1, 4)



    def supervise(self):
        rn = self.radar_pcd.shape[0]
        vn = self.velo_pcd.shape[0]
        fn = self.front_pcd.shape[0]
        N = rn + vn + fn
        pcd = np.concatenate((self.radar_pcd[:, :3], self.velo_pcd[:, :3], self.front_pcd[:, :3]), axis=0)
        supervision = np.zeros(shape=(N,), dtype=np.float32)

        model_plane =  estimate_plane(pcd)
        inliers = segment_plane(model_plane, pcd, 0.1) # 10 cm threshold
        outliers = np.array([i for i in range(N) if i not in inliers])
        others = pcd[outliers]
        from sklearn.cluster import DBSCAN
        partial_supervisor = DBSCAN()
        partial_supervisor.fit(others[:, :3])
        partial_labels = partial_supervisor.labels_
        car_points = np.where(
            (others[:, 0] > 15.0) & (others[:, 0] < 40.0) & (others[:, 1] > -2.0) & (others[:, 1] < 2.0))
        partial_car_labels = partial_labels[car_points]
        try:
            voted_car_label = scipy.stats.mode(partial_car_labels).mode[0]
        except IndexError:
            print(self.pcd_file)

        primary_car_ind = partial_labels == voted_car_label
        secondary_car_ind = outliers[primary_car_ind]
        secondary_car_pcd = pcd[secondary_car_ind]

        carx = min(secondary_car_pcd[:, 0])
        egox = -30
        spray_primary_ind = np.where(
            (others[:, 0] >= egox) & (others[:, 0] < carx) & (others[:, 1] < 2.2) & (others[:, 1] > -2.0))
        spray_secondary_ind = outliers[spray_primary_ind]

        supervision[spray_secondary_ind] = 1

        center = np.where((pcd[:, 0] >= -1.8) & (pcd[:, 0] < 4.0))
        supervision[center] = 0
        return supervision[rn:rn+vn]







