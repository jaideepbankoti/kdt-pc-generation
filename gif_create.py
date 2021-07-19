import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from utils.helpers import process_points, create_dataset
import tqdm
from sklearn.utils.extmath import randomized_svd
from pathlib import Path

DATA_DIR = Path('data/shapenet-chairs-pcd')

def create_original_dataset(data_dir, parse_func):
    data_items = sorted(os.listdir(data_dir))
    stack_points = []
    # pdb.set_trace()
    for i, item in enumerate(data_items):
        item_dir = data_dir / item
        with open(item_dir, 'r') as F:
            lines = F.readlines()
        points = parse_func(lines)
        stack_points.append(points)
    return np.stack(stack_points)

def h_read(fname, key):
    with h5py.File(fname, "r") as f:
        # a_group_key = list(f.keys())[0]
        data = np.array(f['data'])
    return data


def rot_fig(f1, f2, f3, f4, f5, f6):
    color = np.array([[i]*100 for i in range(10)]).reshape(-1,)
    fig = plt.figure()
    d = f1
    ax1 = fig.add_subplot(231, projection='3d')
    xx = d[:, 0]
    yy = d[:, 1]
    zz = d[:, 2]
    ax1.scatter(xx,yy,zz, marker='o', s=15, c=color, alpha=0.9, cmap='brg')
    plt.axis('off')
    d = f2
    ax2 = fig.add_subplot(232, projection='3d')
    xx = d[:, 0]
    yy = d[:, 1]
    zz = d[:, 2]
    ax2.scatter(xx,yy,zz, marker='o', s=15, c=color, alpha=0.9, cmap='brg')
    plt.axis('off')
    d = f3
    ax3 = fig.add_subplot(233, projection='3d')
    xx = d[:, 0]
    yy = d[:, 1]
    zz = d[:, 2]
    ax3.scatter(xx,yy,zz, marker='o', s=15, c=color, alpha=0.9, cmap='brg')
    plt.axis('off')
    d = f4
    ax4 = fig.add_subplot(234, projection='3d')
    xx = d[:, 0]
    yy = d[:, 1]
    zz = d[:, 2]
    ax4.scatter(xx,yy,zz, marker='o', s=15, c=color, alpha=0.9, cmap='brg')
    plt.axis('off')
    d = f5
    ax5 = fig.add_subplot(235, projection='3d')
    xx = d[:, 0]
    yy = d[:, 1]
    zz = d[:, 2]
    ax5.scatter(xx,yy,zz, marker='o', s=15, c=color, alpha=0.9, cmap='brg')
    plt.axis('off')
    d = f6
    ax6 = fig.add_subplot(236, projection='3d')
    xx = d[:, 0]
    yy = d[:, 1]
    zz = d[:, 2]
    ax6.scatter(xx,yy,zz, marker='o', s=15, c=color, alpha=0.9, cmap='brg')
    plt.axis('off')
    for ii in range(0,360,4):
        ax1.view_init(elev=40., azim=ii)
        # plt.axis('off')
        ax2.view_init(elev=40., azim=ii)
        # plt.axis('off')
        ax3.view_init(elev=40., azim=ii)
        # plt.axis('off')
        ax4.view_init(elev=40., azim=ii)
        # plt.axis('off')
        ax5.view_init(elev=40., azim=ii)
        # plt.axis('off')
        ax6.view_init(elev=40., azim=ii)
        # plt.axis('off')
        plt.savefig("gifs_data/movie%d.png" % ii)

def full_dataset_svd(full_dataset, i):
    U, S, Vt = randomized_svd(full_dataset.reshape(5000, 3000).T, 100)
    return (U @ np.diag(S) @ Vt[:, i]).reshape(-1, 3)


def main():
    full_dataset_r = h_read('SVD_reconstruct.h5', 'data')
    # full_dataset = create_original_dataset(data_dir=DATA_DIR, parse_func=process_points)
    # full_dataset, _ = create_dataset(data_dir=DATA_DIR, parse_func=process_points)
    # f1 = full_dataset[0]
    # f2 = full_dataset[150]
    # f3 = full_dataset[240]
    # f4 = full_dataset[350]
    # f5 = full_dataset[450]
    # f6 = full_dataset[550]
    f1 = full_dataset_svd(full_dataset_r, 0)
    f2 = full_dataset_svd(full_dataset_r, 150)
    f3 = full_dataset_svd(full_dataset_r, 240)
    f4 = full_dataset_svd(full_dataset_r, 350)
    f5 = full_dataset_svd(full_dataset_r, 450)
    f6 = full_dataset_svd(full_dataset_r, 550)
    rot_fig(f1, f2, f3, f4, f5, f6)

