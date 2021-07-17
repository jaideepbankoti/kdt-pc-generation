import os
import numpy as np
import pdb
import h5py
import pickle
import shutil
import torch
from pathlib import Path
from KDTree.kdtree import KDTree
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.style.use('seaborn')

def viz_points(pts, title, fig, pos):
    color = np.array([[i]*100 for i in range(10)]).reshape(-1,)
    ax = fig.add_subplot(pos, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=40, c=color, alpha=0.9, cmap='brg')
    ax.view_init(elev=45, azim=45)
    plt.axis('off')
    plt.title(title)


def h_save(fname, data):
	f = h5py.File(fname, 'w')
	f.create_dataset('data', data=data)
	f.close()

def h_read(fname, key):
	f = h5py.File(fname, 'r')
	points = np.array(f[key])
	f.close()
	return points

def p_dump(fname, data):
	with open(fname, 'wb') as F:
		pickle.dump(data, F, protocol=pickle.HIGHEST_PROTOCOL)

def p_load(fname):
	with open(fname, 'rb') as F:
		data = pickle.load(F)
	return data


def process_points(lines):
    points = np.zeros((1000, 3))
    for i, l in enumerate(lines[10:]):
        stripped = l.split(' ')
        points[i, 0], points[i, 1], points[i, 2] = float(stripped[0]), float(stripped[1]), float(stripped[2].split('\n')[0])
    return points


def create_dataset(data_dir, parse_func):
    data_dir = Path(data_dir)
    data_items = sorted(os.listdir(data_dir))
    stack_points = []
    stack_data = []
    # pdb.set_trace()
    for i, item in tqdm(enumerate(data_items), total=len(data_items)):
        item_dir = data_dir / item
        with open(item_dir, 'r') as F:
            lines = F.readlines()
        points = parse_func(lines)
        kdtree = KDTree(points, k=3)
        kdtree.inorder(kdtree.kdt)
        ordered_points = np.array(kdtree.order)
        stack_points.append(ordered_points)
        ordered_points = np.expand_dims(ordered_points.flatten(order='C'), axis=-1)
        stack_data.append(ordered_points)
    return np.stack(stack_points), np.hstack(stack_data)


def reconstruction_err(U, P, mu):
    P_mean = P - mu
    l = np.linalg.norm(np.matmul(np.matmul(P_mean.T, U), U.T) - P_mean.T)**2
    # print(P_mean.shape)
    return l

def reconstruction_err_vectorized(U, P, mu):
    P_mean = P - mu
    l = np.linalg.norm((np.matmul(np.matmul(P_mean.T, U), U.T) - P_mean.T), axis=1)**2
    # print(P_mean.shape)
    return l

class NormalizedScale(object):
    """Normalizes the pointcloud data"""

    def __call__(self, sample_point):
        sample_point = sample_point - sample_point.mean(dim=-2, keepdims=True)
        scale = (1 / sample_point.abs().max()) * 0.999999
        sample_point = sample_point * scale

        return sample_point


def optimize_iter(points, points_ind, dataset, U, k=10000, show=4000):
    # pdb.set_trace()
    mu = dataset.mean(axis=1)
    err = reconstruction_err(U, points.flatten(order='C'), mu)
    # swap_points = np.copy(points)
    # swap_dataset = np.copy(dataset)
    for i in range(k):
        swap_points = np.copy(points)
        swap_ind = np.sort(np.random.choice(points.shape[0], size=2, replace=False))
        swap_points[swap_ind] = swap_points[swap_ind[::-1]]
        r_err = reconstruction_err(U, swap_points.flatten(order='C'), mu)
        if r_err < err:
            # swap_points[swap_ind] = swap_points[swap_points[::-1]]
            # swap_dataset[:, points_ind] = swap_points.flatten(order='C')
            points = swap_points
            dataset[:, points_ind] = swap_points.flatten(order='C')
            mu = dataset.mean(axis=1)
            err = r_err
        if i % show == 0 and i !=0:
            print(f'point_ind: {points_ind}\ni: {i}\nMin: {err}\nRec_err: {r_err}')
    return dataset, err


def optimize_iter_vectorized(full_dataset, dataset, U, k=10000, show=4000):
    mu = np.expand_dims(dataset.mean(axis=1), -1)
    err_vec = reconstruction_err_vectorized(U, dataset, mu)
    for i in range(k):
        swap_dataset = np.copy(full_dataset)
        swap_ind = np.sort(np.random.choice(swap_dataset.shape[1], size=2, replace=False))
        swap_dataset[:, swap_ind, :] = swap_dataset[:, swap_ind[::-1], :]
        r_err_vec = reconstruction_err_vectorized(U, swap_dataset.reshape((-1, 3000), order='C').T, mu)
        rec_flag = r_err_vec < err_vec
        if np.sum(rec_flag) != 0:
            axis_0 = np.where(rec_flag)[0].tolist() * 2
            axis_1 = ([swap_ind[0]] * rec_flag.shape[0]).extend([swap_ind[1]] * rec_flag.shape[0])
            axis_1_I = ([swap_ind[1]] * rec_flag.shape[0]).extend([swap_ind[0]] * rec_flag.shape[0])
            swap_dataset[axis_0 , axis_1, :] = swap_dataset[axis_0, axis_1_I, :]
            full_dataset = swap_dataset
            dataset = swap_dataset.reshape((-1, 3000), order='C').T
            mu = np.expand_dims(dataset.mean(axis=1), -1)
            err_vec[rec_flag] = r_err_vec[rec_flag]
        # if i % show == 0 and i !=0:
            # print(f'i: {i}\nRec_err: {np.sum(err_vec)}\nerr: {np.sum(r_err_vec)}')
    return dataset, full_dataset, np.sum(err_vec)


# checkpoint saving function
def save_ckpt(state, is_best, ckpt_path, best_model_path):
    """
    state: checkpoint to save
    is_best: is this the best checkpoint; min validation loss
    ckpt_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = ckpt_path
    # save checkpoint data to the path given, ckpt_path
    torch.save(state, f_path)
    # if it is the best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given
        shutil.copyfile(f_path, best_fpath)


# checkpoint loading function
def load_ckpt(ckpt_fpath, model, optimizer):
    """
    ckpt_fpath: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load checkpoint
    checkpoint = torch.load(ckpt_fpath, map_location=torch.device('cpu'))
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min


