import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from utils.helpers import *
from time import time

import pdb

DATA_DIR = 'data/shapenet-chairs-pcd'
ASSETS_DIR = Path('assets/process_data')
Full_dataset_ckpt = ASSETS_DIR / 'full_dataset.h5'
Dataset_ckpt = ASSETS_DIR / 'dataset.h5'
R_err_list_ckpt = ASSETS_DIR / 'r_err_list.h5'
U_ckpt = ASSETS_DIR / 'U.h5'
state_ckpt = ASSETS_DIR / 'state.pickle' 
SHAPE_BASIS = 100
K_val = 10000
n_iter = 1000

#-------------- Load Dataset

# will load data in the checkpointing section
# points_dataset, dataset = create_dataset(data_dir=DATA_DIR, parse_func=process_points)

#-------------- Compute PCA Basis

# U, S, Vt = randomized_svd(dataset - np.mean(dataset, axis=0), SHAPE_BASIS)

#-------------- Checkpointing

if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)

if os.path.isfile(Full_dataset_ckpt):
    full_dataset = h_read(Full_dataset_ckpt, 'data')
else:
    points_dataset, dataset = create_dataset(data_dir=DATA_DIR, parse_func=process_points)
    full_dataset = points_dataset
    h_save(Full_dataset_ckpt, full_dataset)

if os.path.isfile(Dataset_ckpt):
    dataset = h_read(Dataset_ckpt, 'data')
else:
    dataset = dataset
    h_save(Dataset_ckpt, dataset)

if os.path.isfile(R_err_list_ckpt):
    r_err_list = h_read(R_err_list_ckpt, 'data')
    r_err_list = r_err_list.tolist()
else:
    r_err_list = []

if os.path.isfile(U_ckpt):
    U = h_read(U_ckpt, 'data')
else:
    U, S, Vt = randomized_svd(dataset, SHAPE_BASIS)


if os.path.isfile(state_ckpt):
    state_dict = p_load(state_ckpt)
    i_resume = state_dict['i']
    try:
        iter_resume = state_dict['iter']
    except:
        iter_resume = 0
else:
    state_dict = {}
    iter_resume = 0
    i_resume = 0

#-------------- Optimize ordering

print('Resumed or Started at I iteration: ', iter_resume)
for iter in range(iter_resume, n_iter):
    print('Resumed or Started at K iteration: ', i_resume)
    r_err = 0
    for i in tqdm(range(i_resume, full_dataset.shape[0])):
        # s = time()
        points = full_dataset[i]
        dataset, err = optimize_iter(points, i, dataset, U, k=K_val, show=999999)
        r_err += err
        if (i+1)%500 == 0:
            print(err)
        # print(time() - s)
    U, S, Vt = randomized_svd(dataset, SHAPE_BASIS)
    r_err_list.extend([r_err])
    # r_err_list.extend([reconstruction_err(U, dataset, np.expand_dims(dataset.mean(axis=1), -1))])
    full_dataset = dataset.T.reshape((-1, 1000, 3), order='C')
    state_dict['iter'] = iter
    state_dict['i'] = 0
    # all saves
    h_save(Dataset_ckpt, dataset)
    h_save(U_ckpt, U)
    h_save(R_err_list_ckpt, np.array(r_err_list))
    h_save(Full_dataset_ckpt, full_dataset)
    p_dump(state_ckpt, state_dict)
    print(f'Overall reconstruction error: {r_err_list[-1]}')
    with open('r_err_log_non_vectorized.txt', 'a') as F:
        F.write(f'Overall reconstruction error for {iter} is : {r_err_list[-1]}\n')

