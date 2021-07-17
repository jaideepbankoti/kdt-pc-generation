import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import PCA_optim
from torch.utils.data import DataLoader
import torch.optim as optim
from model.gan import ShapeGenerator, ShapeDiscriminator
from utils.helpers import save_ckpt, load_ckpt, viz_points, NormalizedScale, h_save
from utils.loss_utils import GenLoss, DisLoss, ChamferLoss
from tqdm import tqdm
from pathlib import Path
import pdb
plt.style.use('seaborn')

# for setting the device id for multiple GPUs
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

# ------------------------ Configurations setting ------------------------------------------

# setting seeds
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)    

# hyperparameters, later port it to a config file or an argument parser
n_epochs = 500
batch_size = 32
lr_gen = 0.0025
lr_dis = 1e-4
dis_train_cutoff = 0.8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = Path("assets/process_data")

#  -----------------------Configuration setting end ----------------------------------------

# loading the train data
train_data = PCA_optim(path, val=False)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# loading the  valid data
valid_data = PCA_optim(path, val=True, vslice=train_data.vslice)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

# defining a loader dictionary
loaders = {"train": train_loader,
    "valid": valid_loader}

model_gen = ShapeGenerator()
model_dis = ShapeDiscriminator()
optimizer_gen = optim.Adam(model_gen.parameters(), lr=lr_gen)
optimizer_dis = optim.Adam(model_dis.parameters(), lr=lr_dis)
chamfer_loss = ChamferLoss()

# --------- checkpointing logic start ----------------------------
start_epoch = 0
valid_loss_min_gen = np.Inf
valid_loss_min_dis = np.Inf
# checkpointing snippet
if os.path.isdir('assets/checkpoint') is False:
    os.makedirs('assets/checkpoint')
if os.path.isdir('assets/best_model') is False:
    os.makedirs('assets/best_model')
if os.path.isdir('experiments/train_save') is False:
    os.makedirs('experiments/train_save')
if os.path.isdir('experiments/valid_save') is False:
    os.makedirs('experiments/valid_save')
gen_ckpt_path = "assets/checkpoint/gen_current_checkpoint.pt"
dis_ckpt_path = "assets/checkpoint/dis_current_checkpoint.pt"
gen_best_model_path = "assets/best_model/gen_best_model.pt"
dis_best_model_path = "assets/best_model/dis_best_model.pt"

# load from saved checkpoint
if os.path.isfile(dis_ckpt_path):
    model_dis, optimizer_dis, start_epoch, valid_loss_min_dis = load_ckpt(dis_ckpt_path, model_dis, optimizer_dis)

if os.path.isfile(gen_ckpt_path):
    model_gen, optimizer_gen, _,  valid_loss_min_gen = load_ckpt(gen_ckpt_path, model_gen, optimizer_gen)

model_gen = model_gen.to(device)
model_dis = model_dis.to(device)

# --------- checkpointing logic end ------------------------------

# the train method
def train(start_epoch, n_epochs, valid_loss_min_input, model, optimizer, ckpt_path, best_model_path):
    valid_loss_min_gen, valid_loss_min_dis = valid_loss_min_input
    model_gen, model_dis = model
    optimizer_gen, optimizer_dis = optimizer
    gen_ckpt_path, dis_ckpt_path = ckpt_path
    gen_best_model_path, dis_best_model_path = best_model_path

    for epoch in range(start_epoch, n_epochs+1):
        # initialize variables to monitor training and validation loss
        gen_train_loss, dis_train_loss = 0.0, 0.0
        gen_valid_loss, dis_valid_loss = 0.0, 0.0
        train_closs, valid_closs = 0.0, 0.0

        # train the model
        gen_running_loss, dis_running_loss, running_closs = 0.0, 0.0, 0.0
        # model_dis.train()
        # model_gen.train()
        train_save_list = np.random.randint(0, int(len(train_data)/loaders['train'].batch_size), size=3)
        for i, data in tqdm(enumerate(loaders['train']), total=int(len(train_data)/loaders['train'].batch_size)):
            # pdb.set_trace()
            data_p, data_i = data
            data_p, data_i = data_p.to(device), data_i.to(device)
            
            # training discriminator only if accuracy < 80 % as per the paper
            model_dis.train()
            model_gen.eval()
            data_uniform = (-2) * torch.rand(data_i.shape) + -1
            data_uniform = data_uniform.to(device)
            optimizer_dis.zero_grad()
            gen_out  = model_gen(data_uniform)
            dis_real, _ = model_dis(data_i)
            dis_fake, _ = model_dis(gen_out)
            loss = DisLoss(dis_real, dis_fake)
            dis_running_loss += loss.item()
            dis_acc = torch.mean(torch.sum(torch.round(dis_real), dim=1)/dis_real.shape[1])
            if dis_acc < dis_train_cutoff:
                loss.backward()
                optimizer_dis.step()

            # training the generator
            model_dis.eval()
            model_gen.train()
            data_uniform = (-2) * torch.rand(data_i.shape) + -1
            data_uniform = data_uniform.to(device)
            optimizer_gen.zero_grad()
            gen_out = model_gen(data_uniform)
            dis_real, real_feat = model_dis(data_i)
            dis_fake, fake_feat = model_dis(gen_out)
            loss = GenLoss(real_feat, fake_feat, dis_fake)
            gen_running_loss += loss.item()
            loss.backward()
            optimizer_gen.step()

            # chamfer distance eval
            data_pred = torch.matmul(data_p, gen_out.unsqueeze(2)).view(-1, 1000, 3)
            data_pred_gt = torch.matmul(data_p, data_i.unsqueeze(2)).view(-1, 1000, 3)
            running_closs += chamfer_loss(NormalizedScale()(data_pred), NormalizedScale()(data_pred_gt))

            # visualize or save points
            if epoch > 0 :
                if i in train_save_list:
                    ret_ind = np.random.randint(0, batch_size)
                    fig = plt.figure(figsize=(15, 5))
                    viz_points(data_pred[ret_ind].detach().cpu().numpy(), 'Pred Points', fig, 121)
                    viz_points(data_pred_gt[ret_ind].detach().cpu().numpy(), 'GT Reconstruction', fig, 122)
                    plt.savefig(f'experiments/train_save/{epoch}_{i}_{ret_ind}.png')
                    h_save(f'experiments/train_save/{epoch}_{i}_{ret_ind}.h5', data_pred[ret_ind].detach().cpu())
                    plt.close()


        gen_train_loss = gen_running_loss/len(loaders['train'].dataset)
        dis_train_loss = dis_running_loss/len(loaders['train'].dataset)
        train_closs = running_closs/len(loaders['train'].dataset)
        print(f"Gen training loss: {gen_train_loss}\tDis training loss: {dis_train_loss}\tTrain Chamfer Distance: {train_closs}\n")

        # Validate the model
        gen_running_loss, dis_running_loss, running_closs = 0.0, 0.0, 0.0
        model_dis.eval()
        model_gen.eval()
        val_save_list = np.random.randint(0, int(len(valid_data)/loaders['valid'].batch_size), size=3)
        for i, data in tqdm(enumerate(loaders['valid']), total=int(len(valid_data)/loaders['valid'].batch_size)):
            # pdb.set_trace()
            data_p, data_i = data
            data_p, data_i = data_p.to(device), data_i.to(device)
            
            data_uniform = (-2) * torch.rand(data_i.shape) + -1
            data_uniform = data_uniform.to(device)
            gen_out  = model_gen(data_uniform)
            dis_real, _ = model_dis(data_i)
            dis_fake, _ = model_dis(gen_out)
            loss = DisLoss(dis_real, dis_fake)
            dis_running_loss += loss.item()

            data_uniform = (-2) * torch.rand(data_i.shape) + -1
            data_uniform = data_uniform.to(device)
            gen_out = model_gen(data_uniform)
            dis_real, real_feat = model_dis(data_i)
            dis_fake, fake_feat = model_dis(gen_out)
            loss = GenLoss(real_feat, fake_feat, dis_fake)
            gen_running_loss += loss.item()

            # chamfer distance eval
            data_pred = torch.matmul(data_p, gen_out.unsqueeze(2)).view(-1, 1000, 3)
            data_pred_gt = torch.matmul(data_p, data_i.unsqueeze(2)).view(-1, 1000, 3)
            running_closs += chamfer_loss(NormalizedScale()(data_pred), NormalizedScale()(data_pred_gt))

            # visualize or save points
            if epoch > 50 :
                if i in val_save_list:
                    ret_ind = np.random.randint(0, batch_size)
                    fig = plt.figure(figsize=(15, 5))
                    viz_points(data_pred[ret_ind].detach().cpu().numpy(), 'Pred Points', fig, 121)
                    viz_points(data_pred_gt[ret_ind].detach().cpu().numpy(), 'GT Reconstruction', fig, 122)
                    plt.savefig(f'experiments/valid_save/{epoch}_{i}_{ret_ind}.png')
                    h_save(f'experiments/valid_save/{epoch}_{i}_{ret_ind}.h5', data_pred[ret_ind].detach().cpu())
                    plt.close()


        gen_valid_loss = gen_running_loss/len(loaders['valid'].dataset)
        dis_valid_loss = dis_running_loss/len(loaders['valid'].dataset)
        valid_closs = running_closs/len(loaders['valid'].dataset)
        print(f"Gen Validation loss: {gen_valid_loss}\tDis Validation loss: {dis_valid_loss}\tValid Chamfer Distance: {valid_closs}\n")

        # create checkpoint variable and add important data
        dis_ckpt = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss_min_dis,
            'state_dict': model_dis.state_dict(),
            'optimizer': optimizer_dis.state_dict(),
        }

        gen_ckpt = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss_min_gen,
            'state_dict': model_gen.state_dict(),
            'optimizer': optimizer_gen.state_dict(),
        }

        # creating log
        with open("main_log.txt", 'a') as File:
            File.write(f"epoch: {epoch + 1}\tGen training loss: {gen_train_loss}\tDis training loss: {dis_train_loss}\tTrain Chamfer Distance: {train_closs}\tGen Validation loss: {gen_valid_loss}\tDis Validation loss: {dis_valid_loss}\tValid Chamfer Distance: {valid_closs}\n")

        # save model discriminator
        if dis_valid_loss < valid_loss_min_dis and epoch % 5 == 0:
            valid_loss_min_dis = dis_valid_loss
            print('Validation loss decreased! Saving model..')
            save_ckpt(dis_ckpt, True, dis_ckpt_path, dis_best_model_path)
        elif epoch % 5 == 0:
            save_ckpt(dis_ckpt, False, dis_ckpt_path, dis_best_model_path)

        # save model generator
        if gen_valid_loss < valid_loss_min_gen and epoch % 5 == 0:
            valid_loss_min_gen = gen_valid_loss
            print('Validation loss decreased! Saving model..')
            save_ckpt(gen_ckpt, True, gen_ckpt_path, gen_best_model_path)
        elif epoch % 5 == 0:
            save_ckpt(gen_ckpt, False, gen_ckpt_path, gen_best_model_path)


train(start_epoch, n_epochs, [valid_loss_min_gen, valid_loss_min_dis], [model_gen, model_dis], [optimizer_gen, optimizer_dis], [gen_ckpt_path, dis_ckpt_path], [gen_best_model_path, dis_best_model_path])


