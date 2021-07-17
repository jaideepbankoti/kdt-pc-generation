import os
import torch
import torch.nn as nn
import pdb


# class definition for chamfer loss
class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        if x.get_device() != -1:
            diag_ind_x = diag_ind_x.cuda(x.get_device())
            diag_ind_y = diag_ind_y.cuda(x.get_device())
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2

def GenLoss(real_feat, fake_feat, dis_fake):
    # pdb.set_trace()
    # real_label = torch.ones_like(dis_fake)
    mean_loss = torch.linalg.norm(torch.mean(real_feat, dim=1) - torch.mean(fake_feat, dim=1)) ** 2
    real_cov = torch.matmul((real_feat - torch.mean(real_feat, dim=1).unsqueeze(1)).unsqueeze(2), (real_feat - torch.mean(real_feat, dim=1).unsqueeze(1)).unsqueeze(1))     # (B, feat, 1) * (B, 1, feat) = (B, feat, feat)
    real_cov = torch.mean(real_cov.reshape(real_feat.shape[0], -1), dim=1)
    fake_cov = torch.matmul((fake_feat - torch.mean(fake_feat, dim=1).unsqueeze(1)).unsqueeze(2), (fake_feat - torch.mean(fake_feat, dim=1).unsqueeze(1)).unsqueeze(1))
    fake_cov = torch.mean(fake_cov.reshape(fake_feat.shape[0], -1), dim=1)
    cov_loss = torch.linalg.norm(real_cov - fake_cov) ** 2
    # fake_loss = nn.BCEWithLogitsLoss()(real_label, dis_fake)
    return mean_loss + cov_loss


def DisLoss(real_feat, fake_feat):
    # pdb.set_trace()
    real_residual = torch.mean(- torch.log(real_feat), dim=1)
    fake_residual = torch.mean(- torch.log(1 - fake_feat), dim=1)
    return torch.mean(real_residual + fake_residual)

"""
def DisLoss(real_feat, fake_feat):
    real_label = torch.ones_like(real_feat)
    fake_label = torch.zeros_like(fake_feat)
    real_loss  = nn.BCEWithLogitsLoss()(real_label, real_feat)
    fake_loss  = nn.BCEWithLogitsLoss()(fake_label, fake_feat)
    return real_loss +  fake_loss
"""

def DisLoss_real(real_feat):
    real_label = torch.ones_like(real_feat)
    real_loss  = nn.BCEWithLogitsLoss()(real_label, real_feat)
    return real_loss

def DisLoss_fake(fake_feat):
    fake_label = torch.zeros_like(fake_feat)
    fake_loss  = nn.BCEWithLogitsLoss()(fake_label, fake_feat)
    return fake_loss


