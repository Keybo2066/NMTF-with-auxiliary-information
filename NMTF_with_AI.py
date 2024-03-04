import torch
import numpy as np
from tensorly import check_random_state
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from tqdm import tqdm
from tqdm import trange
from utils import *


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_nonnegative_matrices_proposed_negative(Iy, Jy, Ky, R, random_state=1234):
    torch.manual_seed(random_state)  
    while True:
        Ty = -torch.rand((Iy, R), requires_grad=True)  
        Uy = -torch.rand((Jy, R), requires_grad=True)   
        Vy = -torch.rand((Ky, R), requires_grad=True)  
        return Ty, Uy, Vy
    
def generate_nonnegative_matrices_proposed_positive(Iy, Jy, Ky, R, random_state=1234):
    torch.manual_seed(random_state)  
    while True:
        Ty = torch.rand((Iy, R), requires_grad=True)  
        Uy = torch.rand((Jy, R), requires_grad=True)   
        Vy = torch.rand((Ky, R), requires_grad=True)  
        result = torch.einsum('ir,jr,kr->ijk', Ty, Uy, Vy)
        if (result > 0).all():
            return Ty, Uy, Vy
        
def cal_time_sparsity(data, tmode):
    ''' Calculate time sparsity for dense torch.Tensor '''
    nonzero_indices = torch.nonzero(data, as_tuple=True)
    npy = [idx.cpu().numpy() for idx in nonzero_indices]
    df = pd.DataFrame(np.array(npy).T)
    r_index = df[tmode].max()
    r_index = set( np.arange(0, r_index + 1 ))
    df = df.groupby(tmode).count()
    df = pd.DataFrame(df.iloc[:, 1])
    df = df.reset_index()
    
    df.columns = [0, 1]
    subset = set(df[0])
    diff = r_index - subset
    if len(diff) != 0 :
        lst = [[i, 1] for i in diff]
        df = df.append(pd.DataFrame(lst))

    df = df.sort_values(by = 0)
    dff = df[1]
    max_, min_ = dff.max(), dff.min()

    min_max = (0.999 - 0.001) * (dff - min_) / (max_ - min_) 
    min_max = np.where(np.isnan(min_max), 1, min_max)
    return 1 - torch.FloatTensor(list(min_max + 0.001)).to(data.device)

def read_dataset(tensor):
    ''' Read data and make metadata '''
    dct = {}
    dct['tmode'] = 2 
    dct['nmode'] = len(tensor.shape)
    dct['ndim'] = max(tensor.shape)
    dct['ts'] = cal_time_sparsity(tensor, dct['tmode'])
    return dct

def euclidean_distance(p, q):
    return (((p - q) ** 2).sum())

def training(model, opt, y, a1, a2, penalty_L2,penalty_time):

    opt.zero_grad()


    y_hat = torch.einsum('ir,jr,kr->ijk', model.factors[0], model.factors[1], model.factors[2])
    a1_hat = torch.einsum('ir,jr,kr->ijk', model.factors[3], model.factors[1], model.factors[4])
    a2_hat = torch.einsum('ir,jr,kr->ijk', model.factors[0], model.factors[5], model.factors[6])

    recon_loss_y = euclidean_distance(y, y_hat)
    recon_loss_a = euclidean_distance(a1, a1_hat) + euclidean_distance(a2, a2_hat)
    smooth_loss = 0
    l2_loss = 0
    for mode in range(len(model.factors)):
        if mode == model.tmode:
            smooth_loss += penalty_time * model.smooth_reg(mode)
        else:
            l2_loss += penalty_L2 * model.l2_reg(mode)

    loss = recon_loss_y + recon_loss_a + smooth_loss + l2_loss

    loss.backward()
    opt.step()

    return loss.item(), recon_loss_y.item(), recon_loss_a.item(), smooth_loss.item(), l2_loss.item(),model.factors[2]


def calculate_rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))