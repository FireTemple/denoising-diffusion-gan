# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import argparse
import math
import torch
import numpy as np

import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from datasets_prep.edges2shoes import Edge2Shoes
from datasets_prep.lsun import LSUN
from datasets_prep.stackmnist_data import StackedMNIST, _data_transforms_stacked_mnist
from datasets_prep.lmdb_datasets import LMDBDataset


from torch.multiprocessing import Process
import torch.distributed as dist
import shutil

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))
            
def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

class Diffusion_Coefficients():
    def __init__(self, args, device):
                
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)
    
def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)
      
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    
    return x_t

def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    
    return x_t, x_t_plus_one

# TODO A-bridge forward
def q_sample_pairs_for_a_bridge(x_start, t_menus_1, y, T):

    assert y is not None, 'Condition is required for forward diffusion.'
    c_lambda = 2
    t = t_menus_1 + 1
    # print(f"t_menus_1: {t_menus_1}")
    # TODO 移到外面去
    # t = torch.randint(2, T + 1, (batch_size,), device=x_start.device).long() if t is None else t.long()
    t_menus_1 = t_menus_1.unsqueeze(1).unsqueeze(2).unsqueeze(3) # BCWH
    t = t.unsqueeze(1).unsqueeze(2).unsqueeze(3) # BCWH
    m_t_menus_1 = t_menus_1 / T
    # step2 create noise
    noise_xt_menus_1 = torch.randn_like(x_start, device=x_start.device)

    # B_t_menus_1 = c_lambda * (1 - m_t_menus_1) * (torch.log(1 / (1 - m_t_menus_1))) ** 0.5
    B_t_menus_1 = c_lambda * (1 - m_t_menus_1) * (torch.log(1 / (1 - m_t_menus_1))) ** 0.5

    # B_t_menus_1 = torch.where(torch.eq(t_menus_1, T), torch.zeros_like(B_t_menus_1), B_t_menus_1)
    B_t_menus_1 = torch.nan_to_num(B_t_menus_1)

    x_t_menus_1 = (1 - m_t_menus_1) * x_start + m_t_menus_1 * y +  B_t_menus_1 * noise_xt_menus_1
    # x_t_menus_1 = torch.where(torch.eq(m_t_menus_1, 0), x_start, x_t_menus_1)
    # objective = m_t * (y - x_start) + B_t * noise
    noise_x_t = torch.randn_like(x_t_menus_1, device=x_t_menus_1.device)

    x_t_mean = ((T - t) / (T - t + 1)) * x_t_menus_1 + (1 / (T - t + 1)) * y

    x_t_var = c_lambda * (1 - t / T) * torch.sqrt(torch.log((T - t + 1) / (T - t))) 
    # 当t = T 的时候var = 0
    x_t_var = torch.nan_to_num(x_t_var)


    x_t = x_t_mean + x_t_var * noise_x_t

    assert not torch.isnan(B_t_menus_1).any(), "NaN detected in B_t_menus_1"
    assert not torch.isnan(x_t_menus_1).any(), "NaN detected in x_t_menus_1"
    assert not torch.isnan(x_t_mean).any(), "NaN detected in x_t_mean"
    assert not torch.isnan(x_t_var).any(), "NaN detected in x_t_var"
    assert not torch.isnan(x_t).any(), "NaN detected in x_t"
    # print(x_t_menus_1)
    return x_t_menus_1, x_t
#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
def sample_posterior(coefficients, x_0,x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos

def sample_posterior_A_bridge(x_0, x_t, t_menus_1, y, T):
        
    def q_posterior(x_0, x_t, t, y):
        # print(x_0)
        t = t.unsqueeze(1).unsqueeze(2).unsqueeze(3) # BCWH
        m_t = t / T
        c_lambda = 2
        beta_t = T - t + 1
        gamma_t = torch.log(T / beta_t)
        C = (1 - 1 / beta_t + 1 / (beta_t * gamma_t))
        c_xt = (1 / C * (1 + 1 / (T * gamma_t)))
        c_yt = (1 / C * (1 / beta_t - (t - 1) / (T * beta_t * gamma_t)))
        c_epst = (1 / C * (1 / (T * gamma_t)))
        c_zt = (c_lambda / C * (1 - m_t + 1 / T) ** 0.5 * (1 / T) ** 0.5)
 
        C = torch.nan_to_num(C)
        c_xt = torch.nan_to_num(c_xt)
        c_yt = torch.nan_to_num(c_yt)
        c_epst = torch.nan_to_num(c_epst)
        c_zt = torch.nan_to_num(c_zt)
        # B_t = c_lambda * (1 - m_t) * (torch.log(1 / (1 - m_t))) ** 0.5
        # B_t = torch.where(t == T, torch.zeros_like(B_t), B_t)

        mean = (
            (c_xt - c_epst) * x_t + c_epst * x_0 - c_yt * y
        )
        var = c_zt
        # print("before")
        # print(mean)

        mean = torch.nan_to_num(mean)
        mean = torch.where(t == 1, x_t - (m_t * (y - x_0)), mean)
        var = torch.where(t == 1, torch.zeros_like(var), var)
        # print("after")
        # mean = torch.where(t == T, 0.9998552 * x_t + 0.0001447648 * (x_t - (m_t * (y - x_0))), mean)
        # var = torch.where(t == T, 0.0014142 , var)

        # if t == 1:
        #     mean = (
        #         x_t - (m_t * (y - x_0))
        #     )
        #     var = 0
        # if t == T:
        #     mean = (
        #         0.9998552 * x_t + 0.0001447648 * (x_t - (m_t * (y - x_0)))
        #     ) 
        #     var = 0.0014142 
        # else:    
        #     mean = (
        #         c_xt * x_t - c_yt * y - c_epst * (m_t * (y - x_0) + B_t)
        #     )
        #     var = c_zt

        return mean, var
    
    
  
        
  
    def p_sample(x_0, x_t, t):
        mean, var = q_posterior(x_0, x_t, t, y)
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 1).type(torch.float32))
        
        return mean - nonzero_mask[:, None, None, None] * var * noise
    
    sample_x_pos = p_sample(x_0, x_t, t_menus_1 + 1)
    # print(sample_x_pos)
    return sample_x_pos

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()               

def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init
    x_steps = [x[0].clone()]
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
            x_steps.append(torch.clamp((x[0].clone()), 0, 1))
    x_steps.append(x[0].clone())        
    return x, x_steps

def sample_from_model_A_bridge(generator, n_time, x_init, opt, y):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(1 , n_time + 1)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(x, t_time, latent_z, y)
            x_new = sample_posterior_A_bridge(x_0, x, t, y, args.num_timesteps)
            x = x_new.detach()
    return x

#%%
def train(rank, gpu, args):
    from score_sde.models.discriminator import Discriminator_small, Discriminator_large
    from score_sde.models.ncsnpp_generator_adagn import NCSNpp
    from EMA import EMA
    
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    
    batch_size = args.batch_size
    
    nz = args.nz #latent dimension
    
    if args.dataset == 'cifar10':
        dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
       
    
    elif args.dataset == 'stackmnist':
        train_transform, valid_transform = _data_transforms_stacked_mnist()
        dataset = StackedMNIST(root='./data', train=True, download=False, transform=train_transform)
        
    elif args.dataset == 'lsun':
        
        train_transform = transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.CenterCrop(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                    ])

        train_data = LSUN(root='/datasets/LSUN/', classes=['church_outdoor_train'], transform=train_transform)
        subset = list(range(0, 120000))
        dataset = torch.utils.data.Subset(train_data, subset)
      
    
    elif args.dataset == 'celeba_256':
        train_transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        dataset = LMDBDataset(root='/datasets/celeba-lmdb/', name='celeba', train=True, transform=train_transform)

    elif args.dataset == 'edges2shoes':
        # train_transform = transforms.Compose([
        #         transforms.Resize(args.image_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        #     ])
        dataset = Edge2Shoes("/mnt/c/Users/Public/Documents/Datasets/edge2shoes", batch_size, 32, device="cuda:0", split="train")
        dataset_val = Edge2Shoes("/mnt/c/Users/Public/Documents/Datasets/edge2shoes", batch_size, 32, device="cuda:0", split="val")
      
    
    print(len(dataset))
    # 固定写法
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last = True)
    
    
    # 生成器网络 TODO 看看具体
    netG = NCSNpp(args).to(device)
    

    # 判别器网络，这里我们大概率用large
    # TODO changed to 3
    if args.dataset == 'cifar10' or args.dataset == 'stackmnist' or args.dataset == 'edges2shoes':    
        netD = Discriminator_small(nc = 2*args.num_channels, ngf = args.ngf,
                               t_emb_dim = args.t_emb_dim,
                               act=nn.LeakyReLU(0.2)).to(device)
    else:
        # num_channels = 3
        # nc = 6 
        # ngf = 64
        # t_emb_dim = 256
        # TODO changed to 3323123123123
        netD = Discriminator_large(nc = 2*args.num_channels, ngf = args.ngf, 
                                   t_emb_dim = args.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(device)
    
    # TODO 看一下做什么的
    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())
    
    # Adam优化器
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    


    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)
    
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch, eta_min=1e-5)
    
    
    
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])

    
    # 保存结果
    exp = args.exp
    parent_dir = "./saved_info/dd_gan/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir,exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('score_sde/models', os.path.join(exp_path, 'score_sde/models'))
    
    
    coeff = Diffusion_Coefficients(args, device)
    print(coeff)
    pos_coeff = Posterior_Coefficients(args, device)
    print("=================")
    print(pos_coeff)
    T = get_time_schedule(args, device)
    
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        netG.load_state_dict(checkpoint['netG_dict'])
        # load G
        
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0
    
    
    for epoch in range(init_epoch, args.num_epoch+1):
        train_sampler.set_epoch(epoch)

        # for name, param in netG.named_parameters():
        #     if param.grad is not None:
        #         # print(f"sGenerator gradient {name}: {param.grad.norm()}")
        #         print(f"Generator gradient {name}: {param.grad.norm()} iteration: {iteration}")

        for iteration, (x, y) in enumerate(data_loader):
            for p in netD.parameters():  
                p.requires_grad = True  

            assert not torch.isnan(x).any(), "NaN detected in input data"
            assert not torch.isnan(y).any(), "NaN detected in target data"
            # print(f'iteration: {iteration} =============================')
            # print(f'x shape: {x.shape} =============================')

            # 新的循环，梯度清零
            netD.zero_grad()
            
            #sample from p(x_0)
            real_data = x.to(device, non_blocking=True)
            
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            # 这里采样得到 x_t 和 x_t+1 TODO 这里替换成A-bridge
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)

            # TODO 这里为什么要？：因为xt在后面座位输入放到了D中
            x_t.requires_grad = True
            
    
            # train with real
            # TODO 研究一下具体怎么计算的
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)
            
            errD_real = F.softplus(-D_real)
            errD_real = errD_real.mean()
            assert not torch.isnan(errD_real).any(), "NaN detected in discriminator real loss value"
            errD_real.backward(retain_graph=True)
            
            
            # TODO lazy_reg?
            if args.lazy_reg is None:
                grad_real = torch.autograd.grad(
                            outputs=D_real.sum(), inputs=x_t, create_graph=True
                            )[0]
                grad_penalty = (
                                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()
                
                
                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad_real = torch.autograd.grad(
                            outputs=D_real.sum(), inputs=x_t, create_graph=True
                            )[0]
                    grad_penalty = (
                                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                                ).mean()
                
                
                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()

            # train with fake
            latent_z = torch.randn(batch_size, nz, device=device)
         

            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            errD_fake = F.softplus(output)
            errD_fake = errD_fake.mean()
            assert not torch.isnan(errD_fake).any(), "NaN detected in discriminator fake loss value"
            errD_fake.backward()
    
            
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
     
            #update G
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            
            
            # t_menus_1 = torch.randint(0, 3, (real_x.size(0),), device=device)
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            
            
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            
            latent_z = torch.randn(batch_size, nz,device=device)
            
            
                
           
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)


            output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)
            # print(f"netD output: {output}")
            
            errG = F.softplus(-output)
            # print(f"Softplus output: {errG}")
            errG = errG.mean()
            assert not torch.isnan(errG).any(), "NaN detected in generator loss value"

            # print(f"Loss value before backward: {errG.item()}")
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            errG.backward()

            # for name, param in netG.named_parameters():
            #     if param.grad is not None:
            #         print(f"Generator gradient {name} before update: {param.grad.norm()}")
            #         if torch.isnan(param.grad).any():
            #             raise ValueError(f"NaN detected in gradient of {name}")
            


            
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            optimizerG.step()
            
            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    print('epoch {} iteration{}, G Loss: {}, D Loss: {}'.format(epoch,iteration, errG.item(), errD.item()))
        
        if not args.no_lr_decay:
            
            schedulerG.step()
            schedulerD.step()
        
        if rank == 0:
            if epoch % 10 == 0:
                torchvision.utils.save_image(x_pos_sample, os.path.join(exp_path, 'xpos_epoch_{}.png'.format(epoch)), normalize=True)
                torchvision.utils.save_image(x_0_predict, os.path.join(exp_path, 'x_0_predict_epoch_{}.png'.format(epoch)), normalize=True)
                x_pos_sample_real = sample_posterior(pos_coeff, real_data, x_tp1, t)
                torchvision.utils.save_image(x_pos_sample_real, os.path.join(exp_path, 'x_pos_sample_real_{}.png'.format(epoch)), normalize=True)
                torchvision.utils.save_image(x_0_predict, os.path.join(exp_path, 'x_0_predict_epoch_{}.png'.format(epoch)), normalize=True)

            x_t_1 = torch.randn_like(real_data[:1])
            
            fake_sample, x_steps = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args)
            steps = torch.cat(x_steps, dim=2)
            torchvision.utils.save_image(steps, os.path.join(exp_path, 'steps_{}.png'.format(epoch)), normalize=True)
            torchvision.utils.save_image(fake_sample, os.path.join(exp_path, 'fake_sample_epoch_{}.png'.format(epoch)), normalize=True)
            
            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                               'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                               'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
                    
                    torch.save(content, os.path.join(exp_path, 'content.pth'))
                
            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                    
                torch.save(netG.state_dict(), os.path.join(exp_path, 'netG_{}.pth'.format(epoch)))
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
            


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6021'
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()  

def cleanup():
    dist.destroy_process_group()    
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    
    parser.add_argument('--resume', action='store_true',default=False)
    
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--num_channels', type=int, default=3,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    
    
    parser.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')
    
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    # parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)
    
    parser.add_argument('--use_ema', action='store_true', default=False,
                            help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    
    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    parser.add_argument('--save_content', action='store_true',default=False)
    parser.add_argument('--save_content_every', type=int, default=50, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=25, help='save ckpt every x epochs')
   
    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')

   
    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        print('starting in debug mode')
        
        init_processes(0, size, train, args)