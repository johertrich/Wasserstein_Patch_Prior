# This code belongs to the paper
#
# J. Hertrich, A. Houdard and C. Redenbach.
# Wasserstein Patch Prior for Image Superresolution.
# ArXiv Preprint#2109.12880
#
# Please cite the paper, if you use this code.
#
# This file contains the core functions for the reconstruction using the Wasserstein Patch Prior.
#
import torch
from torch import nn
from torch.autograd.variable import Variable
import numpy as np
import math
import time
from os import mkdir
from os.path import isdir
import skimage.io as io
import scipy.io
from pykeops.torch import LazyTensor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

def imread(img_name):
    """
    loads an image as torch.tensor on the selected device
    """ 
    np_img = io.imread(img_name)
    tens_img = torch.tensor(np_img, dtype=torch.float, device=DEVICE)
    if torch.max(tens_img) > 1:
        tens_img/=255
    if len(tens_img.shape) < 3:
        tens_img = tens_img.unsqueeze(2)
    if tens_img.shape[2] > 3:
        tens_img = tens_img[:,:,:3]
    tens_img = tens_img.permute(2,0,1)
    return tens_img.unsqueeze(0)

def imsave(save_name, tens_img):
    """
    save a tensor image
    """ 
    np_img = np.clip(tens_img.squeeze(0).permute(1,2,0).data.cpu().numpy(), 0,1)
    if np_img.shape[2] < 3:
        np_img = np_img[:,:,0]
    io.imsave(save_name, np_img)
    return 

def init_weights( kernel_size, sigma,dim):
    """
    creates an gaussian blur kernel of size kernel_size, standard deviation sigma and dimension dim
    """
    if dim==2:
        x_cord = torch.arange(kernel_size,device=DEVICE)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    elif dim==3:
        x_grid = torch.arange(kernel_size,device=DEVICE).view(kernel_size,1,1).repeat(1,kernel_size,kernel_size)
        y_grid = x_grid.permute(1,0,2)
        z_grid = x_grid.permute(2,1,0)
        xy_grid = torch.stack([x_grid,y_grid,z_grid],dim=-1)
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    if type(variance)==float:
        variance=torch.tensor(variance*np.ones(dim),dtype=torch.float,device=DEVICE)
    if len(variance.shape)==0:
        variance=variance*torch.ones(dim,dtype=torch.float,device=DEVICE)
    gaussian_kernel = (1./(2.*math.pi*torch.prod(variance))**(.5*dim))*torch.exp(-torch.sum((xy_grid - mean)**2./variance[np.newaxis,np.newaxis,:], dim=-1)/(2))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    if dim==2:
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    elif dim==3:
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size,kernel_size)

class gaussian_downsample(nn.Module):
    """
    Downsampling module with Gaussian filtering (strided convolution)
    """ 
    def __init__(self, kernel_size, sigma, stride, pad=False, dim=2,bias=False):
        super(gaussian_downsample, self).__init__()
        self.dim=dim
        if dim==2:
            self.gauss = nn.Conv2d(1, 1, kernel_size, stride=stride, groups=1, bias=bias)        
        elif dim==3:
            self.gauss = nn.Conv3d(1, 1, kernel_size, stride=stride, groups=1, bias=bias)        
        gaussian_weights = self.init_weights(kernel_size, sigma)
        self.gauss.weight.data = gaussian_weights.to(DEVICE)
        self.gauss.weight.requires_grad_(False)
        self.pad = pad
        self.padsize = kernel_size-1

    def forward(self, x):
        if self.pad and dim==2:
            x = torch.cat((x, x[:,:,:self.padsize,:]), 2)
            x = torch.cat((x, x[:,:,:,:self.padsize]), 3)
        elif self.pad and dim==3:
            x = torch.cat((x, x[:,:,:self.padsize,:,:]), 2)
            x = torch.cat((x, x[:,:,:,:self.padsize,:]), 3)
            x = torch.cat((x, x[:,:,:,:,:self.padsize]), 4)
        return self.gauss(x)

    def init_weights(self,kernel_size,sigma):
        return init_weights(kernel_size,sigma,self.dim)

class semidual(nn.Module):
    """
    Computes the semi-dual loss between inputy and inputx for the dual variable psi
    """    
    def __init__(self, inputy, usekeops=False):
        super(semidual, self).__init__()
        self.psi = nn.Parameter(torch.zeros(inputy.shape[0], device=DEVICE))
        self.yt = inputy.transpose(1,0)
        self.usekeops = usekeops
        self.y2 = torch.sum(self.yt **2,0,keepdim=True)
    def forward(self, inputx):
        if self.usekeops:
            y = self.yt.transpose(1,0)
            x_i = LazyTensor(inputx.unsqueeze(1).contiguous())
            y_j = LazyTensor(y.unsqueeze(0).contiguous())
            v_j = LazyTensor(self.psi.unsqueeze(0).unsqueeze(2).contiguous())
            sx2_i = LazyTensor(torch.sum(inputx**2,1,keepdim=True).unsqueeze(2).contiguous())
            sy2_j = LazyTensor(self.y2.unsqueeze(2).contiguous())
            rmv = sx2_i + sy2_j - 2*(x_i*y_j).sum(-1) - v_j
            amin = rmv.argmin(dim=1).view(-1)
            psi_conjugate_mean = torch.mean(torch.sum((inputx-y[amin,:])**2,1)-self.psi[amin]) 
            psi_mean = torch.mean(self.psi)
        else:
            cxy = torch.sum(inputx**2,1,keepdim=True) + self.y2 - 2*torch.matmul(inputx,self.yt)
            psi_conjugate_mean = torch.mean(torch.min(cxy - self.psi.unsqueeze(0),1)[0]) 
            psi_mean = torch.mean(self.psi)
        return psi_conjugate_mean,psi_mean
    
class gaussian_layer(nn.Module): 
    """
    Gaussian layer for the dowsampling pyramid (strided convolution)
    """ 	   
    def __init__(self, gaussian_kernel_size, gaussian_std, stride = 2, pad=False,dim=2):
        super(gaussian_layer, self).__init__()
        self.dim=dim
        self.downsample = gaussian_downsample(gaussian_kernel_size, gaussian_std, stride, pad=pad,dim=self.dim)
    def forward(self, input):
        self.down_img = self.downsample(input)
        return self.down_img

class identity(nn.Module):  
    """
    Identity layer for the dowsampling pyramid
    """   
    def __init__(self):
        super(identity, self).__init__()
    def forward(self, input):
        self.down_img = input
        return input

def create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, n_scales, stride = 2, pad=False,dim=2):
    """
    Create a dowsampling Gaussian pyramid
    """ 
    layer = identity()
    gaussian_pyramid = nn.Sequential(layer)
    for i in range(n_scales-1):
        layer = gaussian_layer(gaussian_kernel_size, gaussian_std, stride, pad=pad,dim=dim)
        gaussian_pyramid.add_module("Gaussian_downsampling_{}".format(i+1), layer)
    return gaussian_pyramid

class patch_extractor(nn.Module):   
    """
    Module for creating custom patch extractor
    """ 
    def __init__(self, patch_size, pad=False,center=False,dim=2):
        super(patch_extractor, self).__init__()
        self.dim=dim
        self.im2pat = nn.Unfold(kernel_size=patch_size)
        self.pad = pad
        self.padsize = patch_size-1
        self.center=center
        self.patch_size=patch_size

    def forward(self, input, batch_size=0,split=[1,0]):
        if self.pad and self.dim==2:
            input = torch.cat((input, input[:,:,:self.padsize,:]), 2)
            input = torch.cat((input, input[:,:,:,:self.padsize]), 3)
        elif self.pad and self.dim==3:
            input = torch.cat((input, input[:,:,:self.padsize,:,:]), 2)
            input = torch.cat((input, input[:,:,:,:self.padsize,:]), 3)
            input = torch.cat((input, input[:,:,:,:,:self.padsize]), 4)
        if self.dim==2:
            patches = self.im2pat(input).squeeze(0).transpose(1,0)
            split_size=patches.size(0)//split[0]
            if split[1]==split[0]-1:
                patches=patches[split_size*split[1]:]
            else:
                patches=patches[split_size*split[1]:split_size*(split[1]+1)]
        elif self.dim==3:
            patches = self.im2pat(input[0]).squeeze(0).transpose(1,0).reshape(-1,input.shape[2],self.patch_size,self.patch_size)
            split_size=patches.size(0)//split[0]
            if split[1]==split[0]-1:
                patches=patches[split_size*split[1]:]
            else:
                patches=patches[split_size*split[1]:split_size*(split[1]+1)]
            patches = patches.unfold(1,self.patch_size,self.stride).permute(0,1,4,2,3)
            patches = patches.reshape(-1,self.patch_size**3)
        if batch_size > 0:
            idx = torch.randperm(patches.size(0))[:batch_size]
            patches = patches[idx,:]
        if self.center:
            patches = patches - torch.mean(patches,-1).unsqueeze(-1)
        return patches

def optim_synthesis(args,operator,lr_img,learn_img,lam,init=None,add_boundary=0,center=False,dim=2,n_splits=1):
    """
    Reconstruction of an image by minizing 
    .5*||operator(reconstruction)-lr_img||^2+lam*Wasserstein_patch_prior(reconstruction)
    Parameters:
    - args:
        arguments containing: 
        - args.patch_size           - patch size within the Wasserstein Patch Prior
        - args.n_iter_max           - number of iterations
        - args.n_patches_in         - number of patches collected from the reconstruction for the WPP
        - args.n_patches_out        - number of patches collected from the reference image for the WPP
        - args.scales               - number of scales 
        - args.keops                - True for using pykeops, otherwise False
    - operator:
        forward operator. Function which supports torch gradients.
    - lr_img:
        observation should be approx. operator(ground_truth)+noise
    - learn_img:
        reference image
    - lam:
        weighting of the WPP
    - init:
        initialization. None for uniform random initialization
    - add_boundary:
        added boundary as in section 3.3 of the paper
    - center:
        True for substracting the mean of each patch before applying the Wasserstein patch prior, otherwise False
    - dim:
        dimension of the images. Default 2. Supported: 2 or 3
    - n_splits:
        number of splittings of the patches to reduce the memory requirement on the gpu. Default: 1
        For the 3D examples values higher than 1 might be necessary.
        This parameter has no effect on the result.

    Returns:
        reconstruction
    """
    patch_size = args.patch_size
    n_iter_max = args.n_iter_max
    n_iter_psi = args.n_iter_psi
    n_patches_in = args.n_patches_in
    n_patches_out = args.n_patches_out
    n_scales = args.scales
    usekeops = args.keops
    save = True
    target_img=learn_img

    
    # fixed parameters
    monitoring_step=50
    saving_folder='tmp/'
    
    # parameters for Gaussian downsampling
    gaussian_kernel_size = 4
    gaussian_std = 1
    stride = 2
    
    
    # synthesized size
    nrow = args.size[0]
    ncol = args.size[1]
    
    if save:
        if not isdir(saving_folder):
            mkdir(saving_folder)
        if dim==3:
            print('3D! Not saving the initialization!')
        else:
            imsave(saving_folder+'original.png', target_img)


    # Create Gaussian Pyramid downsamplers
    lr_downsampler = create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, n_scales+1, stride, pad=False,dim=dim)                  
    target_downsampler = create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, n_scales+1, stride, pad=False,dim=dim)                  
    input_downsampler = create_gaussian_pyramid(gaussian_kernel_size, gaussian_std, n_scales+1, stride, pad=False,dim=dim)

    target_downsampler(target_img) # evaluate on the target image
    lr_downsampler(lr_img)

    # create patch extractors
    target_im2pat = patch_extractor(patch_size, pad=False,center=center,dim=dim)
    input_im2pat = patch_extractor(patch_size, pad=False,center=center,dim=dim)

    # create semi-dual module at each scale
    semidual_loss = []
    for s in range(n_scales):        
        real_data = target_im2pat(target_downsampler[s].down_img, n_patches_out) # exctract at most n_patches_out patches from the downsampled target images 
        num_real_data=real_data.shape[0]
        layer = semidual(real_data, usekeops=usekeops)
        semidual_loss.append(layer)

    # Weights on scales
    prop = torch.ones(n_scales, device=DEVICE)/n_scales # all scales with same weight
    
    # initialize the generated image
    if init is None:
        fake_img = torch.rand(1, 1, nrow,ncol, device=DEVICE).requires_grad_()
    else:
        fake_img = torch.tensor(init[np.newaxis,np.newaxis,:,:],dtype=torch.float,device=DEVICE).requires_grad_()
    
    down_shape=lr_downsampler[n_scales].down_img.shape
    lr_img=lr_img.to(DEVICE)
    # intialize optimizer for image
    optim_img = torch.optim.Adam([fake_img], lr=0.01)
    # initialize the loss vector
    total_loss = np.zeros(n_iter_max)
    # Main loop
    t = time.time()
    num_patches=np.prod([s-patch_size+1 for s in args.size])
    for it in range(n_iter_max):
        # 1. update psi
        input_downsampler(fake_img.detach()) # evaluate on the current fake image
        for s in range(n_scales):  
            optim_psi = torch.optim.ASGD([semidual_loss[s].psi], lr=1, alpha=0.5, t0=1)
            # set number of patches
            num_patches=np.prod([size-patch_size+1 for size in input_downsampler[s].down_img.shape[2:]])
            # iterate over update steps of psi
            for i in range(n_iter_psi):
                optim_psi.zero_grad()
                loss=0.
                # iterate over the patch batches
                for split in range(n_splits):
                    fake_data = input_im2pat(input_downsampler[s].down_img, n_patches_in,split=[n_splits,split])
                    psi_conjugate_mean,psi_mean = semidual_loss[s](fake_data)
                    # add the loss of the psi conjugates
                    loss_conj=-psi_conjugate_mean*fake_data.shape[0]/num_patches
                    loss_conj.backward()
                    loss+=loss_conj.item()
                # add the mean of the psi to the loss
                loss_mean=-psi_mean
                loss_mean.backward()
                loss+=loss_mean.item()
                # optimizer step
                optim_psi.step()
            semidual_loss[s].psi.data = optim_psi.state[semidual_loss[s].psi]['ax']
        # 2. perform gradient step on the image
        optim_img.zero_grad()
        tloss = 0
        # iterate over scales
        for s in range(n_scales):
            num_patches=np.prod([size-patch_size+1 for size in input_downsampler[s].down_img.shape[2:]])
            loss=0.
            # iterate over all batches of patches
            for split in range(n_splits):
                input_downsampler(fake_img)    
                fake_data = input_im2pat(input_downsampler[s].down_img, n_patches_in,split=[n_splits,split])
                psi_conjugate_mean,psi_mean=semidual_loss[s](fake_data)
                loss_conj=prop[s]*psi_conjugate_mean*fake_data.shape[0]/num_patches
                loss_conj.backward()
                loss+=loss_conj.item()
            loss_mean = prop[s]*psi_mean
            loss_mean.backward()
            loss+=loss_mean.item()
            tloss += loss
        wloss=tloss
        # add operator to the loss
        loss=1./lam/n_scales*torch.sum((operator(fake_img)-lr_img)**2)
        oloss=loss.item()
        loss.backward()
        tloss+=loss.item()
        optim_img.step()
        # save loss
        total_loss[it] = tloss

        # save some results
        if it % monitoring_step == 0:
            print('iteration '+str(it)+' - elapsed '+str(int(time.time()-t))+'s - loss = '+str(tloss)+', wloss = '+str(wloss)+', oloss = '+str(oloss))
            if save:
                if add_boundary==0:
                    fake_img2=fake_img
                else:
                    if dim==2:
                        fake_img2=fake_img[:,:,add_boundary:-add_boundary,add_boundary:-add_boundary]
                    elif dim==3:
                        fake_img2=fake_img[:,:,add_boundary:-add_boundary,add_boundary:-add_boundary,add_boundary:-add_boundary]
                if dim==2:
                    imsave(saving_folder+'it'+str(it)+'.png', fake_img2)
                elif dim==3:
                    scipy.io.savemat(saving_folder+'it'+str(it)+'.mat',{'hr_img':fake_img2.detach().squeeze().cpu().numpy()})
                    io.imsave(saving_folder+'it'+str(it)+'.tif',np.round(np.minimum(np.maximum(fake_img2.detach().squeeze().cpu().numpy(),0),1)*255.).astype(np.uint8))

    print('DONE - total time is '+str(int(time.time()-t))+'s')
    
    return fake_img



