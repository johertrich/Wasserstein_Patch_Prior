# This code belongs to the paper
#
# J. Hertrich, A. Houdard and C. Redenbach.
# Wasserstein Patch Prior for Image Superresolution.
# IEEE Transactions on Computational Imaging, 2022.
#
# Please cite the paper, if you use this code.
#
# This script applies the Wasserstein Patch Prior reconstruction onto the 2D SiC Diamonds image
# from Section 4.2 of the paper.
#
import argparse
import wgenpatex
import torch
import skimage.transform
import numpy as np
from scipy.interpolate import griddata
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

# set arguments
args=argparse.Namespace()
args.target_image_path='input_imgs/img_hr.png'
args.scales=2
args.keops=True
args.n_iter_max=500
args.save=True
args.n_patches_out=10000
args.learn_image_path='input_imgs/img_learn.png'
args.patch_size=6
args.lam=6000/args.patch_size**2
args.n_iter_psi=10
args.n_patches_in=-1
args.visu=False


# define forward operator
blur_width=2.0
add_boundary=20
kernel_size=16
stride=4
my_layer=wgenpatex.gaussian_layer(kernel_size,blur_width,stride=stride)

def operator(inp):
    if add_boundary==0:
        return my_layer.forward(inp)
    return my_layer.forward(inp[:,:,add_boundary:-add_boundary,add_boundary:-add_boundary])

# read HR ground truth
lr_img=wgenpatex.imread(args.target_image_path)
args.size=lr_img.shape[2:4]
lr_img_=np.zeros((lr_img.shape[2]+2*add_boundary,lr_img.shape[3]+2*add_boundary))
if add_boundary>0:
    lr_img_[add_boundary:-add_boundary,add_boundary:-add_boundary]=lr_img.squeeze().cpu().numpy()
else:
    lr_img_=lr_img.squeeze().cpu().numpy()

# create (artificially) LR observation
lr_img=operator(torch.tensor(lr_img_,dtype=torch.float,device=DEVICE).view(1,1,lr_img_.shape[0],lr_img_.shape[1]))
lr_img+=0.01*torch.randn_like(lr_img)
wgenpatex.imsave('input_imgs/lr_diam.png',lr_img)


# build initialization by rescaling the lr observation and extending it to the boundary
upscaled=skimage.transform.resize(lr_img.squeeze().cpu().numpy(),[lr_img.shape[2]*stride,lr_img.shape[3]*stride])
diff=args.size[0]-upscaled.shape[0]

init=np.zeros(args.size,dtype=bool)
init[diff//2:-diff//2,diff//2:-diff//2]=True
grid_x=np.array(range(init.shape[0]))
grid_x=np.tile(grid_x[:,np.newaxis],[1,init.shape[1]])
grid_y=np.array(range(init.shape[1]))
grid_y=np.tile(grid_y[np.newaxis,:],[init.shape[0],1])
points_x=np.reshape(grid_x[init],[-1])
points_y=np.reshape(grid_y[init],[-1])
values=np.reshape(upscaled,[-1])
points=np.stack([points_x,points_y],0).transpose()
init=griddata(points,values,(grid_x,grid_y),method='nearest')
init_=np.random.uniform(size=(init.shape[0]+2*add_boundary,init.shape[1]+2*add_boundary))
if add_boundary==0:
    init_=init
else:
    init_[add_boundary:-add_boundary,add_boundary:-add_boundary]=init
args.size=init_.shape

# load learn img
learn_img=wgenpatex.imread(args.learn_image_path)

# run reconstruction
synth_img= wgenpatex.optim_synthesis(args,operator,lr_img,learn_img,args.lam,init=init_,add_boundary=add_boundary)


# save reconstruction
if not os.path.isdir('output_imgs_diam'):
    os.mkdir('output_imgs_diam')
if add_boundary>0:
    synth_img=synth_img[:,:,add_boundary:-add_boundary,add_boundary:-add_boundary]
wgenpatex.imsave('output_imgs_diam/synthesized_diam.png', synth_img)



