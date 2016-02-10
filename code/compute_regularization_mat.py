#!/usr/bin/env python

# This is the code to compute regularization matrices based on roi mask used.
#1. D: 
# Euclidean distance matrix 
# (voxel*voxel)

# 2. E: 
# Connectivity matrix (generated from D)
# (edge*voxel)

# 3. G:
# Gaussian kernel matrix on D
# alpha*I-exp(-gamma*D^2)
# (voxel*voxel)

# 4. C:
# E'*E (generated from E)
# (voxel*voxel)

# In the algorithm, you can choose to use G or C

# By Hejia Zhang @Princeton University

import scipy.io
import os,sys,copy
import numpy as np
import nibabel as nib
import random
import argparse
import pprint


## argument parsing
usage = '%(prog)s roi -s subj -g gamma'
parser = argparse.ArgumentParser(usage=usage)
parser.add_argument("roi",    help="name of the roi mask file")
parser.add_argument("-s","--nsubj", type=int,   help="if different subjects have different roi masks, \
	you need to specify how many subjects")
parser.add_argument("-g","--gamma", type=float, help="gamma used in G matrix, default is 2.0")
args = parser.parse_args()
print '--------------experiment arguments--------------'
pprint.pprint(args.__dict__,width=1)

# Find the mask file
template_path = '/jukebox/ramadge/hejiaz/template/'
output_path = '/jukebox/ramadge/hejiaz/data/raw/dmtx/'+args.roi+'/'
if not os.path.exists(output_path):
    os.makedirs(output_path)


if args.nsubj == None:
    nsubj = 1
else:
    nsubj = args.nsubj

D = []
E = []
G = []
C = []

for m in range(nsubj):
    if args.nsubj == None:
        mask_fname = template_path+args.roi+'.nii'
    else:
        mask_fname = template_path+'s'+str(m)+'_'+args.roi+'.nii'

    # Find the 3d location of each voxel
    mask = nib.load(mask_fname)
    maskdata = mask.get_data()
    (i,j,k) = np.where(maskdata>0)
    I=np.reshape(i,(-1,1))
    J=np.reshape(j,(-1,1))
    K=np.reshape(k,(-1,1))
    loc=np.concatenate((I,J,K),axis=1) # (nvoxel by 3, 3d location of each voxel)
    loc=loc.astype(float)

    # Construct the distance matrix
    nvoxel = len(i)
    D_subj = np.zeros((nvoxel, nvoxel))

    for row in range(nvoxel):
        for col in range(row):
            D_subj[row,col] = np.linalg.norm(loc[row,:]-loc[col,:])
            D_subj[col,row] = D_subj[row,col]

    D.append(D_subj)

    # Construct the edge matrix
    (p,q) = np.where(D_subj==1)
    num_edge = len(p)/2
    E_subj = np.zeros((num_edge, nvoxel),dtype=float)
    edge_cnt = 0
    for row in range(nvoxel):
        d_tmp = D_subj[row,row::]
        edge_tmp = np.nonzero(d_tmp==1)
        edge = edge_tmp[0]
        for ind in range(len(edge)):
            E_subj[edge_cnt,row] = 1
            E_subj[edge_cnt,edge[ind]+row] = -1
            edge_cnt += 1         
    
    E.append(E_subj)

    # Construct G matrix
    if args.gamma == None:
        gamma = 2.0
    else:
        gamma = args.gamma
    G_subj = np.exp(-gamma*(D_subj**2))
    alpha = 1.5*max(np.linalg.eigvalsh(G_subj))
    G.append(alpha*np.eye(nvoxel)-G_subj)


    # Construct C matrix
    C.append((E_subj.T).dot(E_subj))
    

# Save results
if args.nsubj == None:
    D_new = copy.copy(D[0])
    E_new = copy.copy(E[0])
    G_new = copy.copy(G[0])
    C_new = copy.copy(C[0])
    np.savez_compressed(output_path+'D_mtx.npz', D=D_new)
    np.savez_compressed(output_path+'E_mtx.npz', E=E_new)
    np.savez_compressed(output_path+'G_mtx.npz', G=G_new)
    np.savez_compressed(output_path+'C_mtx.npz', C=C_new)
else:
    np.savez_compressed(output_path+'D_mtx.npz', D=D)
    np.savez_compressed(output_path+'E_mtx.npz', E=E)
    np.savez_compressed(output_path+'G_mtx.npz', G=G)
    np.savez_compressed(output_path+'C_mtx.npz', C=C)


print 'Done'






