#!/usr/bin/env python

# This is the code to compute regularization matrices based on roi mask used.
#1. D: 
# Euclidean distance matrix 
# (voxel*voxel)

# 2. E: 
# Connectivity matrix (generated from D)
# (edge*voxel)

# 3. C:
# E'*E (generated from E)
# (voxel*voxel)

# By Hejia Zhang @Princeton University

import scipy.io
import os,sys,copy
import numpy as np
import nibabel as nib
import random
import argparse
import pprint


## argument parsing
usage = '%(prog)s roi nsubj'
parser = argparse.ArgumentParser(usage=usage)
parser.add_argument("roi",    help="name of the roi mask file")
parser.add_argument("nsubj",  type=int,  help="number of subjects")
args = parser.parse_args()
print '--------------experiment arguments--------------'
pprint.pprint(args.__dict__,width=1)

# Find the mask file
template_path = '/jukebox/ramadge/hejiaz/template/'
output_path = '/jukebox/ramadge/hejiaz/data/raw/dmtx/'+args.roi+'/'
if not os.path.exists(output_path):
    os.makedirs(output_path)


C = []
for m in range(nsubj):
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
    
    # Construct C matrix
    C.append((E_subj.T).dot(E_subj))
    

# Save results
np.savez_compressed(output_path+'C_mtx.npz', C=C)

print 'Done'






