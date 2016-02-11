# forming transformation matrices for non loo experiment

import numpy as np


def transform(args, workspace, nsubjs):
    W = []
    S = np.zeros((args.nfeature, args.nTR))

    if args.align_algo in ['spatial_srm', 'srm_noneprob']:
        W = workspace['W']
        S = workspace['S']
    elif args.align_algo in ['srm_noneprob_kernel']:
        A = workspace['A']
        #TODO: from A to W
        S = workspace['S']
    elif args.align_algo in ['srm']:
        bW = workspace['bW']
        S  = workspace['ES']
        voxel_str = 0
        for m in range(nsubjs):
            W.append(bW[voxel_str:(voxel_str+nvoxel[m]), :])
            voxel_str = voxel_str + nvoxel[m]
    else:
        exit('alignment algo not recognized')

    return (W, S)
