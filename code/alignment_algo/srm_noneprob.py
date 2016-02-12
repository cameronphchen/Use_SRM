#!/usr/bin/env python

# Inference code for non-probabilistic Shared Response Model with varying number of voxels across subjects

# A Reduced-Dimension fMRI Shared Response Model
# Po-Hsuan Chen, Janice Chen, Yaara Yeshurun-Dishon, Uri Hasson, James Haxby, Peter Ramadge
# Advances in Neural Information Processing Systems (NIPS), 2015.

# movie_data is a list of two dimensional arrays
# movie_data[m] is the data for subject m of size nvoxel x nTR
# nvoxels can be different across subjects

# By Cameron PH Chen @ Princeton

import numpy as np
import sys
import os
from scipy import stats


def align(movie_data, options, args):
    print 'SRM None-Prob' + str(args.nfeature),
    sys.stdout.flush()
    X = movie_data
    nsubjs = len(X)

    for m in range(nsubjs):
        assert X[0].shape[1] == X[m].shape[1], 'numbers of TRs are different among subjects'

    nTR = X[0].shape[1]

    align_algo = args.align_algo
    nfeature = args.nfeature

    current_file = options['working_path']+align_algo+'_current.npz'

    for m in xrange(nsubjs):
        X[m] = stats.zscore(X[m].T, axis=0, ddof=1).T

    if not os.path.exists(current_file):
        W = []
        for m in xrange(nsubjs):
            nvoxel = X[m].shape[0]
            W.append(np.zeros(shape=(nvoxel, nfeature)))
        
        S = np.zeros((nfeature, nTR))

        np.random.seed(args.randseed)
        #initialization        
        for m in xrange(nsubjs):
            nvoxel = X[m].shape[0]
            
            # initialize with random orthogonal matrix
            A = np.mat(np.random.random((nvoxel, nfeature)))
            Q, R_qr = np.linalg.qr(A)

            W[m] = Q
            S = S + W[m].T.dot(X[m])
        S = S/float(nsubjs)

        niter = 0
        np.savez_compressed(options['working_path']+align_algo+'_'+str(niter)+'.npz',
                            W=W, S=S, niter=niter)
    else:
        workspace = np.load(current_file)
        niter = workspace['niter']
        workspace = np.load(options['working_path']+align_algo+'_'+str(niter)+'.npz')
        W = workspace['W']
        S = workspace['S']
        niter = workspace['niter']

    print str(niter+1)+'th',
    for m in range(nsubjs):
        print '.',
        sys.stdout.flush()

        Am = X[m].dot(S.T)
        pert = np.zeros((Am.shape))
        np.fill_diagonal(pert, 1)
        Um, sm, Vm = np.linalg.svd(Am+0.001*pert, full_matrices=False)

        W[m] = Um.dot(Vm)  # W = UV^T

    S = np.zeros((nfeature, nTR))
    for m in range(nsubjs):
        S = S + W[m].T.dot(X[m])
    S = S/float(nsubjs)

    def obj_func(X, W, S):
        obj_val_tmp = 0
        for m in range(nsubjs):
            obj_val_tmp += np.linalg.norm(X[m] - W[m].dot(S), 'fro')**2
        print obj_val_tmp
        return obj_val_tmp

    new_niter = niter + 1
    np.savez_compressed(current_file, niter=new_niter)
    np.savez_compressed(options['working_path']+align_algo+'_'+str(new_niter)+'.npz',
                        W=W, S=S, niter=new_niter)
    np.savez_compressed(options['working_path']+align_algo+'_'+str(new_niter)+'_obj.npz',
                        obj=obj_func(X, W, S))
    # clean up results of previous iteration
    os.remove(options['working_path']+align_algo+'_'+str(new_niter-1)+'.npz')
    return new_niter
