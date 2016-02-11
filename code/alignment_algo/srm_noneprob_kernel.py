#!/usr/bin/env python

# Nonprobabilistic Kernel Shared Response Model 

# movie_data is a three dimensional matrix of size voxel x TR x nsubjs
# movie_data[:,:,m] is the data for subject m, which will be X_m^T in the standard 
# mathematic notation

import numpy as np, scipy, random, sys, math, os
from scipy import stats
from sklearn.metrics.pairwise import  pairwise_kernels

def align(movie_data, options, args, lrh):
    print 'SRM nonprob kernel {} {}, k={}'.format(args.kernel, args.sigma, args.nfeature),
    sys.stdout.flush()

    X = movie_data
    nsubjs = len(X)
    for m in range(nsubjs):
        assert X[0].shape[1] == X[m].shape[1], 'numbers of TRs are different among subjects'
    nTR = X[0].shape[1]
    align_algo = args.align_algo
    nfeature = args.nfeature

    current_file = options['working_path']+align_algo+'_'+lrh+'_current.npz' 

    K = np.zeros ((nTR,nTR,nsubjs))
    for m in xrange(nsubjs):
        X[m] = stats.zscore(X[m].T, axis=0, ddof=1).T

    kwds = {}
    if args.kernel in ['rbf','sigmoid','poly']:
        kwds['gamma'] = args.sigma
    if args.kernel in ['poly']:
        kwds['degree'] = args.degree


    for m in range(nsubjs):
        # Identity Kernel
        K[:,:,m] = pairwise_kernels(X[m].T ,X[m].T, metric=args.kernel, **kwds)
        
    if not os.path.exists(current_file):
        A = np.zeros((nTR,nfeature,nsubjs))
        S = np.zeros((nfeature,nTR))
        
        #initialization
        np.random.seed(args.randseed)
        for m in range(nsubjs):
            nvoxel = X[m].shape[0]
            Q_qr, R_qr = np.linalg.qr(np.random.random((nvoxel,nfeature)))
            A[:,:,m] = np.linalg.pinv(X[m]).dot(Q_qr)        
            S = S + A[:,:,m].T.dot(K[:,:,m])
        S = S/float(nsubjs)
        niter = 0
        np.savez_compressed(options['working_path']+align_algo+'_'+lrh+'_'+str(niter)+'.npz',\
                          A = A, S = S, niter=niter)
    else:
        workspace = np.load(current_file)
        niter = workspace['niter']
        workspace = np.load(options['working_path']+align_algo+'_'+lrh+'_'+str(niter)+'.npz')
        A = workspace['A'] 
        S = workspace['S']
        niter = workspace['niter']

    print str(niter+1)+'th',
    for m in range(nsubjs):
        print '.',
        sys.stdout.flush()

        S_tmp = S*nsubjs - A[:,:,m].T.dot(K[:,:,m]) 
        S_tmp = S_tmp/float(nsubjs-1)

        Qm, sm, Qmt = np.linalg.svd(S_tmp.dot(K[:,:,m]).dot(S_tmp.T) ,full_matrices=False)
        A[:,:,m] = S_tmp.T.dot(Qm).dot(np.diag(sm**(-0.5))).dot(Qmt)

        if np.linalg.norm(A[:,:,m].T.dot(K[:,:,m]).dot(A[:,:,m]) - np.eye(nfeature),'fro') < 1e-5:
            print 'I',
        S_tmp = S_tmp*(nsubjs-1) + A[:,:,m].T.dot(K[:,:,m]) 
        S = S_tmp/float(nsubjs)    

    def obj_func(bX, A, S):
        obj_val_tmp = 0
        for m in range(nsubjs):
            obj_val_tmp += np.trace(K[:,:,m] - 2*K[:,:,m].dot(A[:,:,m]).dot(S) + S.T.dot(S))
        print obj_val_tmp
        return obj_val_tmp

    
    obj_func(movie_data_zscore, A, S)

    new_niter = niter + 1
    np.savez_compressed(current_file, niter = new_niter)
    np.savez_compressed(options['working_path']+align_algo+'_'+lrh+'_'+str(new_niter)+'.npz',\
                        A = A, S = S, niter=new_niter)
    return new_niter