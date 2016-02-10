#!/usr/bin/env python

# Nonprobabilistic Kernel Shared Response Model 

# movie_data is a three dimensional matrix of size voxel x TR x nsubjs
# movie_data[:,:,m] is the data for subject m, which will be X_m^T in the standard 
# mathematic notation

import numpy as np, scipy, random, sys, math, os
from scipy import stats
sys.path.append('/jukebox/ramadge/pohsuan/scikit-learn/sklearn')
from sklearn.metrics.pairwise import  pairwise_kernels

def align(movie_data, options, args, lrh):
    print 'SRM nonprob kernel {} {}, k={}'.format(args.kernel, args.sigma, args.nfeature),
    sys.stdout.flush()

    nvoxel = movie_data.shape[0]
    nTR    = movie_data.shape[1]
    nsubjs = movie_data.shape[2]
    align_algo = args.align_algo
    nfeature = args.nfeature

    current_file = options['working_path']+align_algo+'_'+lrh+'_current.npz' 

    movie_data_zscore = np.zeros ((nvoxel,nTR,nsubjs))
    K = np.zeros ((nTR,nTR,nsubjs))
    for m in range(nsubjs):
        movie_data_zscore[:,:,m] = stats.zscore(movie_data[:,:,m].T, axis=0, ddof=1).T

    kwds = {}
    if args.kernel in ['rbf','sigmoid','poly']:
        kwds['gamma'] = args.sigma
    if args.kernel in ['poly']:
        kwds['degree'] = args.degree


    for m in range(nsubjs):
        # TODO based on different kernel should be using different way to calculate K
        # Identity Kernel
        #K[:,:,m] = movie_data_zscore[:,:,m].T.dot(movie_data_zscore[:,:,m])
        K[:,:,m] = pairwise_kernels(movie_data_zscore[:,:,m].T ,movie_data_zscore[:,:,m].T, metric=args.kernel, **kwds)
        
    if not os.path.exists(current_file):
        A = np.zeros((nTR,nfeature,nsubjs))
        S = np.zeros((nfeature,nTR))
        
        #initialization
        np.random.seed(args.randseed)
        Q_qr, R_qr = np.linalg.qr(np.mat(np.random.random((nvoxel,nfeature))))
        pert = np.zeros((nTR,nTR)) 
        np.fill_diagonal(pert,1)
        for m in range(nsubjs):
            U, s, Ut = np.linalg.svd(K[:,:,m]+0.001*pert, full_matrices=False)
            
            # AKA = I
            # A[:,:,m] = U[:,:nfeature].dot(np.diag(s[:nfeature]**(-0.5))) #.dot(Ut[:nfeature,:])
            
            # using original nonprobabilistic SRM 
            A[:,:,m] = np.linalg.pinv(movie_data_zscore[:,:,m]).dot(Q_qr)
            
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

    """
    def svd(data, template):
        ## Linear Kernel
        #Am = data.dot(template.T)
        ## Quadratic Kernel
        Am = data.dot(template.T)
        Am = (Am + 1)**2
        ## Gaussian Kernel
        #Am = np.outer(data.dot(data.T).diagonal(),np.ones(template.shape[0])) - \
        #     2*data.dot(template.T) + np.outer(np.ones(data.shape[0]),template.dot(template.T).diagonal())
        #Am = scipy.exp(Am/10000)
        pert = np.zeros((Am.shape))
        np.fill_diagonal(pert,1)
        return np.linalg.svd(Am+0.001*pert,full_matrices=False)"""


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