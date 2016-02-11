#!/usr/bin/env python

# Constrainted EM algorithm for Shared Response Model

# A Reduced-Dimension fMRI Shared Response Model
# Po-Hsuan Chen, Janice Chen, Yaara Yeshurun-Dishon, Uri Hasson, James Haxby, Peter Ramadge 
# Advances in Neural Information Processing Systems (NIPS), 2015. (to appear) 

# movie_data is a three dimensional matrix of size voxel x TR x nsubjs
# movie_data[:,:,m] is the data for subject m, which will be X_m^T in the standard 
# mathematic notation

# E-step:
# E_s   : nvoxel x nTR
# E_sst : nvoxel x nvoxel x nTR
# M-step:
# W_m   : nvoxel x nvoxel x nsubjs
# sigma_m2 : nsubjs 
# Sig_s : nvoxel x nvoxel 

import numpy as np, scipy, random, sys, math, os
from scipy import stats

def align(movie_data, options, args, lrh):
    print 'SRM',
    sys.stdout.flush()
  
    nsubjs = len(movie_data)
    for m in range(nsubjs):
        assert movie_data[0].shape[1] == movie_data[m].shape[1], 'numbers of TRs are different among subjects'
    nTR = movie_data[0].shape[1]
    nfeature = args.nfeature
    align_algo = args.align_algo
  
    current_file = options['working_path']+align_algo+'_'+lrh+'_current.npz'
    # zscore the data
    nvoxel = np.zeros((nsubjs,),dtype=int)
    for m in xrange(nsubjs):
        nvoxel[m] = movie_data[m].shape[0] 
    bX = np.zeros((sum(nvoxel),nTR))

    voxel_str = 0
    for m in range(nsubjs):
        bX[voxel_str:(voxel_str+nvoxel[m]),:] = stats.zscore(movie_data[m].T ,axis=0, ddof=1).T
        voxel_str = voxel_str + nvoxel[m]

    del movie_data

    # initialization when first time run the algorithm
    if not os.path.exists(current_file):
        bSig_s = np.identity(nfeature)
        bW     = np.zeros((bX_len,nfeature))
        sigma2 = np.zeros(nsubjs)
        ES     = np.zeros((nfeature,nTR))
        bmu = []
        for m in xrange(nsubjs):
            bmu.append(np.zeros((nvoxel[m],)))
  
        #initialization
        voxel_str = 0
        if args.randseed != None:
            print 'randinit',
            np.random.seed(args.randseed)
            for m in xrange(nsubjs):
                A = np.random.random((nvoxel[m],nfeature))
                Q, R_qr = np.linalg.qr(A)
                bW[voxel_str:(voxel_str+nvoxel[m]),:] = Q 
                sigma2[m] = 1
                bmu[m] = np.mean(bX[voxel_str:(voxel_str+nvoxel[m]),:],1)
                voxel_str = voxel_str + nvoxel[m]
        else:
            for m in xrange(nsubjs):
                Q = np.identity(nvoxel,nfeature)
                bW[voxel_str:(voxel_str+nvoxel[m]),:] = Q
                sigma2[m] = 1
                bmu[m] = np.mean(bX[voxel_str:(voxel_str+nvoxel[m]),:],1)
                voxel_str = voxel_str + nvoxel[m]
  
        niter = 0
        np.savez_compressed(options['working_path']+align_algo+'_'+lrh+'_'+str(niter)+'.npz',\
                            bSig_s = bSig_s, bW = bW, bmu=bmu, sigma2=sigma2, ES=ES, nvoxel=nvoxel, niter=niter)
  
        # more iterations starts from previous results
    else:
        workspace = np.load(current_file)
        niter = workspace['niter']
        workspace = np.load(options['working_path']+align_algo+'_'+lrh+'_'+str(niter)+'.npz')
        bSig_s = workspace['bSig_s'] 
        bW     = workspace['bW']
        bmu    = workspace['bmu']
        sigma2 = workspace['sigma2']
        ES     = workspace['ES']
        niter  = workspace['niter']



    # remove mean
    bX = bX - bX.mean(axis=1)[:,np.newaxis]
  
    print str(niter+1)+'th',
   
    bSig_x = bW.dot(bSig_s).dot(bW.T)
  
    voxel_str = 0  
    for m in range(nsubjs):
        bSig_x[voxel_str:(voxel_str+nvoxel[m]),voxel_str:(voxel_str+nvoxel[m])] += sigma2[m]*np.identity(nvoxel[m]])
        voxel_str = voxel_str + nvoxel[m]

    inv_bSig_x = scipy.linalg.inv(bSig_x)
    ES = bSig_s.T.dot(bW.T).dot(inv_bSig_x).dot(bX)
    bSig_s = bSig_s - bSig_s.T.dot(bW.T).dot(inv_bSig_x).dot(bW).dot(bSig_s) + ES.dot(ES.T)/float(nTR)
  
    voxel_str = 0  
    for m in range(nsubjs):
        print ('.'),
        sys.stdout.flush()
        Am = bX[voxel_str:(voxel_str+nvoxel[m]),:].dot(ES.T)
        pert = np.zeros((Am.shape))
        np.fill_diagonal(pert,1)
        Um, sm, Vm = np.linalg.svd(Am+0.001*pert,full_matrices=0)
        bW[voxel_str:(voxel_str+nvoxel[m]),:] = Um.dot(Vm)
        sigma2[m] =    np.trace(bX[voxel_str:(voxel_str+nvoxel[m]),:].T.dot(bX[voxel_str:(voxel_str+nvoxel[m]),:]))\
                    -2*np.trace(bX[voxel_str:(voxel_str+nvoxel[m]),:].T.dot(bW[voxel_str:(voxel_str+nvoxel[m]),:]).dot(ES))\
                  +nTR*np.trace(bSig_s)
        sigma2[m] = sigma2[m]/float(nTR*nvoxel[m])
        voxel_str = voxel_str + nvoxel[m]

    new_niter = niter + 1
    np.savez_compressed(current_file, niter = new_niter)  
    np.savez_compressed(options['working_path']+align_algo+'_'+lrh+'_'+str(new_niter)+'.npz',\
                        bSig_s = bSig_s, bW = bW, bmu=bmu, sigma2=sigma2, ES=ES, nvoxel=nvoxel, niter=new_niter)
    os.remove(options['working_path']+align_algo+'_'+lrh+'_'+str(new_niter-1)+'.npz')

    # calculate log likelihood
    sign , logdet = np.linalg.slogdet(bSig_x)
    if sign == -1:
        print str(new_niter)+'th iteration, log sign negative'
  
    loglike = - 0.5*nTR*logdet - 0.5*np.trace(bX.T.dot(inv_bSig_x).dot(bX)) #-0.5*nTR*nvoxel*nsubjs*math.log(2*math.pi)
  
    np.savez_compressed(options['working_path']+align_algo+'_'+'loglikelihood_'+lrh+'_'+str(new_niter)+'.npz',\
                        loglike=loglike)
    
    # print str(-0.5*nTR*logdet)+','+str(-0.5*np.trace(bX.T.dot(inv_bSig_x).dot(bX)))
    print str(loglike) 
  
    return new_niter
