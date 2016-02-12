#!/usr/bin/env python

# Hyperalignment with anatomical information, update W in stiefel manifold

# movie_data is a three dimensional matrix of size voxel x TR x nsubjs
# movie_data[:,:,m] is the data for subject m, which will be X_m^T in the standard 
# mathematic notation
# Needs args.roi and args.mu

# By Hejia Zhang @Princeton University


import numpy as np, scipy, random, sys, math, copy, os
from scipy import stats
import scipy.linalg

def align(movie_data, options, args):
    print 'Spatial_SRM',
    sys.stdout.flush()
    
    #parameters
    lamda = args.mu
    mxitr  = 1000
    wtol = 1e-6
    gtol = 1e-6
    ftol = 1e-12
    gamma = 0.85
    eta = 0.2
    rho = 1e-4
    nt = 5
    tau0 = 1e-3
    

    X = movie_data
    nsubjs = len(X)
    for m in range(nsubjs):
        assert X[0].shape[1] == X[m].shape[1], 'numbers of TRs are different among subjects'
    nTR = X[0].shape[1]
    align_algo = args.align_algo
    nfeature = args.nfeature
    eye2k = np.eye(2*nfeature)

    for m in xrange(nsubjs):
        X[m] = stats.zscore(X[m].T, axis=0, ddof=1).T
     
    current_file = options['working_path']+align_algo+'_current.npz'   

    if not os.path.exists(current_file):
        #initialization
        S = np.zeros((nfeature,nTR))
        W = []
        for m in xrange(nsubjs):
            nvoxel = X[m].shape[0]
            W.append(np.zeros(shape=(nvoxel, nfeature)))
                 
        if args.randseed != None:
            print 'randinit',
            np.random.seed(args.randseed)
            for m in xrange(nsubjs):
                nvoxel = X[m].shape[0]           
                A = np.random.random((nvoxel,nfeature))
                Q, R_qr = np.linalg.qr(A)
                W[m] = Q
                S = S + W[m].T.dot(X[m])
        else:
            for m in xrange(nsubjs):
                nvoxel = X[m].shape[0]           
                W[m] = np.eye(nvoxel,nfeature)
                S = S + W[m].T.dot(X[m])

        S = S/float(nsubjs)
  
        niter = 0
        np.savez_compressed(options['working_path']+align_algo+'_'+str(niter)+'.npz',\
                        W = W, S = S, niter=niter)
    else:
        workspace = np.load(current_file)
        niter = workspace['niter']
        workspace = np.load(options['working_path']+align_algo+'_'+str(niter)+'.npz')
        W = workspace['W']
        S = workspace['S']
        niter = workspace['niter']
  
    # Extract the regularization matrix
    dist_file = options['mask_path']+args.roi+'/C_mtx.npz'
    dist = np.load(dist_file)
    E = dist['C']
    
    print str(niter+1)+'th',
    # Update W
    for m in range(nsubjs):
        print '.',  
        nvoxel = X[m].shape[0]
        if (nfeature>=nvoxel/2):
            invH = True
        else:
            invH = False      
        sys.stdout.flush()
        crit = np.array([]).reshape(0,3)
        out = {}
        F = np.linalg.norm(X[m]-W[m].dot(S),ord='fro')**2+lamda*np.trace((W[m].T).dot(E[m]).dot(W[m]))
        G = 2.0*(W[m].dot(S.dot(S.T))+lamda*(E[m].T).dot(W[m])-X[m].dot(S.T))                                                                        
        out['nfe'] = 1
        GW = (G.T).dot(W[m])
    
        if invH:
            GWT = G.dot(W[m].T)
            H = 0.5*(GWT - GWT.T)
            RW = H.dot(W[m])
        else:
            U = np.concatenate((G, W[m]),axis=1) 
            V = np.concatenate((W[m], -G),axis=1)      
            VU = (V.T).dot(U)
            VW = (V.T).dot(W[m])
    
        dtW = G - W[m].dot(GW)
        nrmG  = np.linalg.norm(dtW, ord='fro')
        Q = 1.0
        Cval = F 
        tau = tau0
    
        #main iterations
        for itr in range(mxitr):
            #values from last iteration
            WP = copy.copy(W[m])   
            FP = copy.copy(F)
            GP = copy.copy(G)   
            dtWP = copy.copy(dtW)
    
            #scale step size
            nls = 1 
            deriv = rho*nrmG**2
            while 1:
                #calculate G, F
                if invH:
                    W_tmp= np.linalg.solve(np.eye(nvoxel) + tau*H, WP - tau*RW)
                    W[m] = W_tmp
                else:
                    aa_tmp = np.linalg.solve(eye2k + (0.5*tau)*VU, VW)
                    aa = aa_tmp
                    W[m] = WP - U.dot(tau*aa)
    
                F = np.linalg.norm(X[m]-W[m].dot(S),ord='fro')**2+lamda*np.trace((W[m].T).dot(E[m]).dot(W[m]))
                out['nfe'] = out['nfe'] + 1       
                if (F <= Cval - tau*deriv) or (nls >= 5):
                    break
                tau = eta*tau
                nls += 1
    
            #main calculation
            G = 2.0*(W[m].dot(S.dot(S.T))+lamda*(E[m].T).dot(W[m])-X[m].dot(S.T)) 
            GW = (G.T).dot(W[m])
            if invH:
                GWT = G.dot(W[m].T) 
                H = 0.5*(GWT - GWT.T)
                RW = H.dot(W[m])
            else:
                U = np.concatenate((G, W[m]),axis=1) 
                V = np.concatenate((W[m], -G),axis=1)      
                VU = (V.T).dot(U)
                VW = (V.T).dot(W[m])
    
            dtW = G - W[m].dot(GW)
            nrmG  = np.linalg.norm(dtW, ord='fro')
    
            #calculate next step size using Barzilai-Borwein
            Sk = W[m] - WP
            WDiff = np.linalg.norm(Sk,ord='fro')/np.sqrt(nvoxel)
            tau = tau0
            FDiff = abs(FP-F)/(abs(FP)+1.0)
            Y = dtW - dtWP     
            SY = abs(np.trace((Sk.T).dot(Y)))
            if itr%2==0:
                tau = np.trace((Sk.T).dot(Sk))/SY
            else:
                tau = SY/np.trace((Y.T).dot(Y))
            tau = max(min(tau, 1e20), 1e-20)
    
            #early stop criterion
            crit = np.concatenate((crit,np.array([nrmG, WDiff, FDiff]).reshape(1,3)),axis=0)
            if itr<nt:
                mcrit = np.mean(crit[0:itr+1, :],axis=0)
            else:
                mcrit = np.mean(crit[itr-nt:itr+1, :],axis=0)
            if ((WDiff < wtol) and (FDiff < ftol)) or (nrmG < gtol) or ((mcrit[1] < 10*wtol) and (mcrit[2] < 10*ftol)):  
                if itr <= 2:
                    ftol = 0.1*ftol
                    wtol = 0.1*wtol
                    gtol = 0.1*gtol
                else:
                    out['msg'] = 'converge'
                    break
    
            #update parameters
            Qp = Q
            Q = gamma*Qp + 1
            Cval = (gamma*Qp*Cval + F)/Q
    
    
        #After iterations
        if itr == mxitr-1:
            out['msg'] = 'exceed max iteration'
        #Make sure orthogonality constraint holds
        out['feasi'] = np.linalg.norm((W[m].T).dot(W[m])-np.eye(nfeature),ord='fro')
        if out['feasi'] > 1e-13:
            W_tmp = copy.copy(W[m])
            #MGramSchmidt
            for dj in range(nfeature):
                for di in range(dj-1):
                    W_tmp[:,dj] = W_tmp[:,dj] - ((W_tmp[:,dj].T.dot(W_tmp[:,di]))/(W_tmp[:,di].T.dot(W_tmp[:,di])))*W_tmp[:,di]
                W_tmp[:,dj] = W_tmp[:,dj]/np.linalg.norm(W_tmp[:,dj],ord=2)
            W[m] = copy.copy(W_tmp)
            #re-calculate objective value
            F = np.linalg.norm(X[m]-W[m].dot(S),ord='fro')**2+lamda*np.trace((W[m].T).dot(E[m]).dot(W[m]))
            out['nfe'] += 1
            out['feasi'] = np.linalg.norm((W[m].T).dot(W[m])-np.eye(nfeature),ord='fro')
        #output information
        out['nrmG'] = nrmG
        out['fval'] = F
        out['itr'] = itr
    

    # Update S
    S_tmp = np.zeros((nfeature,nTR))
    for m in range(nsubjs):
        S_tmp += (W[m].T).dot(X[m])
    S = S_tmp/nsubjs
    
    # calculate objective value
    F_tmp = 0.0
    for m in range(nsubjs):
        F_tmp += np.linalg.norm(X[m]-W[m].dot(S),ord='fro')**2+lamda*np.trace((W[m].T).dot(E[m]).dot(W[m]))
    print 'obj_val = '+str(F_tmp)
    
    new_niter = niter + 1
    np.savez_compressed(current_file, niter = new_niter)
    np.savez_compressed(options['working_path']+align_algo+'_'+str(new_niter)+'.npz',\
                        W = W, S = S, niter=new_niter)
    np.savez_compressed(options['working_path']+align_algo+'_'+str(new_niter)+'_obj.npz',
                        obj=F_tmp)
    os.remove(options['working_path']+align_algo+'_'+str(new_niter-1)+'.npz')
    
    # print objectives
    sys.stdout.flush()
    return new_niter
