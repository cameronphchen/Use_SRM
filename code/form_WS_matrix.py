# forming W and S matrices for different algorithms

import numpy as np
import scipy.io
from scipy import stats

def transform(args, options, workspace, nsubjs):
    W = []

    if args.align_algo in ['spatial_srm', 'srm_noneprob']:
        W = workspace['W']
        S = workspace['S']
        
    elif args.align_algo in ['srm_noneprob_kernel']:
        A = workspace['A']
        S = workspace['S']
        if args.format is None:  # If the data is in .npz
            data = np.load(options['input_path']+args.datapath)
            data = data['data']
        # If the input and output data format is MAT
        elif args.format == 'MAT':
            data = scipy.io.loadmat(options['input_path']+args.datapath)
            data = data['data'][0]
        for m in xrange(nsubjs):
            data_zscore = stats.zscore(data[m].T, axis=0, ddof=1).T
            W.append(data_zscore.dot(A[:,:,m]))
        
    elif args.align_algo in ['srm']:
        W = workspace['bW']
        S  = workspace['ES']

    else:
        exit('alignment algo not recognized')

    return (W, S)
