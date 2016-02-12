#!/usr/bin/env python

# This is the code to run SRM only
# Please refer to --help for arguments setting
# There is also an option to input matlab files directly. The input and output files
# will be in the same format.
#
# align_algo must be spatial_srm, srm_noneprob_kernel, srm_nonprob, or srm
#
# by Cameron Po-Hsuan Chen and Hejia Zhang @ Princeton


import numpy as np, scipy, random, sys, math, os, copy
import scipy.io
from scipy import stats
import argparse
import importlib
import pprint
import form_WS_matrix

## argument parsing
usage = '%(prog)s datapath align_algo niter nfeature [-r RANDSEED] [-k kernel] [-s sigma]\
[-d degree] [-o roi] [-m mu] [-f format] [--strfresh]'
parser = argparse.ArgumentParser(usage=usage)

parser.add_argument("dataset",    help="name of the dataset")
parser.add_argument("datapath",    help="the data file with its path")
parser.add_argument("align_algo", help="name of the alignment algorithm")
parser.add_argument("niter"     , type = int,  
                    help="number of iterations to the algorithm")
parser.add_argument("nfeature", type=int, 
                    help="number of features")
parser.add_argument("-r", "--randseed", type=int, metavar='',
                    help="random seed for initialization")

# kernel SRM specific
parser.add_argument("-k", "--kernel", metavar='',
                    help="type of kernel to use for sklearn.metrics.pairwise.pairwise_kernels : rbf, sigmoid, polynomial, poly, linear, cosine ")
parser.add_argument("-s", "--sigma" , type = float,  
                    help="sigma value")
parser.add_argument("-d", "--degree" , type = int,  
                    help="degree for poly kernel")
# spatial SRM specific
parser.add_argument("-o", "--roi" ,  
                    help="ROI name (mask file name)")
parser.add_argument("-m", "--mu" , type = float,  
                    help="regularization parameter")

parser.add_argument("-f","--format", metavar='',help="format of input and output data, e.g. MAT. If not set, default is python array")
parser.add_argument("--strfresh", action="store_true" ,
                    help="start alignment fresh, not picking up from where was left")

args = parser.parse_args()
print '--------------experiment arguments--------------'
pprint.pprint(args.__dict__,width=1)

# sanity check
if args.align_algo == 'srm_noneprob_kernel' and args.kernel == None:
    sys.exit('Please specify kernel type for kernel SRM')
if args.align_algo == 'spatial_srm':
    if args.roi == None:
        sys.exit('Please specify roi for spatial srm')
    if args.mu == None:
        sys.exit('Please specify mu for spatial srm')

# rondo options
algo_folder = args.align_algo + '/'
if args.align_algo == 'srm_noneprob_kernel' :
    algo_folder = algo_folder+args.kernel+'/'+('sigma'+str(args.sigma)+'/' if args.sigma != None else "" )+\
    ('degree'+str(args.degree)+'/' if args.degree != None else "" )
elif args.align_algo == 'spatial_srm':
    algo_folder = algo_folder+args.roi+'/'+'mu'+str(args.mu)+'/'

opt_folder  = str(args.nfeature) + 'feat/' + ("rand"+str(args.randseed)+'/' if args.randseed != None else "identity/" )

options = {'input_path'  : '/jukebox/ramadge/hejiaz/data/input/',
           'working_path': '/fastscratch/hejiaz/tmp/data/working/'+args.dataset+'/'+algo_folder+opt_folder,
           'output_path' : '/fastscratch/hejiaz/tmp/data/output/'+args.dataset+'/'+algo_folder+opt_folder,
           'mask_path': '/jukebox/ramadge/hejiaz/data/raw/dmtx/'}
print '----------------SRM paths----------------'
pprint.pprint(options, width=1)
print '------------------------------------------------'

# load data for alignment
print 'start loading data'
if args.format is None:  # If the data is in .npz
    data = np.load(options['input_path']+args.datapath)
    data = data['data']
# If the input and output data format is MAT
elif args.format == 'MAT':
    data = scipy.io.loadmat(options['input_path']+args.datapath)
    data = data['data'][0]

# zscore and sanity check
nsubjs = len(data)
nvoxel = np.zeros((nsubjs,))
nTR = data[0].shape[1]
align_data = []
for m in range(nsubjs):
    assert data[0].shape[1] == data[m].shape[1], 'numbers of TRs are different among subjects'
    nvoxel[m] = data[m].shape[0]
    assert nvoxel[m] >= args.nfeature, 'number of features is larger than number of voxels'
    align_data.append(stats.zscore(data[m].T, axis=0, ddof=1).T)

# creating working folder
if not os.path.exists(options['working_path']):
    os.makedirs(options['working_path'])
if not os.path.exists(options['output_path']):
    os.makedirs(options['output_path'])

if args.strfresh:
    if os.path.exists(options['working_path']+args.align_algo+'_current.npz'):
        os.remove(options['working_path']+args.align_algo+'_current.npz')


# run alignment
print 'start alignment'
algo = importlib.import_module('alignment_algo.'+args.align_algo)
if os.path.exists(options['working_path']+args.align_algo+'_current.npz'):
  workspace = np.load(options['working_path']+args.align_algo+'_current.npz')
  new_niter = workspace['niter']
else:
  new_niter = 0

while (new_niter<args.niter):  
    new_niter = algo.align(align_data, options, args)


# form WS matrix
print 'start transform'
workspace = np.load(options['working_path']+args.align_algo+'_'+str(args.niter)+'.npz')
W, S = form_WS_matrix.transform(args, options, workspace, nsubjs)

if args.format == None:
    np.savez_compressed(options['output_path']+args.align_algo+str(args.niter)+'_WS.npz',
                        W=W, S=S)
elif args.format == 'MAT':
    WS_mat = {'W' : W}
    WS_mat['S'] = S
    scipy.io.savemat(options['output_path']+args.align_algo+str(args.niter)+'_WS.mat',WS_mat)

print 'alignment done'