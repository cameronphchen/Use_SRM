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
import pprint
from transform_matrix import form_transformation_matrix
from alignment_algo import srm_nonprob_kernel

## argument parsing
usage = '%(prog)s dataset nvoxel nTR align_algo niter nfeature [-r RANDSEED] [--strfresh] \
[-f format] [-m matname]'
parser = argparse.ArgumentParser(usage=usage)

parser.add_argument("dataset",    help="name of the dataset")
parser.add_argument("nvoxel", type = int,
                    help="number of voxels in the dataset")
parser.add_argument("nTR", type = int,
                    help="number of TRs in the dataset")
parser.add_argument("align_algo", help="name of the alignment algorithm")
parser.add_argument("-k", "--kernel", metavar='',
                    help="type of kernel to use for sklearn.metrics.pairwise.pairwise_kernels : rbf, sigmoid, polynomial, poly, linear, cosine ")
parser.add_argument("-s", "--sigma" , type = float,  
                    help="sigma value")
parser.add_argument("-d", "--degree" , type = int,  
                    help="degree for poly kernel")
parser.add_argument("niter"     , type = int,  
                    help="number of iterations to the algorithm")
parser.add_argument("nfeature", type=int, 
                    help="number of features")
parser.add_argument("-r", "--randseed", type=int, metavar='',
                    help="random seed for initialization")
parser.add_argument("--strfresh", action="store_true" ,
                    help="start alignment fresh, not picking up from where was left")
parser.add_argument("-f","--format", metavar='',help="format of input and output data, e.g. MAT. If not set, default is python array")
parser.add_argument("-m","--matname", metavar='',help="Name of the MAT file")

args = parser.parse_args()
print '--------------experiment arguments--------------'
pprint.pprint(args.__dict__,width=1)

# sanity check
assert args.nvoxel >= args.nfeature

data_folder = args.dataset+'/'+str(args.nvoxel)+'vx/'+str(args.nTR)+'TR/'
opt_folder  = str(args.nfeature) + 'feat/' + ("rand"+str(args.randseed)+'/' if args.randseed != None else "identity/" )+ '/'

# rondo options
# Change the input_path to where you stored the MAT file if run the MAT mode, also remember to change the working path and output path
options = {'input_path'  : '/jukebox/ramadge/pohsuan/tmp/',
           'working_path': '/fastscratch/pohsuan/tmp/data/working/'+data_folder+opt_folder,
           'output_path' : '/fastscratch/pohsuan/tmp/data/output/'+data_folder+opt_folder}
print '----------------SRM paths----------------'
pprint.pprint(options, width=1)
print '------------------------------------------------'

# creating working folder
if not os.path.exists(options['working_path']):
    os.makedirs(options['working_path'])
if not os.path.exists(options['output_path']):
    os.makedirs(options['output_path'])

if args.strfresh:
    if os.path.exists(options['working_path']+args.align_algo+'__current.npz'):
        os.remove(options['working_path']+args.align_algo+'__current.npz')

print 'start loading data'
# load data for alignment

if args.format is None:  # If the data is in .npz
    movie_data = np.load(options['input_path']+'movie_data.npz')
    movie_data = movie_data['movie_data']
# If the input and output data format is MAT
elif args.format == 'MAT':
    movie_data = scipy.io.loadmat(options['input_path']+args.matname)
    movie_data = movie_data[args.matname]

# zscore
align_data = np.zeros((movie_data.shape))
  
for m in range(align_data.shape[2]):
    align_data[:, :, m] = stats.zscore(movie_data[:, :, m].T, axis=0, ddof=1).T

(nvoxel, nTR, nsubjs) = align_data.shape

# run alignment
print 'start alignment'
new_niter = srm_nonprob_kernel.align(align_data, options, args, '')

# load transformation matrices
print 'start transform'
workspace = np.load(options['working_path']+args.align_algo+'__'+str(new_niter)+'.npz')
A = workspace['A']
transform = np.zeros((nvoxel,args.nfeature,nsubjs))
for m in range(nsubjs):
    transform[:, :, m] = align_data[:, :, m].dot(A[:, :, m])
S = workspace['S']

# transformed mkdg data with learned transformation matrices
transformed_data = np.zeros((args.nfeature, nTR, nsubjs))
for m in range(nsubjs):
    trfed_tmp = transform[:, :, m].T.dot(align_data[:, :, m])
    transformed_data[:, :, m] = stats.zscore(trfed_tmp.T, axis=0, ddof=1).T

if args.format == None:
    np.savez_compressed(options['output_path']+args.align_algo+'_transformed_data.npz',
                        transformed_data=transformed_data, W=transform, S=S)
elif args.format == 'MAT':
    transformed_data_mat = {'transformed_data' : transformed_data}
    transformed_data_mat['W'] = transform
    transformed_data_mat['S'] = S
    scipy.io.savemat(options['output_path']+args.align_algo+'_transformed_data.mat',transformed_data_mat)

print 'SRM done'