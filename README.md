# Use_SRM

Written by Cameron Chen, Hejia Zhang (Ramadge Lab @ Princeton)

For any questions, please email poshuan@princeton.edu or hejiaz@princeton.edu

If you use this code or SRM in scientific publication, please cite the following paper: 

**A Reduced-Dimension fMRI Shared Response Model**

Po-Hsuan Chen, Janice Chen, Yaara Yeshurun, Uri Hasson, James V. Haxby, Peter J. Ramadge 
Advances in Neural Information Processing Systems (NIPS), 2015. 
[Paper](http://papers.nips.cc/paper/5855-a-reduced-dimension-fmri-shared-response-model)

Bibtex:
```
@inproceedings{phchen2015srm,
  title={A Reduced-Dimension f{MRI} Shared Response Model},
  author={Chen, Po-Hsuan and Chen, Janice and Yeshurun, Yaara and Hasson, Uri and Haxby, James V. and Ramadge, Peter J. },
  year={2015},
  booktitle={Advances in Neural Information Processing Systems (NIPS) },
}
```

##Code for running Shared Response Model (SRM): 
It always takes in brain maps X and generates W and S such that X is approximately equal to W*S.

1. SRM:  
Full fledged probabilistic SRM

2. Non-probabilistic SRM:  
Non-probabilistic version of SRM. When the number of subjects is big, it has similar performance as SRM. When the number of subjects is small, it can be slightly worse than SRM.

## Which SRM should I use?
1. If you are pushing for predictive performance, go for SRM.
2. If you are giving a quick try of SRM, use Non-probabilistic SRM. The performance is close to SRM with many subjects, and it's also much faster.

##What do I need to modify:
1. options in run_algo_only.py.  
(1) input_path is where you store the brain maps X. The data file name with its full path should be options['input_path']+args.datapath.  
(2) working_path is where some of the intermediate results are stored.  
(3) output_path is where the generated W and S are stored.   

##Input and Output format
1. Input brain maps X:  
For example, you have 3 subjects each with 20 TRs, and the number of voxels are 5, 6, and 7, respectively.  
(1) First please make sure you stored the brain maps in 3 matrices (Matlab) /arrays (Python) : X1 (5x20) , X2 (6x20) , and X3 (7x20) .  
(2) For Matlab:  
data = cell(1,3);  
data{1} = X1;  
data{2} = X2;  
data{3} = X3;  
For Python:  
data = []  
data.append(X1)  
data.append(X2)  
data.append(X3)  
(3) Save 'data' as '.mat' or '.npz' file. Note that the variable name has to be 'data'.  

2. To load output functional topographies W:  

(1) If it is saved as '.mat' file:  
ws = scipy.io.loadmat(options['output_path']+args.align_algo+str(args.niter)+'_WS.mat')  
W = ws['W'][0]

(2) If it is saved as '.npz' file:  
ws = np.load(options['output_path']+args.align_algo+str(args.niter)+'_WS.npz')  
W = ws['W']  
Then W is a length-nsubjs cell array. For example, to extract W for the first subject:  
W1 = W[0]  


Please refer to script_example.sh for example usage. 
