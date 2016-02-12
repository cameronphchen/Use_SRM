# SRM_only

##Code for running Shared Response Model (SRM): 
It always takes in brain maps X and generates W and S such that X is approximately equal to W*S.

1. SRM: 
Full fledged probabilistic SRM

2. Non-probabilistic SRM:
Non-probabilistic version of SRM. When the number of subjects is big, it has similar performance as SRM. When the number of subjects is small, it can be slightly worse than SRM.

3. Kernel SRM:
Kernel version of SRM. It can be faster than SRM especially when the number of voxels is big. The code only supports linear kernel now, so the arguments sigma and degree should not be used at this moment.

4. Spatial SRM:
Non-probabilistic SRM with anatomical information. It can be used to generate more spatially smooth functional topographies (columns of W). Before running this algorithm, please run compute_regularization_mat.py first to generate the required regularization matrices. 


##What do I need to modify:

1. template_path and out_path in compute_regularization_mat.py

(1) template_path is where you store the roi mask nifti files for each subject. For example, if there are 16 subjects in the dataset, and the roi is PMC, then the argument roi (args.roi) should be 'PMC'. The mask files should be in the directory "template_path", and the files should be named as 's0_PMC.nii' through 's15_PMC.nii'.

(2) out_path is where you want to save the generated regularization matrices. 

2. options in run_algo_only.py.
3. 
(1) input_path is where you store the brain maps X. The data file name with its full path should be options['input_path']+args.datapath. 

(2) working_path is where some of the intermediate results are stored.

(3) output_path is where the generated W and S are stored. 

(4) mask_path is where the regularization matrices are stored. It should be exactly the same as out_path in compute_regularization_mat.py. If you do not need to run spatial SRM, you can put a random path here and it would not do anything.

##Input and Output format


Please refer to script_example.sh for example usage. 
