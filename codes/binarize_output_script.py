import os
from binarize_output import *

roots= ['/media/bmi/varkey/new_n4/Recon_2013_data/log_training','/media/bmi/varkey/new_n4/Recon_2013_data/log_validation']

prefixs=['new_n4_axis2dropout_xyz','new_n4_axis3dropout_xyz']

for i in range(2):
	root=roots[i]
	print 'root:',root
	for j in range(2):
		prefix=prefixs[j]
		new_prefix=prefix+'Masked_RawOutput.nii'
		print 'new_prefix:',new_prefix
		binarize(root,new_prefix)