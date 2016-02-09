import os
import getopt
import sys

new_prefix = 'new_n4_ub_zc_log'
# new_prefix='new_n4'+'_mse_WT2_log'
prefix_list=[new_prefix + 'Masked_RawOutput.nii' + 'WT', new_prefix + 'Masked_RawOutput.nii'+ 'TC', new_prefix + 'Masked_RawOutput.nii'+ 'AT']
truth_list = ['binary_truth_whole_tumor', 'binary_truth_tumor_core', 'binary_truth_active_tumor']
# prefix_list=[new_prefix + 'Masked_RawOutput.nii']
# truth_list = ['binary_truth_whole_tumor']
ANTSPATH = '/home/bmi/antsbin/bin/'
# LABELPATH = '/media/bmi/varkey/new_n4/Recon_2013_data/N4_zscore_testing_t1_t1c_hist_match/'
# TRUTHPATH = '/media/bmi/varkey/new_n4/Recon_2013_data/N4_zscore_testing_t1_t1c_hist_match/'
LABELPATH = '/media/bmi/MyPassport/n4_entire/n4_t1_t1c_hist_match_z_score/'
TRUTHPATH = '/media/bmi/MyPassport/n4_entire/n4_t1_t1c_hist_match_z_score/'

# print prefix_list
for prefix, truth in zip(prefix_list, truth_list):

	files = os.listdir(LABELPATH)
	LABELPATH, patients, files = os.walk(LABELPATH).next()
	#print patients
	for p in patients:
		# print p
	   	imgs=os.listdir(LABELPATH+'/'+p)
	   	for i in imgs:
	   		if prefix in i and i[-3:]=='nii':
	   			classified_path=LABELPATH+'/'+p+'/'+i
	   		if truth in i :
	   			truth_path=TRUTHPATH+'/'+p+'/'+i
	            
	#    print classified_path
	#    print truth_path
	#    print(ANTSPATH+'LabelOverlapMeasures 3 ' +truth_path+' '+classified_path+' 1')
		os.system(ANTSPATH+'LabelOverlapMeasures 3 ' +truth_path+' '+classified_path+' ')


