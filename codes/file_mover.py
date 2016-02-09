import os

import shutil

posterior_path = '/media/bmi/varkey/BRATS2014_training/HGG_Atropos/'
folder_path='/media/bmi/varkey/n4_hist_z_score/Recon_2013_data/N4_zscore_testing_t1_t1c_hist_match'

inputfiles,patients,folders=os.walk(folder_path).next()

for p in patients:
	img= os.listdir(posterior_path+p)
	for i in img:
		if 'mha' not in i:
			j=posterior_path+p+'/'+i+'/'+'posteriors_mask.nii'
			destination=folder_path+'/'+p
			shutil.copy(j,destination)


