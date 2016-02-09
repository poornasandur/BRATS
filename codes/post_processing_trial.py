import scipy.ndimage.morphology as morphologicalOperations
from scipy.ndimage.filters import gaussian_filter
from mha import *
from mha2 import *
import numpy as np
import os
import nibabel as nib
from scipy import ndimage

path = '/home/bmi/varghese/BRATS_FINAL_2015/'

maskedImages = []
folders = []
rawPrediction = []
Flair = []
T1 = []
T1c = []
T2 = []
posteriors = []

print '==> Loading data..'

for subdir, dirs, files in os.walk(path):
	if '294' in subdir:
		for file1 in files:
			if 'combined' in file1 and '.mha' in file1:
				try:
					image = new(subdir + '/' + file1)
				except:
					image = new2(subdir + '/' + file1)
				maskedImages.append(image)
				folders.append(subdir+'/')
			if 'Ensemble' in file1:
				image = nib.load(subdir + '/' + file1)
				rawPrediction.append(image)
			
			if 'Flair' in file1:
				try:
					image = new(subdir + '/' + file1)
				except:
					image = new2(subdir + '/' + file1)
				Flair.append(image)
			if 'T1.' in file1:
				try:
					image = new(subdir + '/' + file1)
				except:
					image = new2(subdir + '/' + file1)
				T1.append(image)
			if 'T1c' in file1:
				try:
					image = new(subdir + '/' + file1)
				except:
					image = new2(subdir + '/' + file1)
				T1c.append(image)
			if 'T2' in file1:
				try:
					image = new(subdir + '/' + file1)
				except:
					image = new2(subdir + '/' + file1)
				T2.append(image)

			if file1 == 'posteriors_mask.nii':
				posteriors.append(subdir + '/' + file1)


for i in xrange(len(maskedImages)):

	print 'Iteration: ', i+1
	print 'Folder: ', folders[i]

	# if i == 1:
	# 	break

	rawData = rawPrediction[i].get_data()

	posterior = nib.load(posteriors[i])
	posterior = posterior.get_data()

	data = np.copy(rawData)

	data[np.where(posterior!=0)] = 0

	data[np.where(data > 0)] = 1

	affine = [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
	img = nib.Nifti1Image(data, affine)
	img.set_data_dtype(np.int32)
	nib.save(img,folders[i] +'PM.nii')

	###################################################
	flair_data = Flair[i].data
	T1_data = T1[i].data
	T2_data = T2[i].data
	T1c_data = T1c[i].data

	brainMean_flair = np.mean(flair_data[flair_data > 0])
	brainMean_T1 = np.mean(T1_data[T1_data > 0])
	brainMean_T2 = np.mean(T2_data[T2_data > 0])
	brainMean_T1c = np.mean(T1c_data[T1c_data > 0])
	################################################### added today
	m_f=1
	m_t1=float(brainMean_flair)/brainMean_T1
	m_t1c=float(brainMean_flair)/brainMean_T1c
	m_t2=float(brainMean_flair)/brainMean_T2
############################ Multiply with constants###############################
	brainMean_flair=m_f*brainMean_flair
	brainMean_T1=m_t1*brainMean_T1
	brainMean_T1c=m_t1c*brainMean_T1c
	brainMean_T2=m_t2*brainMean_T2
	#######################################################

	gen = nib.load(folders[i] + 'PM.nii')
	gen = gen.get_data()

	mask = gen > 0
	    
	c,n = ndimage.label(mask)
	connectedComponents = np.copy(c)
	sizes = ndimage.sum(mask, c, range(n+1))
######################################
	less_than_10=sizes < 10
	new_cc=less_than_10[c]
	c[new_cc]=0
	# c[np.where(c!=0)]=c
#####################################

	flair_mean = []
	flair_std = []

	T1_mean = []
	T1_std = []

	T2_mean = []
	T2_std = []

	T1c_mean = []
	T1c_std = []

	distance = []
	print 'Number of connectedComponents:',n

	print '==> Finding distance..'

	for j in xrange(n+1):
		flair_mean.append(m_f*np.mean(flair_data[np.where(c==j)]))
		T1_mean.append(m_t1*np.mean(T1_data[np.where(c==j)]))
		T2_mean.append(m_t2*np.mean(T2_data[np.where(c==j)]))
		T1c_mean.append(m_t1c*np.mean(T1c_data[np.where(c==j)]))

		distance.append( (flair_mean[j] - brainMean_flair)*(flair_mean[j] - brainMean_flair)
						+(T1_mean[j] - brainMean_T1)*(T1_mean[j] - brainMean_T1)
						+(T2_mean[j] - brainMean_T2)*(T2_mean[j] - brainMean_T2)
						+(T1c_mean[j] - brainMean_T1c)*(T1c_mean[j] - brainMean_T1c))



	print 'max(sizes):',max(sizes)
	# print np.where(sizes=max(sizes))
	# print sizes[1]

	mask_size = sizes < (max(sizes) * 0.1)
	distance_mask = distance < np.mean(distance)

	remove_voxels = mask_size[c]
	remove_distance_voxels = distance_mask[c]
	c[remove_distance_voxels] = 0
	c[remove_voxels] = 0
	c[np.where(c!=0)]=1

	
	


	# data = data.astype(int)

	# for j in xrange(data.shape[2]):
	# 	data[:,:,j] = morphologicalOperations.binary_fill_holes(data[:,:,j], structure = np.ones((2,2))).astype(int)

	# gaussian_kernel = 0.7
	# data = gaussian_filter(data,gaussian_kernel)
	# data[np.where(data!=0)] = 1

	# data = data.astype(float)

	# data[np.where(c==0)] = 0

	# rawData = rawPrediction[i].get_data()
	data[np.where(c==0)] = 0

	if len(np.unique(data)) == 1:
		print '==> Blank Image'
		print '==> Unique labels: ', len(np.unique(rawData))

		rawData = rawPrediction[i].get_data()



		data = np.copy(rawData)

		data[np.where(posterior!=0)] = 0
		data[np.where(data > 0)] = 1
		mask = data > 0
		c,n = ndimage.label(mask)
		connectedComponents = np.copy(c)
		sizes = ndimage.sum(mask, c, range(n+1))
		mask_size = sizes < (max(sizes) * 0.6)
		remove_voxels = mask_size[c]
		c[remove_voxels] = 0
		c[np.where(c!=0)]=1
		data[np.where(c==0)] = 0

	rawData[np.where(data == 0)] = 0

	affine = [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
	img = nib.Nifti1Image(rawData, affine)
	img.set_data_dtype(np.int32)
	nib.save(img,folders[i] +'Filled.nii')

	img = nib.Nifti1Image(connectedComponents, affine)
	img.set_data_dtype(np.int32)
	nib.save(img,folders[i] +'connectedComponents.nii')



