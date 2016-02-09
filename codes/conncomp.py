import scipy.ndimage.morphology as morphologicalOperations
from scipy.ndimage.filters import gaussian_filter
from mha import *
from mha2 import *
import numpy as np
import os
import nibabel as nib
from scipy import ndimage

path = '/media/varkey/BRATS_2015_TRAING_UPLOAD/HGG_Norm/'

maskedImages = []
folders = []
rawPrediction = []
Flair = []
T1 = []
T1c = []
T2 = []
posteriors = []

print '==> Loading data..'

i = 0

for subdir, dirs, files in os.walk(path):
	# if '294' in subdir:
	for file1 in files:
		if 'combined__' in file1 and '.mha' in file1:
			# try:
			# 	image = new(subdir + '/' + file1)
			# except:
			# 	image = new2(subdir + '/' + file1)
			maskedImages.append(subdir + '/' + file1)
		if 'combined_HGG_LGG' in file1:
			# print '==> Loading: ', i+1
			# i = i+1
			# image = nib.load(subdir + '/' + file1)
			rawPrediction.append(subdir + '/' + file1)
			folders.append(subdir+'/')
		if 'Flair' in file1:
			# try:
			# 	image = new(subdir + '/' + file1)
			# except:
			# 	image = new2(subdir + '/' + file1)
			Flair.append(subdir + '/' + file1)
		if 'T1.' in file1:
			# try:
			# 	image = new(subdir + '/' + file1)
			# except:
			# 	image = new2(subdir + '/' + file1)
			T1.append(subdir + '/' + file1)
		if 'T1c' in file1:
			# try:
			# 	image = new(subdir + '/' + file1)
			# except:
			# 	image = new2(subdir + '/' + file1)
			T1c.append(subdir + '/' + file1)
		if 'T2' in file1:
			# try:
			# 	image = new(subdir + '/' + file1)
			# except:
			# 	image = new2(subdir + '/' + file1)
			T2.append(subdir + '/' + file1)

		if file1 == 'posteriors_mask.nii':
			posteriors.append(subdir + '/' + file1)


def reject_outliers(data, m=1):

	return data

for i in xrange(len(rawPrediction)):

	print 'Iteration: ', i+1
	print 'Folder: ', folders[i]

	

	rawData = nib.load(rawPrediction[i])
	rawData = rawData.get_data()

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

	try:
		flair_data = new(Flair[i])
	except:
		flair_data = new2(Flair[i])

	try:
		T1_data = new(T1[i])
	except:
		T1_data = new2(T1[i])

	try:
		T2_data = new(T2[i])
	except:
		T2_data = new2(T2[i])

	try:
		T1c_data = new(T1c[i])
	except:
		T1c_data = new2(T1c[i])

	flair_data = flair_data.data
	T1_data = T1_data.data
	T2_data = T2_data.data
	T1c_data = T1c_data.data

	brainMean_flair = np.mean(flair_data[flair_data > 0])
	brainMean_T1 = np.mean(T1_data[T1_data > 0])
	brainMean_T2 = np.mean(T2_data[T2_data > 0])
	brainMean_T1c = np.mean(T1c_data[T1c_data > 0])

	brainStd_flair = np.std(flair_data[flair_data > 0])
	brainStd_T1 = np.std(T1_data[T1_data > 0])
	brainStd_T2 = np.std(T2_data[T2_data > 0])
	brainStd_T1c = np.std(T1c_data[T1c_data > 0])
	###################################################

	gen = nib.load(folders[i] + 'PM.nii')
	gen = gen.get_data()

	mask = gen > 0
	    
	c,n = ndimage.label(mask)
	connectedComponents = np.copy(c)
	sizes = ndimage.sum(mask, c, range(n+1))

	flair_mean = []
	flair_std = []

	T1_mean = []
	T1_std = []

	T2_mean = []
	T2_std = []

	T1c_mean = []
	T1c_std = []

	distance = []
	std = []
	entropy = []
	flair_n = []
	T1_n = []
	T2_n = []
	T1c_n = []
	eps = 1e-07

	print '==> Finding distance..'
	print '==> Connected Components: ', n

	for j in xrange(n+1):

		flair_n.append(reject_outliers(flair_data[np.where(c==j)]))
		T1_n.append(reject_outliers(T1_data[np.where(c==j)]))
		T2_n.append(reject_outliers(T2_data[np.where(c==j)]))
		T1c_n.append(reject_outliers(T1c_data[np.where(c==j)]))

		flair_mean.append(np.mean(flair_n[j]))
		T1_mean.append(np.mean(T1_n[j]))
		T2_mean.append(np.mean(T2_n[j]))
		T1c_mean.append(np.mean(T1c_n[j]))

		flair_std.append(np.std(flair_n[j]))
		T1_std.append(np.std(T1_n[j]))
		T2_std.append(np.std(T2_n[j]))
		T1c_std.append(np.std(T1c_n[j]))

		if flair_std[j] == 0:
			flair_std[j] = 1
		if T1_std[j] == 0:
			T1_std[j] = 1
		if T2_std[j] == 0:
			T2_std[j] = 1
		if T1c_std[j] == 0:
			T1c_std[j] = 1



		# flair_hist = np.histogram(flair_data[np.where(c==j)])
		# T1_hist = np.histogram(T1_data[np.where(c==j)])
		# T1c_hist = np.histogram(T1c_data[np.where(c==j)])
		# T2_hist = np.histogram(T2_data[np.where(c==j)])

		# p1 = flair_hist[0] / float(np.sum(flair_hist[0]))
		# p2 = T1_hist[0] / float(np.sum(T1_hist[0]))
		# p3 = T2_hist[0] / float(np.sum(T2_hist[0]))
		# p4 = T1c_hist[0] / float(np.sum(T1c_hist[0]))

		# entropy.append( np.sum(-p1*np.log2(p1 + eps)) + np.sum(-p2*np.log2(p2+eps)) + np.sum(-p3*np.log2(p3+eps)) + np.sum(-p4*np.log2(p4+eps)) )

		distance.append((flair_mean[j] - brainMean_flair)*(flair_mean[j] - brainMean_flair)
						+(T1_mean[j] - brainMean_T1)*(T1_mean[j] - brainMean_T1)
						+(T2_mean[j] - brainMean_T2)*(T2_mean[j] - brainMean_T2)
						+(T1c_mean[j] - brainMean_T1c)*(T1c_mean[j] - brainMean_T1c))

	# print '==> Distance: ', distance

	print '==> Distance calculated!'

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

		rawData = nib.load(rawPrediction[i])
		rawData = rawData.get_data()



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
	nib.save(img,folders[i] +'Filled_test.nii')

	img = nib.Nifti1Image(connectedComponents, affine)
	img.set_data_dtype(np.int32)
	nib.save(img,folders[i] +'connectedComponents.nii')



