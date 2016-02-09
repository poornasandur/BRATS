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
	if 'x116' in subdir:
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
	mahalanobis_distance = []
	eps = 1e-07

	print '==> Finding distance..'

	for j in xrange(n+1):
		flair_mean.append(np.mean(flair_data[np.where(c==j)]))
		T1_mean.append(np.mean(T1_data[np.where(c==j)]))
		T2_mean.append(np.mean(T2_data[np.where(c==j)]))
		T1c_mean.append(np.mean(T1c_data[np.where(c==j)]))

		flair_std.append(np.std(flair_data[np.where(c==j)]))
		T1_std.append(np.std(T1_data[np.where(c==j)]))
		T2_std.append(np.std(T2_data[np.where(c==j)]))
		T1c_std.append(np.std(T1c_data[np.where(c==j)]))

		if flair_std[j] == 0:
			flair_std[j] = 1
		if T1_std[j] == 0:
			T1_std[j] = 1
		if T2_std[j] == 0:
			T2_std[j] = 1
		if T1c_std[j] == 0:
			T1c_std[j] = 1


		cov = [[0 for x in range(4)] for x in range(4)] 


		blob_mean = [flair_mean[j], T1_mean[j], T2_mean[j], T1c_mean[j]]
		blob = [flair_data[np.where(c==j)], T1_data[np.where(c==j)], T2_data[np.where(c==j)], T1c_data[np.where(c==j)]]

		for u in xrange(4):
			for v in xrange(4):
				cov[u][v] = np.mean((blob[u] - blob_mean[u])*(blob[v] - blob_mean[v]))

		try:
			inv_cov = np.linalg.inv(cov)
		except:
			inv_cov = 0

		blob_mean = np.asarray(blob_mean)
		brain_mean = np.asarray([brainMean_flair, brainMean_T1, brainMean_T2, brainMean_T1c])

		mahalanobis_distance.append(np.dot(np.dot(blob_mean - brain_mean, inv_cov), blob_mean - brain_mean))


		# flair_hist = np.histogram(flair_data[np.where(c==j)])
		# T1_hist = np.histogram(T1_data[np.where(c==j)])
		# T1c_hist = np.histogram(T1c_data[np.where(c==j)])
		# T2_hist = np.histogram(T2_data[np.where(c==j)])

		# p1 = flair_hist[0] / float(np.sum(flair_hist[0]))
		# p2 = T1_hist[0] / float(np.sum(T1_hist[0]))
		# p3 = T2_hist[0] / float(np.sum(T2_hist[0]))
		# p4 = T1c_hist[0] / float(np.sum(T1c_hist[0]))

		# entropy.append( np.sum(-p1*np.log2(p1 + eps)) + np.sum(-p2*np.log2(p2+eps)) + np.sum(-p3*np.log2(p3+eps)) + np.sum(-p4*np.log2(p4+eps)) )

		distance.append( (flair_mean[j] - brainMean_flair)*(flair_mean[j] - brainMean_flair)
						+(T1_mean[j] - brainMean_T1)*(T1_mean[j] - brainMean_T1)
						+(T2_mean[j] - brainMean_T2)*(T2_mean[j] - brainMean_T2)
						+(T1c_mean[j] - brainMean_T1c)*(T1c_mean[j] - brainMean_T1c))

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

	# affine = [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
	# img = nib.Nifti1Image(rawData, affine)
	# img.set_data_dtype(np.int32)
	# nib.save(img,folders[i] +'Filled.nii')

	# img = nib.Nifti1Image(connectedComponents, affine)
	# img.set_data_dtype(np.int32)
	# nib.save(img,folders[i] +'connectedComponents.nii')



