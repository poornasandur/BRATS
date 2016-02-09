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

for subdir, dirs, files in os.walk(path):
	# if '294' in subdir:
	for file1 in files:
		if 'Filled_test' in file1:
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


for i in xrange(len(maskedImages)):

	print 'Iteration: ', i+1
	print 'Folder: ', folders[i]

	# if i == 1:
	# 	break

	maskedImage = nib.load(maskedImages[i])
	maskedImage = maskedImage.get_data()
	maskedImage[maskedImage > 0] = 1

	gaussian_kernel = 0.7
	maskedImage = gaussian_filter(maskedImage, gaussian_kernel)
	maskedImage[maskedImage > 0] = 1

	rawOutput = nib.load(rawPrediction[i])
	rawOutput = rawOutput.get_data()
	rawOutput[maskedImage == 0] = 0

	affine = [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
	img = nib.Nifti1Image(rawOutput, affine)
	img.set_data_dtype(np.int32)
	nib.save(img,folders[i] +'Gaussian_smoothened.nii')



