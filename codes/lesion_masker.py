"""
Created on Tue Jul  7 15:32:15 2015

@author: kiran
"""

import nibabel as nib
from scipy.ndimage.filters import gaussian_filter
# import matplotlib.pyplot as plt
import numpy as np
#from skimage.morphology import ball, erosion, dilation
from scipy import ndimage
import os
from mha import *
def LesionMasker(root, prefix):
    
#    gaussian_kernel = 0.5
    volume_threshold = 0.7
    brush_size = 2
    
    # root = '/home/bmi/varghese/Recon_2013_data/testing/'
    # prefix = 'rms_9x9x9_perfect_5000-2000-500_M111'
    
    filenames = []
    priors = []
    mask_f=[]
    folders = []
    
    for subdirs, dirs, files in os.walk(root):
        # if len(filenames) == 1:
        #     break
        for file1 in files:
            if prefix in file1 and file1[-3:] == 'nii' and 'pen' not in file1 and'Masked_RawOutput' not in file1 and 'dynamic' not in file1:
                filenames.append(file1)
                folders.append(subdirs + '/')
            if file1 == 'posteriors_mask.nii':
                priors.append(file1)
            if file1 =='mask.mha':
                mask_f.append(file1)
    
    print 'No. of images: ', len(priors)
    print 'No. of filenames: ',len(filenames)            
    for i in xrange(len(filenames)):
        
        
        print 'Image: ', filenames[i]
        
        img = nib.load(folders[i] + filenames[i])
        prior = nib.load(folders[i] + priors[i])
        
        img = img.get_data()
        o_img = img
        prior = prior.get_data()
        m=new(folders[i]+mask_f[i])
        m=m.data

        img[np.where(m==0)]=0
        
        img[np.where(prior!=0)] = 0
        
        mask = img>0
        
        c,n = ndimage.label(mask)
        
        sizes = ndimage.sum(mask, c, range(n+1))
        mask_size = sizes < (max(sizes) * volume_threshold)
        remove_voxels = mask_size[c]
        c[remove_voxels] = 0
        c[np.where(c!=0)]=1
    
    #
    #    
    #    b = gaussian_filter(c,gaussian_kernel)
    #    b[np.where(b!=0)] = 1
        
        
        ##Erosion##
#        brush = ball(brush_size)
#        c = dilation(c,brush)
        
        o_img[np.where(c==0)] = 0
    
        affine = [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
        # o_img = nib.Nifti1Image(o_img, affine)
        # o_img.set_data_dtype(np.int32)
        # nib.save(o_img,folders[i] +prefix'PnL_output.nii')
        
        img = nib.Nifti1Image(o_img, affine)
        img.set_data_dtype(np.int32)
        nib.save(img,folders[i] +prefix+'Masked_RawOutput.nii')
    
    print 'Masks created'


if __name__ == '__main__':
    root = '/media/bmi/MyPassport/n4_entire/n4_t1_t1c_hist_match_z_score/'
    prefix = 'new_n4_ub_zc_log'
    LesionMasker(root, prefix)
