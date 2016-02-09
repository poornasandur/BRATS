# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:28:03 2015
@author: subru
"""

from mha import *
from mha2 import *
import numpy as np
import os
from sklearn.feature_extraction import image
from random import shuffle
import nibabel as nib
#from matplotlib import pyplot as plt

def slice_perfect_balance_2D(patch_size_x=5,patch_size_y=5,prefix='Sda',in_root='',out_root='',slice_num =1):
    
    #Initialize user variables
    patch_size = patch_size_x
    patch_pixels = patch_size*patch_size
    pixel_offset = patch_size
    padding = patch_size
    #threshold = patch_pixels*0.3
    recon_num = 4
    label_num = 5
    patches = np.zeros(patch_pixels*recon_num)
#    ground_truth = np.zeros(1)
    ground_truth = np.zeros(1) 
    #paths to images
    path = in_root
    
    Flair = []
    T1 = []
    T2 = []
    T_1c = []
    Truth = []
    Folder = []
    
    for subdir, dirs, files in os.walk(path):
        # if len(Flair) is 1:
        #     break
        for file1 in files:
            #print file1
            if file1[-3:]=='nii' and ('Flair' in file1):                
                Flair.append(file1)
                Folder.append(subdir+'/')
            elif file1[-3:]=='nii' and ('T1' in file1 and 'T1c' not in file1):
                T1.append(file1)
            elif file1[-3:]=='nii' and ('T2' in file1):
                T2.append(file1)
            elif file1[-3:]=='nii' and ('T1c' in file1 or 'T_1c' in file1):
                T_1c.append(file1)
            elif file1[-3:]=='mha' and 'OT' in file1:
                Truth.append(file1)
                
    number_of_images = len(Flair)
    print 'Number of images : ', number_of_images
    
    
    for image_iterator in range(number_of_images):
        print 'Iteration : ',image_iterator+1
        print 'Folder : ', Folder[image_iterator]
        Flair_image = nib.load(Folder[image_iterator]+Flair[image_iterator])
        T1_image = nib.load(Folder[image_iterator]+T1[image_iterator])
        T2_image = nib.load(Folder[image_iterator]+T2[image_iterator])
        T_1c_image = nib.load(Folder[image_iterator]+T_1c[image_iterator])
        try:
            Truth_image = new(Folder[image_iterator]+Truth[image_iterator])
        except:
            Truth_image = new2(Folder[image_iterator]+Truth[image_iterator])
        Flair_image = Flair_image.get_data()
        T1_image = T1_image.get_data()
        T2_image = T2_image.get_data()
        T_1c_image = T_1c_image.get_data()
        Truth_image = Truth_image.data
        
        if slice_num ==2:
            Flair_image = np.swapaxes(Flair_image,0,1)
            Flair_image = np.swapaxes(Flair_image,1,2)
            T1_image = np.swapaxes(T1_image, 0,1)
            T1_image = np.swapaxes(T1_image, 1, 2)
            T2_image = np.swapaxes(T2_image, 0,1)
            T2_image = np.swapaxes(T2_image, 1, 2)
            T_1c_image = np.swapaxes(T_1c_image, 0,1)
            T_1c_image = np.swapaxes(T_1c_image, 1, 2)
            Truth_image = np.swapaxes(Truth_image,0,1)
            Truth_image = np.swapaxes(Truth_image,1,2)
        elif slice_num == 3:
            Flair_image = np.swapaxes(Flair_image,0,1)
            Flair_image = np.swapaxes(Flair_image,0,2)
            T1_image = np.swapaxes(T1_image, 0,1)
            T1_image = np.swapaxes(T1_image, 0, 2)
            T2_image = np.swapaxes(T2_image, 0,1)
            T2_image = np.swapaxes(T2_image, 0, 2)
            T_1c_image = np.swapaxes(T_1c_image, 0,1)
            T_1c_image = np.swapaxes(T_1c_image, 0, 2)
            Truth_image = np.swapaxes(Truth_image,0,1)
            Truth_image = np.swapaxes(Truth_image,0,2)


        # Truth_image[np.where(Truth_image==3)]=1
        # Truth_image[np.where(Truth_image==4)]=1
        
        x_span,y_span,z_span = np.where(Truth_image!=0)        
        start_slice = min(z_span)
        stop_slice = max(z_span)
        x_start = min(x_span)-padding
        x_stop = max(x_span)+padding
        y_start = min(y_span)-padding
        y_stop = max(y_span)+padding
        
        Flair_patch = image.extract_patches(Flair_image[x_start:x_stop,y_start:y_stop,start_slice:stop_slice],[patch_size_x,patch_size_y,1])
        Flair_patch = Flair_patch.reshape(Flair_patch.shape[0]*Flair_patch.shape[1]*Flair_patch.shape[2], patch_size_x*patch_size_y)
        
        T1_patch = image.extract_patches(T1_image[x_start:x_stop,y_start:y_stop,start_slice:stop_slice],[patch_size_x,patch_size_y,1])
        T1_patch = T1_patch.reshape(T1_patch.shape[0]*T1_patch.shape[1]*T1_patch.shape[2], patch_size_x*patch_size_y)
        
        T2_patch = image.extract_patches(T2_image[x_start:x_stop,y_start:y_stop,start_slice:stop_slice],[patch_size_x,patch_size_y,1])
        T2_patch = T2_patch.reshape(T2_patch.shape[0]*T2_patch.shape[1]*T2_patch.shape[2], patch_size_x*patch_size_y)
        
        T_1c_patch = image.extract_patches(T_1c_image[x_start:x_stop,y_start:y_stop,start_slice:stop_slice],[patch_size_x,patch_size_y,1])
        T_1c_patch = T_1c_patch.reshape(T_1c_patch.shape[0]*T_1c_patch.shape[1]*T_1c_patch.shape[2], patch_size_x*patch_size_y)
        
        T_patch = image.extract_patches(Truth_image[x_start:x_stop,y_start:y_stop,start_slice:stop_slice],[patch_size_x,patch_size_y,1])
        T_patch = T_patch.reshape(T_patch.shape[0]*T_patch.shape[1]*T_patch.shape[2],patch_size_x, patch_size_y, 1)
        T_patch = T_patch[:,(patch_size-1)/2,(patch_size-1)/2]
        
        num_of_class = []
        for i in xrange(0,label_num):
            num_of_class.append(np.sum((T_patch==i).astype(int)))
        minim = min(x for x in num_of_class if x!=0)
        if minim>3000:
            minim = 3000
#        flair_patch = np.zeros(patch_size_x*patch_size_y*recon_num)
#        t1_patch = np.zeros(patch_size_x*patch_size_y*recon_num)
#        t2_patch = np.zeros(patch_size_x*patch_size_y*recon_num)
#        t1c_patch = np.zeros(patch_size_x*patch_size_y*recon_num)
        slice_patch = np.zeros(patch_size_x*patch_size_y*recon_num)
        Truth_patch = np.zeros(1)
#        print minim
#        print 
        for i in xrange(5):
            if num_of_class[i]==0:
                continue
            index_x, index_y = np.where(T_patch==i)
            index1 = np.arange(len(index_x))
            shuffle(index1)
#            print index1
#            print index_x[index1[0:minim]].shape
            #print Flair_patch.shape
            slice_patch1 = np.concatenate([Flair_patch[index_x[index1[0:minim]],:],T1_patch[index_x[index1[0:minim]],:], T2_patch[index_x[index1[0:minim]],:], T_1c_patch[index_x[index1[0:minim]],:]], axis=1)
            slice_patch = np.vstack([slice_patch, slice_patch1])
#            flair_patch = np.vstack([flair_patch, Flair_patch[index_x[index1[0:minim]],:]])
#            t1_patch = np.vstack([t1_patch,T1_patch[index_x[index1[0:minim]],:]])
#            t2_patch = np.vstack([t2_patch, T2_patch[index_x[index1[0:minim]],:]])
#            t1c_patch = np.vstack([t1c_patch, T_1c_patch[index_x[index1[0:minim]],:]])
#            print Truth_patch.shape
#            print T_patch.shape
            Truth_patch = np.vstack([Truth_patch, T_patch[index_x[index1[0:minim]]]])
        print 'No. of 0 : ', np.sum((Truth_patch==0).astype(int))    
        print 'No. of 1 : ', np.sum((Truth_patch==1).astype(int))
        print 'No. of 2 : ', np.sum((Truth_patch==2).astype(int))
        print 'No. of 3 : ', np.sum((Truth_patch==3).astype(int))
        print 'No. of 4 : ', np.sum((Truth_patch==4).astype(int))
        
        Truth_patch = Truth_patch.reshape(len(Truth_patch))
        print 'look here ---->',Truth_patch.shape

#        np.save(out_root+'patches_patient_flair'+str(image_iterator+1)+'.npy',flair_patch)
#        np.save(out_root+'patches_patient_t1'+str(image_iterator+1)+'.npy',t1_patch)
#        np.save(out_root+'patches_patient_t2'+str(image_iterator+1)+'.npy',t2_patch)
#        np.save(out_root+'patches_patient_t1c'+str(image_iterator+1)+'.npy',t1c_patch)
#        np.save(out_root+'patches_patient_'+str(image_iterator+1)+'.npy',slice_patch)
#        np.save(out_root+'labels_patient_'+str(image_iterator+1)+'.npy',Truth_patch)
        patches = np.vstack([patches,slice_patch])
        print 'patches balanced shape',patches.shape
        ground_truth = np.append(ground_truth, Truth_patch)
        print 'ground shape--->',ground_truth.shape
    index1 = np.arange(patches.shape[0])
    shuffle(index1)
    print np.shape(patches)
    patches = patches[index1,:]
    ground_truth = ground_truth[index1]
    patches = np.float32(patches)
    ground_truth = np.float32(ground_truth)
    if 'training' in out_root:
        np.save(out_root+'perfect_balance_trainpatch_'+prefix+'_.npy',patches)
        np.save(out_root+'perfect_balance_traintruth_'+prefix+'_.npy',ground_truth)
    else:        
        np.save(out_root+'perfect_balance_validpatch_'+prefix+'_.npy',patches)
        np.save(out_root+'perfect_balance_validtruth_'+prefix+'_.npy',ground_truth)

#out_root+'b_validpatch_2D_'+prefix+'_.npy',patches
