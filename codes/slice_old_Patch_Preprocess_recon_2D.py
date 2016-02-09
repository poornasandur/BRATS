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

def slice_U_Patch_Preprocess_recon_2D(patch_size_x=5,patch_size_y=5,prefix='SdA',in_root='',out_root='',slice_num=1, data_augment=False):
    
    #Initialize user variables
    patch_size = patch_size_x
    patch_pixels = patch_size*patch_size
    pixel_offset = int(patch_size*0.5)
    padding = patch_size
    #threshold = patch_pixels*0.3
    recon_num = 4
    patches = np.zeros(patch_pixels*recon_num)
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


        x_span,y_span,z_span = np.where(Truth_image!=0)
        
        start_slice = min(z_span)
        stop_slice = max(z_span)
        print start_slice, stop_slice
        image_patch = np.zeros(patch_size*patch_size*recon_num)
        image_label = np.zeros(1)
        for i in range(start_slice, stop_slice+1):
            #print 'Slice : '+str(i)    
            Flair_slice = np.transpose(Flair_image[:,:,i])
            T1_slice = np.transpose(T1_image[:,:,i])
            
            T2_slice = np.transpose(T2_image[:,:,i])
            T_1c_slice = np.transpose(T_1c_image[:,:,i])    
            Truth_slice = np.transpose(Truth_image[:,:,i])
            
            x_dim,y_dim = np.size(Flair_slice,axis=0), np.size(Flair_slice, axis=1)
            
            x_span,y_span = np.where(Truth_slice!=0)
            if len(x_span)==0 or len(y_span)==0:
                continue
            x_start = np.min(x_span) - padding
            x_stop = np.max(x_span) + padding+1
            y_start = np.min(y_span) - padding
            y_stop = np.max(y_span) + padding+1
            #print 'X start, X stop ', x_start,x_stop
            #print 'Y start, Y stop ', y_start, y_stop
            Flair_patch = image.extract_patches(Flair_slice[x_start:x_stop, y_start:y_stop], patch_size, extraction_step = pixel_offset)
            T1_patch = image.extract_patches(T1_slice[x_start:x_stop, y_start:y_stop], patch_size, extraction_step = pixel_offset)
            T2_patch = image.extract_patches(T2_slice[x_start:x_stop, y_start:y_stop], patch_size, extraction_step = pixel_offset)
            T_1c_patch = image.extract_patches(T_1c_slice[x_start:x_stop, y_start:y_stop], patch_size, extraction_step = pixel_offset)     
            Truth_patch = image.extract_patches(Truth_slice[x_start:x_stop, y_start:y_stop], patch_size, extraction_step = pixel_offset)
            
            if data_augment == True:
                Fp = Flair_patch.reshape(Flair_patch.shape[0]*Flair_patch.shape[1], patch_size_x, patch_size_y)
                T1p = T1_patch.reshape(T1_patch.shape[0]*T1_patch.shape[1], patch_size,patch_size)
                T2p = T2_patch.reshape(T2_patch.shape[0]*T2_patch.shape[1], patch_size,patch_size)  
                T1cp = T_1c_patch.reshape(T_1c_patch.shape[0]*T_1c_patch.shape[1], patch_size,patch_size)
                Truthp = Truth_patch.reshape(Truth_patch.shape[0]*Truth_patch.shape[1], patch_size, patch_size)
                Truthp = Truthp[:,(patch_size-1)/2,(patch_size-1)/2]
                Truthp = np.array(Truthp)
                Truthp = Truthp.reshape(len(Truthp))

                indexx = np.where(Truthp!=0)
                # print indexx[0]
                indexx=indexx[0]

                Fp = Fp[indexx,:,:]
                T1p = T1p[indexx,:,:]
                T2p = T2p[indexx,:,:]
                T1cp = T1cp[indexx,:,:]
                Truthp = Truthp[indexx]

                for angle in range(3):
                    Fp = np.asarray([np.rot90(Fp[x,:,:]) for x in range(Fp.shape[0])])
                    T1p = np.asarray([np.rot90(T1p[x,:,:]) for x in range(T1p.shape[0])])
                    T2p = np.asarray([np.rot90(T2p[x,:,:]) for x in range(T2p.shape[0])])
                    T1cp = np.asarray([np.rot90(T1cp[x,:,:]) for x in range(T1cp.shape[0])])

                    Fp_n = np.asarray(Fp)
                    T1p_n = np.asarray(T1p)
                    T2p_n = np.asarray(T2p)
                    T1cp_n = np.asarray(T1cp)

                    Fp_n = Fp_n.reshape(Fp_n.shape[0], patch_size_x*patch_size_y)
                    T1p_n = T1p_n.reshape(T1p_n.shape[0], patch_size_x*patch_size_y)
                    T2p_n = T2p_n.reshape(T2p_n.shape[0], patch_size_x*patch_size_y)
                    T1cp_n = T1cp_n.reshape(T1cp_n.shape[0], patch_size_x*patch_size_y)
                    Truthp_n = Truthp.reshape(len(Truthp),1)
                    sp_n = np.concatenate([Fp_n, T1p_n, T2p_n, T1cp_n], axis=1)

                    patches = np.vstack([patches, sp_n])
                    ground_truth = np.vstack([ground_truth, Truthp_n])
                    # print 'Angle : '+str(90*(angle+1))
                    # print 'Augmented Data: '+str(Fp_n.shape[0])+' patches'

            # print Flair_patch.shape
            Flair_patch = Flair_patch.reshape(Flair_patch.shape[0]*Flair_patch.shape[1], patch_size*patch_size)
            T1_patch = T1_patch.reshape(T1_patch.shape[0]*T1_patch.shape[1], patch_size*patch_size)
            T2_patch = T2_patch.reshape(T2_patch.shape[0]*T2_patch.shape[1], patch_size*patch_size)  
            T_1c_patch = T_1c_patch.reshape(T_1c_patch.shape[0]*T_1c_patch.shape[1], patch_size*patch_size)      
            Truth_patch = Truth_patch.reshape(Truth_patch.shape[0]*Truth_patch.shape[1], patch_size, patch_size)
            
            #print '2. truth dimension :', Truth_patch.shape
            slice_patch = np.concatenate([Flair_patch, T1_patch, T2_patch, T_1c_patch], axis=1)
            Truth_patch = Truth_patch[:,(patch_size-1)/2,(patch_size-1)/2]
            Truth_patch = np.array(Truth_patch)
            Truth_patch = Truth_patch.reshape(len(Truth_patch),1)
            
            patches = np.vstack([patches,slice_patch])
            ground_truth = np.vstack([ground_truth, Truth_patch])
    print 'Number of non-zeros in ground truth : ', np.sum((ground_truth!=0).astype(int))
    print 'Number of zeros in ground truth : ', np.sum((ground_truth==0).astype(int))
    # ground_truth[np.where(ground_truth==3)]=1
    # ground_truth[np.where(ground_truth==4)]=1
    print
    print 'No. of 1 : ', np.sum((ground_truth==1).astype(int))
    print 'No. of 2 : ', np.sum((ground_truth==2).astype(int))
    print 'No. of 3 : ', np.sum((ground_truth==3).astype(int))
    print 'No. of 4 : ', np.sum((ground_truth==4).astype(int))
    
    ground_truth = ground_truth.reshape(len(ground_truth))
    print 'Shape of Un-balanced patches numpy array : ',patches.shape
    print 'Shape of Un-balanced ground truth : ',ground_truth.shape
    
    patches = np.float32(patches)
    ground_truth = np.float32(ground_truth)
    if 'training' in out_root:
        print'... Saving the 2D training patches'
        np.save(out_root+'u_trainpatch_2D_'+prefix+'_.npy',patches)
        np.save(out_root+'u_trainlabel_2D_'+prefix+'_.npy',ground_truth)
        
    elif 'validation' in out_root:
        print '... Saving the 2D validation patches'
        np.save(out_root+'u_validpatch_2D_'+prefix+'_.npy',patches)
        np.save(out_root+'u_validlabel_2D_'+prefix+'_.npy',ground_truth)
