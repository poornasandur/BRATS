
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:12:53 2015

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

#Initialize user variables
#patch_size = 125
def perfect_balance_3D(patch_size_x=5,patch_size_y=5,patch_size_z=5,prefix='SdA',in_root='',out_root='',recon_flag=False):
    
    patch_pixels = patch_size_x*patch_size_y*patch_size_z
    
    pixel_offset_x = int(patch_size_x*0.5)
    pixel_offset_y = int(patch_size_y*0.5)
    pixel_offset_z = 1
    
    padding = patch_size_x
    #threshold = patch_pixels*0.3
    #patches = np.zeros(patch_pixels*4)
    if recon_flag is True:
        recon_num = 5
    else:
        recon_num = 4
    patches = np.zeros(patch_size_x*patch_size_y*patch_size_z*recon_num)
    ground_truth = np.zeros(1)
    
    #paths to images
    path = in_root
    
    Flair = []
    T1 = []
    T2 = []
    T_1c = []
    Truth = []
    Recon=[]
    Folder = []

    print 'Path: ',path
    
    for subdir, dirs, files in os.walk(path):
#        if len(Flair) is 1:
#            break
        for file1 in files:     

            if file1[-3:]=='mha' and ('Flair' in file1):
                Flair.append(file1)
                Folder.append(subdir+'/')
            elif file1[-3:]=='mha' and ('T1' in file1 and 'T1c' not in file1):
                T1.append(file1)
            elif file1[-3:]=='mha' and ('T2' in file1):
                T2.append(file1)
            elif file1[-3:]=='mha' and ('T1c' in file1 or 'T_1c' in file1):
                T_1c.append(file1)
            elif file1[-3:]=='mha' and 'OT' in file1:
                Truth.append(file1)            
            elif file1[-3:]=='mha' and 'Recon' in file1:
                Recon.append(file1)
                
    number_of_images = len(T1)
    print 'Number of Patients : ', number_of_images
    

#    
#    
    for image_iterator in range(number_of_images):
        print 'Image number : ',image_iterator+1
        print 'Folder : ', Folder[image_iterator]
        
        Flair_image = new(Folder[image_iterator]+Flair[image_iterator])
        T1_image = new(Folder[image_iterator]+T1[image_iterator])
        T2_image = new(Folder[image_iterator]+T2[image_iterator])
        T_1c_image = new(Folder[image_iterator]+T_1c[image_iterator])


#        print 'image created'
        print Folder[image_iterator] + Truth[image_iterator]
        try:
            Truth_image = new( Folder[image_iterator] + Truth[image_iterator] )
        except:
            Truth_image = new2( Folder[image_iterator] + Truth[image_iterator] )
#        print 'image created'
        
        
        Flair_image = Flair_image.data
        T1_image = T1_image.data
        T2_image = T2_image.data
        T_1c_image = T_1c_image.data
        Truth_image = Truth_image.data
        
        x_span,y_span,z_span = np.where(Truth_image!=0)  ### have to change this and include condition i!=0 and i!=5... x,y,z=np.where(Truth_image!=0) and np.where(Truth_image!=5)
        x_start = np.min(x_span) - padding
        x_stop = np.max(x_span) + padding+1
        y_start = np.min(y_span) - padding
        y_stop = np.max(y_span) + padding+1
        z_start = np.min(z_span) - padding
        z_stop = np.max(z_span) +padding+1


        Flair_patch = image.extract_patches(Flair_image[x_start:x_stop, y_start:y_stop, z_start:z_stop], [patch_size_x,patch_size_y,patch_size_z])
        Flair_patch = Flair_patch.reshape(Flair_patch.shape[0]*Flair_patch.shape[1]*Flair_patch.shape[2], patch_size_x*patch_size_y*patch_size_z)
        
        T1_patch = image.extract_patches(T1_image[x_start:x_stop, y_start:y_stop, z_start:z_stop],[patch_size_x,patch_size_y,patch_size_z])
        T1_patch = T1_patch.reshape(T1_patch.shape[0]*T1_patch.shape[1]*T1_patch.shape[2], patch_size_x*patch_size_y*patch_size_z)
        
        T2_patch = image.extract_patches(T2_image[x_start:x_stop, y_start:y_stop, z_start:z_stop],[patch_size_x,patch_size_y,patch_size_z])
        T2_patch = T2_patch.reshape(T2_patch.shape[0]*T2_patch.shape[1]*T2_patch.shape[2], patch_size_x*patch_size_y*patch_size_z)
        
        T_1c_patch = image.extract_patches(T_1c_image[x_start:x_stop, y_start:y_stop, z_start:z_stop],[patch_size_x,patch_size_y,patch_size_z])
        T_1c_patch = T_1c_patch.reshape(T_1c_patch.shape[0]*T_1c_patch.shape[1]*T_1c_patch.shape[2], patch_size_x*patch_size_y*patch_size_z)
        
        T_patch = image.extract_patches(Truth_image[x_start:x_stop, y_start:y_stop, z_start:z_stop],[patch_size_x,patch_size_y,patch_size_z])
        T_patch = T_patch.reshape(T_patch.shape[0]*T_patch.shape[1]*T_patch.shape[2],patch_size_x, patch_size_y, patch_size_z)
        T_patch = T_patch[:,(patch_size_x-1)/2,(patch_size_y-1)/2,(patch_size_z-1)/2]
        T_patch = T_patch.reshape(len(T_patch),1)
        num_of_class = []
        for i in xrange(0,5):   # have to change this too... 0,6
            num_of_class.append(np.sum((T_patch==i).astype(int)))
        minim = min(x for x in num_of_class if x!=0)   # have to change this too. include effect of class 5. Dont think this will create any issue and hence o change required..
        if minim > 1000:
            minim = 1000
        slice_patch = np.zeros(patch_size_x*patch_size_y*patch_size_z*recon_num)
#        print ' CHECK ', slice_patch.shape
        Truth_patch = np.zeros(1)
#        print minim
#        print 
        for i in xrange(5):  # change required. 6
            if num_of_class[i]==0:  # change required...
                continue
#            print 'LOOK ' , i
            index_x, index_y = np.where(T_patch==i)
#            print index_x
            index1 = np.arange(len(index_x))
            shuffle(index1)
#            print index1
#            print index_x[index1[0:minim]].shape
            slice_patch1 = np.concatenate([Flair_patch[index_x[index1[0:minim]],:],T1_patch[index_x[index1[0:minim]],:], T2_patch[index_x[index1[0:minim]],:], T_1c_patch[index_x[index1[0:minim]],:]], axis=1)
#            print 'Slice_patch 1', slice_patch1.shape
#            print 'Slice patch ', slice_patch.shape
            slice_patch = np.vstack([slice_patch, slice_patch1])
#            print Truth_patch.shape
#            print T_patch.shape
            Truth_patch = np.vstack([Truth_patch, T_patch[index_x[index1[0:minim]]]])
        print 'No. of 0 : ', np.sum((Truth_patch==0).astype(int))    
        print 'No. of 1 : ', np.sum((Truth_patch==1).astype(int))
        print 'No. of 2 : ', np.sum((Truth_patch==2).astype(int))
        print 'No. of 3 : ', np.sum((Truth_patch==3).astype(int))
        print 'No. of 4 : ', np.sum((Truth_patch==4).astype(int)) 

        print 'No. of class 5',np.sum((Truth_patch==5).astype(int))   # added  now ...
            
        patches = np.vstack([patches,slice_patch])
        
        ground_truth = np.vstack([ground_truth, Truth_patch])
        print ground_truth.shape
        print patches.shape

    ground_truth = ground_truth.reshape(len(ground_truth))
    print 'No. of 0 : ', np.sum((ground_truth==0).astype(int))    
    print 'No. of 1 : ', np.sum((ground_truth==1).astype(int))
    print 'No. of 2 : ', np.sum((ground_truth==2).astype(int))
    print 'No. of 3 : ', np.sum((ground_truth==3).astype(int))
    print 'No. of 4 : ', np.sum((ground_truth==4).astype(int))
    print 'No. of 5 : ', np.sum((ground_truth==5).astype(int))  # added now...
    if recon_flag==False:
        patches = patches[:,0:patch_size_x*patch_size_y*patch_size_z*4]
    
    #np.save('Training_patches.npy',patches)
    #np.save('Training_labels.npy',ground_truth)
    #print ground_truth.shape
    #print patches.shape
    if 'training' in out_root and recon_flag==True:
        print'... Saving the balanced training patches'
        np.save(out_root+'b_trainpatch_3D_'+prefix+'_.npy',patches)
        np.save(out_root+'b_trainlabel_3D_'+prefix+'_.npy',ground_truth)
    elif recon_flag==True:
        print '... Saving the balance validation patches'
        np.save(out_root+'b_validpatch_3D_'+prefix+'_.npy',patches)
        np.save(out_root+'b_validlabel_3D_'+prefix+'_.npy',ground_truth)
        
    if 'training' in out_root and recon_flag==False:
        print'... Saving the balanced training patches'
        np.save(out_root+'b_trainpatch_3D_'+prefix+'_.npy',patches)
        np.save(out_root+'b_trainlabel_3D_'+prefix+'_.npy',ground_truth)
    elif recon_flag==False:
        print '... Saving the balanced testing patches'
        np.save(out_root+'b_validpatch_3D_'+prefix+'_.npy',patches)
        np.save(out_root+'b_validlabel_3D_'+prefix+'_.npy',ground_truth)
        