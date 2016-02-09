# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 19:19:06 2015

@author: administrator
"""

import os
# from medpy.io import load
from mha import *
from mha2 import *

import numpy as np
import nibabel as nib

path = '/media/bmi/MyPassport/n4_entire/n4_t1_t1c_hist_match_z_score/'

for subdir, dirs, files in os.walk(path):
    for file1 in files:
        if 'OT' in file1:
            # m,h = load(subdir+'/'+file1)
            try:
                m = new(subdir+'/'+file1)
            except:
                m = new2(subdir+'/'+file1)

            m = m.data
            print 'saving'
            m_w=np.copy(m)
            m_tc=np.copy(m)
            m_at=np.copy(m)
	    #m = m.data

            m_w[np.where(m_w!=0)] = 1   ### whole tumor

            m_tc[np.where(m_tc==2)] = 0    #### for tumor core
            m_tc[np.where(m_tc!=0)] = 1    ### for tumor core

            m_at[np.where(m_at!=np.unique(m_at)[-1])] = 0
            m_at[np.where(m_at!=0)] = 1

            affine = [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
            # img = nib.Nifti1Image(m_w, affine)
            # img.set_data_dtype(np.int32)
            # nib.save(img,subdir + '/binary_truth_whole_tumor.nii')
            # img = nib.Nifti1Image(m, affine)
            # img.set_data_dtype(np.int32)
            # nib.save(img,subdir + '/binary_truth_tumor_core.nii')
            img_w = nib.Nifti1Image(m_w, affine)
            img_w.set_data_dtype(np.int32)
            nib.save(img_w,subdir + '/binary_truth_whole_tumor.nii')

            img_t = nib.Nifti1Image(m_tc, affine)
            img_t.set_data_dtype(np.int32)
            nib.save(img_t,subdir + '/binary_truth_tumor_core.nii')

            img_a = nib.Nifti1Image(m_at, affine)
            img_a.set_data_dtype(np.int32)
            nib.save(img_a,subdir + '/binary_truth_active_tumor.nii')