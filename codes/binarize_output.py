# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 19:19:06 2015

@author: administrator
"""

import os
import nibabel as nib
import numpy as np
from mha import *
# import medpy
# from medpy.io import load

def binarize(path, prefix):

      i = 0

      for subdir, dirs, files in os.walk(path):
            # if i == 1:
            #       break
            # print 'entering'
            for file1 in files:

                  if prefix in file1 and 'WT' not in file1 and 'TC' not in file1 and 'AT' not in file1 and 'dynamic' not in file1 and 'Best' not in file1:
                  
                        print 'Iteration: ', i+1

                        print subdir

                        # img,h = load(subdir+'/'+file1)
                        img = nib.load(subdir + '/' + file1)
                        img = img.get_data()
                        # img = new(subdir + '/' + file1)
                        # img = img.data

                        wholeTumor = np.copy(img)
                        tumorCore = np.copy(img)
                        activeTumor = np.copy(img)

                        affine = [[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]

                        wholeTumor[np.where(wholeTumor!=0)] = 1

                        img = nib.Nifti1Image(wholeTumor, affine)
                        img.set_data_dtype(np.int32)
                        nib.save(img,subdir+'/' + prefix + 'WT.nii')


                        tumorCore[np.where(tumorCore==2)] = 0
                        tumorCore[np.where(tumorCore!=0)] = 1


                        img = nib.Nifti1Image(tumorCore, affine)
                        img.set_data_dtype(np.int32)
                        nib.save(img,subdir+'/'+ prefix + 'TC.nii')

                        print 'Tumor core saved..'


                        activeTumor[np.where(activeTumor!=np.unique(activeTumor)[-1])] = 0
                        activeTumor[np.where(activeTumor!=0)] = 1

                                    
                        img = nib.Nifti1Image(activeTumor, affine)
                        img.set_data_dtype(np.int32)
                        nib.save(img,subdir+'/'+prefix + 'AT.nii')

                        i = i + 1

if __name__ == '__main__':
      
      test_root = '/media/bmi/MyPassport/n4_entire/n4_t1_t1c_hist_match_z_score/'
    
      prefix = 'new_n4_ub_zc_log'
      # new_prefix = prefix + '_' + str(0) + 'dropout_'
      new_prefix=prefix
      print 'Prefix: ', new_prefix

      binarize(test_root, new_prefix + 'Masked_RawOutput.nii')