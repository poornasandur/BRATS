# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:07:35 2015

@author: bmi
"""
from perfect_balance_3D import *
from perfect_balance_2d import *
from B_Patch_Preprocess_recon_3D import *
from Patch_Preprocess_recon_3D import *
from old_Patch_Preprocess_recon_2D import *
from test_SdA import *
from test_network import *
from axis_test_network import *
# from slice_selection import *
#from testitk import *
from slice_perfect_balance_2d import *
from slice_old_Patch_Preprocess_recon_2D import *
import os
import time
from lesion_masker import *
from binarize_output import *
from mlp2 import *
from diceplotter import *
import subprocess
from ET_balance_finetune_3D import *

testingImages = 35

prefix = 'new_n4'

new_prefix = prefix + 'dropout_'

prefix_list = [new_prefix + 'Masked_RawOutput.nii' + 'WT', new_prefix + 'Masked_RawOutput.nii'+ 'TC', new_prefix + 'Masked_RawOutput.nii'+ 'AT']

# prefix_list = [prefix, prefix, prefix]

if __name__ == '__main__':
    
    root = '/media/bmi/varkey/new_n4/Recon_2013_data/'
    patch_size_x = 21
    patch_size_y = 21
    patch_size_z = 1
    recon_flag = False
    batch_size = 100
    
    n_ins = patch_size_x * patch_size_y * patch_size_z * 4
    n_outs = 5
    noise_type = 0
    noise_dict = {1:'Gaussian noise', 0:'Masking Noise'}
    
    ## CHECK IMAGE SHAPES FOR PRINTING WEIGHTS ##
    hidden_layers_sizes = [2500,1000,500]
    # hidden_layers_sizes = [800,400,200]
    ## CHECK IMAGE SHAPES FOR PRINTING WEIGHTS ##
    corruption_levels = [0.2,0.2,0.2,0.2]
    
    test_path = root + 'N4_zscore_testing_t1_t1c_hist_match'    
#    prefix = 'rms_9x9x9_5000-2000-500_M222'
    
    slice_num = 1
    data_augment = False


        ###################### PIXEL OFFSET !!!!!!!!!!!!!!!!!!!!!!!!! @@@
    # print 'Extracting perfect Balanced Patches...'
    # slice_perfect_balance_2D(patch_size_x, patch_size_y, prefix, root+'N4_zscore_training_t1_t1c_hist_match', root+'training_patches/',slice_num)
    slice_perfect_balance_2D(patch_size_x, patch_size_y, prefix, root+'N4_zscore_validation_t1_t1c_hist_match', root+'validation_patches/',slice_num)

    # # # ## #     ########################### PIXEL OFFSET !!!!!!!!!!!!!!!!!!!!!!!!! @@@
    # print 'Extracting UnBalanced training patches...'
    # slice_U_Patch_Preprocess_recon_2D(patch_size_x,patch_size_y, prefix,root+'N4_zscore_training_t1_t1c_hist_match', root+'training_patches/',slice_num, data_augment)
    # print 'Training patches extracted!'                                      

    # print 'Extracting Unbalanced validation patches...'
    # slice_U_Patch_Preprocess_recon_2D(patch_size_x,patch_size_y,prefix, root+'N4_zscore_validation_t1_t1c_hist_match', root+'validation_patches/',slice_num,data_augment)

    # path = '/media/bmi/varkey/new_n4/results/'
    # for subdir, dirs, files in os.walk(path):
    #   test_num = len(dirs)+1
    #   break


    # os.mkdir(path+'test_'+str(test_num)+'_'+prefix)
    # test_root = path+'test_'+str(test_num)+'_'+prefix+'/'
                                      
    test_root = '/media/bmi/varkey/new_n4/results/test_1_new_n4/'
    print 'Calling test_SdA...'
    
    finetune_lr = 0.01
    pretraining_epochs = 351                     # WAS 300 CHANGED TO  500...........................
    pretrain_lr = 0.001
    training_epochs = 801
    layer_sizes = [1764,2500,1000,500,5]
    dropout_rates = [[0.01,0.0, 0.0,0.0], [0.0,0.0,0.4,0.4], [0.2,0.2,0.4,0.4]]
   
    # f = open(test_root+prefix+'_params_info.txt', 'w')
    # f.write( "Current date & time " + time.strftime("%c"))
    # f.write('\nPrefix : '+prefix)
    # f.write('\nAxis: '+str(slice_num))
    # if data_augment:
    # 	f.write('\n Data Augmentation : True')
    # else:
    # 	f.write('\n Data Augmentation : False')
    # f.write('\n2D Patches. Patch_size : '+str(patch_size_x)+', '+str(patch_size_y))
    # f.write('\nBatch Size : '+str(batch_size))
    # f.write('\nBatch Size : '+str(batch_size))
    # f.write('\nHidden Layer Sizes : ['+', '.join(map(str,hidden_layers_sizes))+' ]')
    # f.write('\nNoise Type : '+noise_dict[noise_type])
    # f.write('\nCorruption Levels : ['+', '.join(map(str,corruption_levels))+' ]')
    # f.write('\nNo. of pre-training epochs : '+str(pretraining_epochs))
    # f.write('\nNo. of Fine-tuning epochs : '+str(training_epochs))
    # f.write('\nPretraining Learning rate : '+str(pretrain_lr))
    # f.write('\nFine-tuning learning rate : '+str(finetune_lr))
    # f.write('\ndropouts:'+str(dropout_rates[0]))
    # f.write('\nDescription: n4 bias, histogram matching (128,10) , zscore, perfect balance, Unbalanced on 2D, Using nestrov\'s momentum')
    # f.close()
    
    # test_SdA(finetune_lr, pretraining_epochs,
    #          pretrain_lr, training_epochs,              
    #             root+'training_patches/perfect_balance_trainpatch_'+prefix+'_.npy',
    #             root+'training_patches/perfect_balance_traintruth_'+prefix+'_.npy',
    #             root+'validation_patches/perfect_balance_validpatch_'+prefix+'_.npy',
    #             root+'validation_patches/perfect_balance_validtruth_'+prefix+'_.npy',
    #             root+'training_patches/u_trainpatch_2D_'+prefix+'_.npy',
    #             root+'training_patches/u_trainlabel_2D_'+prefix+'_.npy',
    #             root+'validation_patches/u_validpatch_2D_'+prefix+'_.npy',
    #             root+'validation_patches/u_validlabel_2D_'+prefix+'_.npy',
    #             batch_size, n_ins, n_outs, hidden_layers_sizes, test_root + prefix, corruption_levels, False, True)
               
    # print 'Network Trained and Saved!'                
               

    # runMLP2(finetune_lr,
    #       training_epochs,
    #       batch_size,
    #       layer_sizes,
    #       dropout_rates[0],
    #       test_root + prefix + 'pre_training.pkl',
    #       root+'training_patches/u_trainpatch_2D_'+prefix+'_.npy',
    #       root+'training_patches/u_trainlabel_2D_'+prefix+'_.npy',
    #       root+'validation_patches/u_validpatch_2D_'+prefix+'_.npy',
    #       root+'validation_patches/u_validlabel_2D_'+prefix+'_.npy',
    #       test_root + prefix)


#    runMLP2(finetune_lr,
#            training_epochs,
#            batch_size,
#            layer_sizes,
#            dropout_rates[0],
#            test_root + prefix + 'pre_training.pkl',
#            root+'BRATS_training_patches/u_trainpatch_3D_'+prefix+'_.npy',
#            root+'BRATS_training_patches/u_trainlabel_3D_'+prefix+'_.npy',
#            root+'BRATS_validation_patches/u_validpatch_3D_'+prefix+'_.npy',
#            root+'BRATS_validation_patches/u_validlabel_3D_'+prefix+'_.npy',
#            test_root + prefix)

    # for i in xrange(len(dropout_rates)):
    #     if i == 1:
    #         break
       #  runMLP2(finetune_lr,
       #      150,
       #      batch_size,
       #      layer_sizes,
       #      dropout_rates[i],
       #      test_root + prefix + 'pre_training.pkl',
       #      root+'BRATS_training_patches/u_trainpatch_3D_'+prefix+'_.npy',
       #      root+'BRATS_training_patches/u_trainlabel_3D_'+prefix+'_.npy',
       #      root+'BRATS_validation_patches/u_validpatch_3D_'+prefix+'_.npy',
       #      root+'BRATS_validation_patches/u_validlabel_3D_'+prefix+'_.npy',
       #      test_root + prefix + '_' + str(i))
# ########################################################################################
    axis_test_network(test_root,new_prefix,test_path,21,21,1,False,2,slice_num)
    # print 'done predictions....'
    # #START OF POST-PROCESSING PIPELINE ###

    # LesionMasker(test_path + '/' , new_prefix)
    # print 'binarizing...'
    # binarize(root, new_prefix + 'Masked_RawOutput.nii')
    # print new_prefix + 'Masked_RawOutput.nii'
    # print 'binarized'

    # callString = 'python analysis.py > ' + new_prefix + '.txt'
    # print callString
    # subprocess.call(callString, shell = True)

    # calculateDice(new_prefix + '.txt', testingImages)
# # # # #    
    
                

                
