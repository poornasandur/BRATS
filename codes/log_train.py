import numpy as np
from mha import *
from mha2 import *
import os
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import nibabel as nib
from lesion_masker import *
from binarize_output import *
import subprocess
from diceplotter import *
import cPickle

def extract_input(path, prefix1, prefix2, prefix3, mse_prefix, out_pref):
    mse_flag = True
    One = []
    Two = []
    Three = []
    Truth = []
    Folder = []
    Output = []
    MSE = []
    l = len('dropout_xyz.nii')+len(prefix1)
    for subdir, dirs, files in os.walk(path):
        # if len(Truth)==2:
        #     break
        for file1 in files:
            if (file1[-3:]=='npy') and (prefix1 in file1) and (prefix2 not in file1) and (prefix3 not in file1):
                One.append(file1)
                Folder.append(subdir+'/')
            elif file1[-3:]=='npy' and prefix2 in file1:
                Two.append(file1)
            elif file1[-3:]=='npy' and prefix3 in file1:
                Three.append(file1)
            elif file1[-3:]=='mha' and 'OT' in file1:
                Truth.append(file1)
            elif file1[-l:]==prefix1+'dropout_xyz.nii':
                Output.append(file1)
            elif file1[-3:]=='npy' and mse_prefix in file1:
                MSE.append(file1)



    number_of_patients = len(One)
    print 'Number of patients : ', number_of_patients
    print len(One), len(Output), len(Two), len(MSE)
    
    maxim = 5000000
    if mse_flag==True:
        train_file = np.zeros([maxim,16])
    else:
        train_file = np.zeros([maxim,15])
    train_label = np.zeros(maxim)
    counter = 0
    flag = 1
    offset_x = 3
    offset_y = 3
    offset_z = 2



    for i in xrange(number_of_patients):
        print
        print ' Folder : ', i+1

        print Folder[i]
        # print One[i]
        # print Two[i]
        # print Three[i]
        print
        try:
            Truth_image = new(Folder[i]+Truth[i])
        except:
            Truth_image = new2(Folder[i]+Truth[i])
        Truth_image = Truth_image.data
        axis1 = np.load(Folder[i]+One[i])
        axis2 = np.load(Folder[i]+Two[i])
        axis3 = np.load(Folder[i]+Three[i])

        axis1 = np.transpose(axis1,(1,0,2,3))
        axis2 = np.transpose(axis2,(1,0,2,3))
        axis3 = np.transpose(axis3,(1,0,2,3))

        axis_predict = nib.load(Folder[i]+Output[i])
        axis_predict = axis_predict.get_data()
        if mse_flag==True:
            mse = np.load(Folder[i]+MSE[i])

        x_span,y_span,z_span = np.where(axis_predict!=0)
        padding = 0      
        z_start = np.min(z_span)-padding
        z_stop = np.max(z_span)+padding
        x_start = np.min(x_span)-padding
        x_stop = np.max(x_span)+padding
        y_start = np.min(y_span)-padding
        y_stop = np.max(y_span)+padding
        
        print 'Extracting rows with offset'
        Truth_image = Truth_image[x_start:x_stop:offset_x, y_start:y_stop:offset_y, z_start:z_stop:offset_z]
        axis1 = axis1[x_start:x_stop:offset_x, y_start:y_stop:offset_y, z_start:z_stop:offset_z, :]
        axis2 = axis2[x_start:x_stop:offset_x, y_start:y_stop:offset_y, z_start:z_stop:offset_z, :]
        axis3 = axis3[x_start:x_stop:offset_x, y_start:y_stop:offset_y, z_start:z_stop:offset_z, :]
        if mse_flag==True:
            mse = mse[x_start:x_stop:offset_x, y_start:y_stop:offset_y, z_start:z_stop:offset_z]

        class_num = 5
        num_of_class = []

        print 'reshaping'


        ###########################################
        axis1 = axis1.reshape(axis1.shape[0]*axis1.shape[1]*axis1.shape[2],class_num)
        axis2 = axis2.reshape(axis2.shape[0]*axis2.shape[1]*axis2.shape[2],class_num)
        axis3 = axis3.reshape(axis3.shape[0]*axis3.shape[1]*axis3.shape[2],class_num)
        Truth_image = Truth_image.reshape(Truth_image.shape[0]*Truth_image.shape[1]*Truth_image.shape[2])
        if mse_flag==True:
            mse = mse.reshape(mse.shape[0]*mse.shape[1]*mse.shape[2])

        num = axis1.shape[0]

        if [0,0,0,0,0] in axis2:
            print 'Found zeros in axis2! Check this patient'
            axis2[np.where(axis2==[0,0,0,0,0])[0],:] = [0.99999, 0.0000025, 0.0000025, 0.0000025, 0.0000025]
            print Folder[i]+Two[i]
        if [0,0,0,0,0] in axis3:
            print 'Found zeros in axis3! Check this patient'
            axis3[np.where(axis3==[0,0,0,0,0])[0],:] = [0.99999, 0.0000025, 0.0000025, 0.0000025, 0.0000025]
            print Folder[i]+Three[i]



        if counter+num > maxim:
            print 'Reached maximum. Hence breaking'
            break

        print 'Adding rows : ', num
        # for j in range(5):
        #     print 'Number of class '+str(j)+' : ',np.sum((Truth_image==j).astype(int))

        train_file[counter:counter+num, 0:5] = axis1
        train_file[counter:counter+num, 5:10] = axis2
        train_file[counter:counter+num, 10:15] = axis3
        train_file[counter:counter+num, 15] = mse
        train_label[counter:counter+num] = Truth_image
        counter = counter+num


        # for j in range(class_num):
        #     num_of_class.append(np.sum((Truth_image==j).astype(int)))
        # print num_of_class
        
        # if min(num_of_class)==0:
        #     continue
        # minim = min(x for x in num_of_class if x!=0)
        # if minim>5000:
        #     minim = 5000

        # for j in xrange(class_num):
        #     if num_of_class[j]==0:
        #         continue
        #     indexx, indexy, indexz = np.where(Truth_image==j)
        #     # print 'indexx : ', len(indexx)
        #     index1 = np.arange(len(indexx))
        #     shuffle(index1)
        #     # print 'index1 : ',index1.shape
        #     indexx = indexx[index1[0:minim]]
        #     indexy = indexy[index1[0:minim]]
        #     indexz = indexz[index1[0:minim]]

        #     # a1 = axis1[indexx,:]
        #     # b2 = axis2[indexx,:]
        #     # c3 = axis3[indexx,:]

        #     train_file[counter:counter+minim, 0:5] = axis1[indexx,indexy,indexz,:]
        #     train_file[counter:counter+minim, 5:10] = axis2[indexx,indexy,indexz, :]
        #     train_file[counter:counter+minim, 10:15] =axis3[indexx,indexy,indexz, :]
        #     train_label[counter:counter+minim] = np.ones(minim)*j

        #     counter = counter+minim
        #     if counter>=maxim:
        #         print ' Maximum reached!'
        #         flag = 2
        #         break
        #     print 'Added rows of class '+str(j)+' : '+str(minim)
        # if flag==2:
        #     break
    # counter = counter-minim
    train_file = train_file[0:counter,:]
    train_label = train_label[0:counter]
    print 'Extracted'
    print 'Size of training : ', train_file.shape
    
    if 'training' in path:
        np.save('/media/bmi/varkey/new_n4/Recon_2013_data/log_training_patch/'+out_pref+'_train_input.npy',train_file)
        np.save('/media/bmi/varkey/new_n4/Recon_2013_data/log_training_patch/'+out_pref+'_train_label.npy',train_label)
    elif 'validation' in path:
        np.save('/media/bmi/varkey/new_n4/Recon_2013_data/log_validation_patch/'+out_pref+'_valid_input.npy',train_file)
        np.save('/media/bmi/varkey/new_n4/Recon_2013_data/log_validation_patch/'+out_pref+'_valid_label.npy',train_label)
    elif 'testing' in path:
        np.save('/media/bmi/varkey/new_n4/Recon_2013_data/log_testing_patch/'+out_pref+'_test_input.npy',train_file)
        np.save('/media/bmi/varkey/new_n4/Recon_2013_data/log_testing_patch/'+out_pref+'_test_label.npy',train_label)

def MSE_WT_extract_input(path, prefix1, prefix2, prefix3, mse_prefix, out_pref):
    mse_flag = True
    One = []
    Two = []
    Three = []
    Truth = []
    Folder = []
    Output = []
    MSE = []

    l = len('dropout_xyz.nii')+len(prefix1)
    for subdir, dirs, files in os.walk(path):
        # if len(Truth)==2:
        #     break
        for file1 in files:
            if (file1[-3:]=='npy') and (prefix1 in file1) and (prefix2 not in file1) and (prefix3 not in file1):
                One.append(file1)
                Folder.append(subdir+'/')
            elif file1[-3:]=='npy' and prefix2 in file1:
                Two.append(file1)
            elif file1[-3:]=='npy' and prefix3 in file1:
                Three.append(file1)
            elif file1[-3:]=='mha' and 'OT' in file1:
                Truth.append(file1)
            elif file1[-l:]==prefix1+'dropout_xyz.nii':
                Output.append(file1)
            elif file1[-3:]=='npy' and mse_prefix in file1:
                MSE.append(file1)



    number_of_patients = len(One)
    print 'Number of patients : ', number_of_patients
    print len(One), len(Output), len(Two), len(MSE)
    
    maxim = 5000000
    if mse_flag==True:
        train_file = np.zeros([maxim,7])
    else:
        train_file = np.zeros([maxim,6])
    train_label = np.zeros(maxim)
    counter = 0
    flag = 1
    offset_x = 3
    offset_y = 3
    offset_z = 2



    for i in xrange(number_of_patients):
        print
        print ' Folder : ', i+1

        print Folder[i]
        # print One[i]
        # print Two[i]
        # print Three[i]
        print
        try:
            Truth_image = new(Folder[i]+Truth[i])
        except:
            Truth_image = new2(Folder[i]+Truth[i])
        Truth_image = Truth_image.data
        axis1 = np.load(Folder[i]+One[i])
        axis2 = np.load(Folder[i]+Two[i])
        axis3 = np.load(Folder[i]+Three[i])

        axis1 = np.transpose(axis1,(1,0,2,3))
        axis2 = np.transpose(axis2,(1,0,2,3))
        axis3 = np.transpose(axis3,(1,0,2,3))

        axis_predict = nib.load(Folder[i]+Output[i])
        axis_predict = axis_predict.get_data()
        if mse_flag==True:
            mse = np.load(Folder[i]+MSE[i])

        x_span,y_span,z_span = np.where(axis_predict!=0)
        padding = 0      
        z_start = np.min(z_span)-padding
        z_stop = np.max(z_span)+padding
        x_start = np.min(x_span)-padding
        x_stop = np.max(x_span)+padding
        y_start = np.min(y_span)-padding
        y_stop = np.max(y_span)+padding
        
        print 'Extracting rows with offset'
        Truth_image = Truth_image[x_start:x_stop:offset_x, y_start:y_stop:offset_y, z_start:z_stop:offset_z]
        axis1 = axis1[x_start:x_stop:offset_x, y_start:y_stop:offset_y, z_start:z_stop:offset_z, :]
        axis2 = axis2[x_start:x_stop:offset_x, y_start:y_stop:offset_y, z_start:z_stop:offset_z, :]
        axis3 = axis3[x_start:x_stop:offset_x, y_start:y_stop:offset_y, z_start:z_stop:offset_z, :]
        if mse_flag==True:
            mse = mse[x_start:x_stop:offset_x, y_start:y_stop:offset_y, z_start:z_stop:offset_z]

        class_num = 5
        num_of_class = []

        print 'reshaping'


        ###########################################
        axis1 = axis1.reshape(axis1.shape[0]*axis1.shape[1]*axis1.shape[2],class_num)
        axis2 = axis2.reshape(axis2.shape[0]*axis2.shape[1]*axis2.shape[2],class_num)
        axis3 = axis3.reshape(axis3.shape[0]*axis3.shape[1]*axis3.shape[2],class_num)
        Truth_image = Truth_image.reshape(Truth_image.shape[0]*Truth_image.shape[1]*Truth_image.shape[2])
        if mse_flag==True:
            mse = mse.reshape(mse.shape[0]*mse.shape[1]*mse.shape[2])

        num = axis1.shape[0]

        if [0,0,0,0,0] in axis2:
            print 'Found zeros in axis2! Check this patient'
            axis2[np.where(axis2==[0,0,0,0,0])[0],:] = [0.99999, 0.0000025, 0.0000025, 0.0000025, 0.0000025]
            print Folder[i]+Two[i]
        if [0,0,0,0,0] in axis3:
            print 'Found zeros in axis3! Check this patient'
            axis3[np.where(axis3==[0,0,0,0,0])[0],:] = [0.99999, 0.0000025, 0.0000025, 0.0000025, 0.0000025]
            print Folder[i]+Three[i]



        if counter+num > maxim:
            print 'Reached maximum. Hence breaking'
            break

        print 'Adding rows : ', num
        # for j in range(5):
        #     print 'Number of class '+str(j)+' : ',np.sum((Truth_image==j).astype(int))

        train_file[counter:counter+num, 0] = axis1[:,0]
        train_file[counter:counter+num, 1] = np.sum(axis1[:,1:],axis=1)
        train_file[counter:counter+num, 2] = axis2[:,0]
        train_file[counter:counter+num, 3] = np.sum(axis2[:,1:],axis=1)
        train_file[counter:counter+num, 4] = axis3[:,0]
        train_file[counter:counter+num, 5] = np.sum(axis3[:,1:],axis=1)
        train_file[counter:counter+num, 6] = mse
        train_label[counter:counter+num] = Truth_image
        counter = counter+num

    train_file = train_file[0:counter,:]
    train_label = train_label[0:counter]
    train_label[train_label>0] = 1
    print 'Extracted'
    print 'Size of training : ', train_file.shape
    print 'Truth changed to 1/0 for WT training'
    
    if 'training' in path:
        np.save('/media/bmi/varkey/new_n4/Recon_2013_data/log_training_patch/'+out_pref+'_train_input.npy',train_file)
        np.save('/media/bmi/varkey/new_n4/Recon_2013_data/log_training_patch/'+out_pref+'_train_label.npy',train_label)
    elif 'validation' in path:
        np.save('/media/bmi/varkey/new_n4/Recon_2013_data/log_validation_patch/'+out_pref+'_valid_input.npy',train_file)
        np.save('/media/bmi/varkey/new_n4/Recon_2013_data/log_validation_patch/'+out_pref+'_valid_label.npy',train_label)
    elif 'testing' in path:
        np.save('/media/bmi/varkey/new_n4/Recon_2013_data/log_testing_patch/'+out_pref+'_test_input.npy',train_file)
        np.save('/media/bmi/varkey/new_n4/Recon_2013_data/log_testing_patch/'+out_pref+'_test_label.npy',train_label)

def train_log_reg(root, patch_prefix, prefix, log):
    train_folder = root +'log_training_patch/'
    valid_folder = root + 'log_validation_patch/'

    train_input = np.load(train_folder+patch_prefix+'_train_input.npy')
    train_label = np.load(train_folder+patch_prefix+'_train_label.npy')
    # valid_input = np.load(valid_folder+prefix+'_valid_input.npy')
    # valid_label = np.load(valid_folder+prefix+'_valid_label.npy')
    train_input = train_input[:,0:6]
    if log==True:
        print ' Converting to log'
        train_input = np.log(train_input)
        prefix = prefix+'_log' 
    print 'Training Logistic regression : '
    print '...'

    # model = LogisticRegression(solver='newton-cg',multi_class='multinomial')
    model = LogisticRegression()
    model = model.fit(train_input, train_label)
    print 'Logistic regression trained!'

    pickle_file = open(prefix+'.pkl', 'wb')
    cPickle.dump(model,pickle_file, protocol = cPickle.HIGHEST_PROTOCOL)
    pickle_file.close()
    return model

def test_log_reg(test_path, prefix1, prefix2, prefix3, mse_prefix, out_prefix, model, log=False):
    One = []
    Two = []
    Three = []
    Folder = []
    MSE = []
    Subdir_array = []
    for subdir, dirs, files in os.walk(test_path):
        # if len(One)==2:
        #     break
        for file1 in files:
            if (file1[-3:]=='npy') and (prefix1 in file1) and (prefix2 not in file1) and (prefix3 not in file1):
                One.append(file1)
                Folder.append(subdir+'/')
            elif file1[-3:]=='npy' and prefix2 in file1:
                Two.append(file1)
            elif file1[-3:]=='npy' and prefix3 in file1:
                Three.append(file1)
            elif file1[-3:]=='npy' and mse_prefix in file1:
                MSE.append(file1)
    number_of_patients = len(One)
    print 'Number of patients : ', number_of_patients

    for i in range(number_of_patients):
        print 'Testing patient : '
        print 'Iteration : ', i+1
        print Folder[i]
        axis1 = np.load(Folder[i]+One[i])
        axis2 = np.load(Folder[i]+Two[i])
        axis3 = np.load(Folder[i]+Three[i])

        axis1 = np.transpose(axis1,(1,0,2,3))
        axis2 = np.transpose(axis2,(1,0,2,3))
        axis3 = np.transpose(axis3,(1,0,2,3))
        mse = np.load(Folder[i]+MSE[i])



        zero_ind = np.where(axis1==[0,0,0,0,0])
        axis1[zero_ind[0], zero_ind[1], zero_ind[2],:] = [0.99999, 0.0000025, 0.0000025, 0.0000025, 0.0000025]
        zero_ind = np.where(axis2==[0,0,0,0,0])
        axis2[zero_ind[0], zero_ind[1], zero_ind[2],:] = [0.99999, 0.0000025, 0.0000025, 0.0000025, 0.0000025]
        zero_ind = np.where(axis3==[0,0,0,0,0])
        axis3[zero_ind[0], zero_ind[1], zero_ind[2],:] = [0.99999, 0.0000025, 0.0000025, 0.0000025, 0.0000025]

        xdim, ydim, zdim, class_num = axis1.shape
        print 'Dimensions : ', xdim, ydim, zdim

        
        test_mat = np.zeros([xdim, ydim, zdim, (class_num*3)+1])
        test_mat[:,:,:,0:class_num] = axis1
        test_mat[:,:,:,class_num:2*class_num] = axis2
        test_mat[:,:,:,2*class_num:3*class_num] = axis3
        test_mat[:,:,:,15] = mse
        del axis1
        del axis2
        del axis3

        test_mat = test_mat.reshape(xdim*ydim*zdim, (3*class_num)+1)
        if log==True:
            test_mat = np.log(test_mat)
        out = model.predict(test_mat)
        output = out.reshape(xdim,ydim,zdim)
        # test_mat = np.argmax(test_mat, axis=3)
        # test_mat = test_mat%5
        # output = test_mat

        affine=[[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
        img = nib.Nifti1Image(output, affine)
        img.set_data_dtype(np.int32)
        nib.save(img,Folder[i]+out_prefix+'.nii')

def test_log_reg_MSEWT(test_path, prefix1, prefix2, prefix3, mse_prefix, out_prefix, model, log=False):
    One = []
    Two = []
    Three = []
    Folder = []
    MSE = []
    Subdir_array = []
    for subdir, dirs, files in os.walk(test_path):
        # if len(One)==2:
        #     break
        for file1 in files:
            if (file1[-3:]=='npy') and (prefix1 in file1) and (prefix2 not in file1) and (prefix3 not in file1):
                One.append(file1)
                Folder.append(subdir+'/')
            elif file1[-3:]=='npy' and prefix2 in file1:
                Two.append(file1)
            elif file1[-3:]=='npy' and prefix3 in file1:
                Three.append(file1)
            elif file1[-3:]=='npy' and mse_prefix in file1:
                MSE.append(file1)
    number_of_patients = len(One)
    print 'Number of patients : ', number_of_patients

    for i in range(number_of_patients):
        print 'Testing patient : '
        print 'Iteration : ', i+1
        print Folder[i]
        axis1 = np.load(Folder[i]+One[i])
        axis2 = np.load(Folder[i]+Two[i])
        axis3 = np.load(Folder[i]+Three[i])

        axis1 = np.transpose(axis1,(1,0,2,3))
        axis2 = np.transpose(axis2,(1,0,2,3))
        axis3 = np.transpose(axis3,(1,0,2,3))
        mse = np.load(Folder[i]+MSE[i])



        zero_ind = np.where(axis1==[0,0,0,0,0])
        axis1[zero_ind[0], zero_ind[1], zero_ind[2],:] = [0.99999, 0.0000025, 0.0000025, 0.0000025, 0.0000025]
        zero_ind = np.where(axis2==[0,0,0,0,0])
        axis2[zero_ind[0], zero_ind[1], zero_ind[2],:] = [0.99999, 0.0000025, 0.0000025, 0.0000025, 0.0000025]
        zero_ind = np.where(axis3==[0,0,0,0,0])
        axis3[zero_ind[0], zero_ind[1], zero_ind[2],:] = [0.99999, 0.0000025, 0.0000025, 0.0000025, 0.0000025]

        xdim, ydim, zdim, class_num = axis1.shape
        print 'Dimensions : ', xdim, ydim, zdim

        
        test_mat = np.zeros([xdim, ydim, zdim, 6])
        test_mat[:,:,:,0] = axis1[:,:,:,0]
        test_mat[:,:,:,1] = np.sum(axis1[:,:,:,1:], axis=3)
        test_mat[:,:,:,2] = axis2[:,:,:,0]
        test_mat[:,:,:,3] = np.sum(axis2[:,:,:,1:], axis=3)
        test_mat[:,:,:,4] = axis3[:,:,:,0]
        test_mat[:,:,:,5] = np.sum(axis3[:,:,:,1:], axis=3)
        # test_mat[:,:,:,6] = mse
        del axis1
        del axis2
        del axis3

        test_mat = test_mat.reshape(xdim*ydim*zdim, 6)
        if log==True:
            test_mat = np.log(test_mat)
        out = model.predict(test_mat)
        output = out.reshape(xdim,ydim,zdim)
        # test_mat = np.argmax(test_mat, axis=3)
        # test_mat = test_mat%5
        # output = test_mat

        affine=[[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
        img = nib.Nifti1Image(output, affine)
        img.set_data_dtype(np.int32)
        nib.save(img,Folder[i]+out_prefix+'.nii')
        
def test_geom_mean(test_path, prefix1, prefix2, prefix3, out_prefix):
    One = []
    Two = []
    Three = []
    Folder = []
    Subdir_array = []
    for subdir, dirs, files in os.walk(test_path):
        # if len(One)==2:
        #     break
        for file1 in files:
            if (file1[-3:]=='npy') and (prefix1 in file1) and (prefix2 not in file1) and (prefix3 not in file1):
                One.append(file1)
                Folder.append(subdir+'/')
            elif file1[-3:]=='npy' and prefix2 in file1:
                Two.append(file1)
            elif file1[-3:]=='npy' and prefix3 in file1:
                Three.append(file1)
    number_of_patients = len(One)
    print 'Number of patients : ', number_of_patients

    for i in range(number_of_patients):
        print 'Testing patient : '
        print 'Iteration : ', i+1
        print Folder[i]
        axis1 = np.load(Folder[i]+One[i])
        axis2 = np.load(Folder[i]+Two[i])
        axis3 = np.load(Folder[i]+Three[i])

        axis1 = np.transpose(axis1,(1,0,2,3))
        axis2 = np.transpose(axis2,(1,0,2,3))
        axis3 = np.transpose(axis3,(1,0,2,3))

        xdim, ydim, zdim, class_num = axis1.shape
        print 'Dimensions : ', xdim, ydim, zdim

        test_mat = axis1*axis2*axis3
        x = float(1)/3
        test_mat = np.power(test_mat, x)

        # test_mat = np.zeros([xdim, ydim, zdim, class_num*3])
        # test_mat[:,:,:,0:class_num] = axis1
        # test_mat[:,:,:,class_num:2*class_num] = axis2
        # test_mat[:,:,:,2*class_num:3*class_num] = axis3
        del axis1
        del axis2
        del axis3

        # test_mat = test_mat.reshape(xdim*ydim*zdim, 3*class_num)
        # out = model.predict(test_mat)
        # output = out.reshape(xdim,ydim,zdim)
        test_mat = np.argmax(test_mat, axis=3)
        # test_mat = test_mat%5
        output = test_mat

        affine=[[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
        img = nib.Nifti1Image(output, affine)
        img.set_data_dtype(np.int32)
        nib.save(img,Folder[i]+out_prefix+'.nii')









prefix1 = 'new_n4'
prefix2 = 'new_n4_axis2'
prefix3 = 'new_n4_axis3'
mse_prefix = 'flair_t2_2500_noise'
out_prefix = 'new_n4_mse_WT2'
patch_prefix = 'new_n4_mse_WT'

#########################################################################
# Extracting the numpy matrix

print 'Doing log_training folder : '
path = '/media/bmi/varkey/new_n4/Recon_2013_data/log_training/'
# extract_input(path, prefix1, prefix2, prefix3, mse_prefix, out_prefix)
# MSE_WT_extract_input(path, prefix1, prefix2, prefix3, mse_prefix, out_prefix)

# print '###################################################################'

# print 'Doing log_validation folder : '
# path = '/media/bmi/varkey/new_n4/Recon_2013_data/log_validation/'
# extract_input(path, prefix1, prefix2, prefix3, out_prefix)

# print '###################################################################'

# print 'Doing N4_zscore_testing_t1_t1c_hist_match folder : '
# path = '/media/bmi/varkey/new_n4/Recon_2013_data/N4_zscore_testing_t1_t1c_hist_match/'
# extract_input(path, prefix1, prefix2, prefix3, out_prefix)

############################################################################

#Logistic regression Training

root = '/media/bmi/varkey/new_n4/Recon_2013_data/'

log = True

model = train_log_reg(root, patch_prefix, out_prefix, log)
# pickle_file = out_prefix+'.pkl'
# pickle_file = open(pickle_file, 'rb')
# model = cPickle.load(pickle_file)
# pickle_file.close()

if log==True:
    out_prefix = out_prefix+'_log'





#####################################


#Testing
save_prefix = out_prefix
test_path = '/media/bmi/varkey/new_n4/Recon_2013_data/N4_zscore_testing_t1_t1c_hist_match/'
test_log_reg_MSEWT(test_path, prefix1, prefix2, prefix3, mse_prefix, save_prefix, model, log)
# test_log_reg(test_path, prefix1, prefix2, prefix3, mse_prefix, save_prefix, model, log)
# test_log_reg(test_path, prefix1, prefix2, prefix3, save_prefix)


root=test_path
LesionMasker(root,save_prefix)

binarize(root, save_prefix+'Masked_RawOutput.nii')
# new_prefix=save_prefix
# prefix_list = [new_prefix + 'Masked_RawOutput.nii' + 'WT', new_prefix + 'Masked_RawOutput.nii'+ 'TC', new_prefix + 'Masked_RawOutput.nii'+ 'AT']
# print 'Calling log_analysis'
# callString = 'python log_analysis.py > ' + save_prefix + '.txt'
# print callString
# subprocess.call(callString, shell = True)
# testingImages=35 # no. of images 
# calculateDice(save_prefix + '.txt', testingImages)