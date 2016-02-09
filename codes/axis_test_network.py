import theano.tensor as T
import theano
import numpy as np
import cPickle
import time
import mha
from sklearn.feature_extraction import image
import os
#import itk
import nibabel as nib

def initializeNetwork(prefix):
    
    openFile_fineTuning = open(prefix+'fine_tuning.pkl','rb')
    hidden_layers_sizes = cPickle.load(openFile_fineTuning)
    ins = T.matrix('ins')
    
    theano_weights = []
    theano_biases = []
    theano_layers = []
    for i in xrange(len(hidden_layers_sizes) + 1):
        theano_weights.append(T.matrix('weights'))
        theano_biases.append(T.vector('biases'))
        if i is 0:
            theano_layers.append(T.nnet.sigmoid(T.dot(ins,theano_weights[i]) + theano_biases[i]))
        elif i is len(hidden_layers_sizes):
            theano_layers.append(T.nnet.softmax(T.dot(theano_layers[i-1],theano_weights[i]) + theano_biases[i]))
        else:
            theano_layers.append(T.nnet.sigmoid(T.dot(theano_layers[i-1],theano_weights[i]) + theano_biases[i]))
        
    answer = T.argmax(theano_layers[len(hidden_layers_sizes)],axis=1)
    answer_prob = theano_layers[len(hidden_layers_sizes)]
    print ins
    print theano_weights
    print theano_biases
    
    _input = [ins] + theano_weights + theano_biases
    
    activate2 = theano.function(inputs = _input, outputs=answer_prob,allow_input_downcast=True)
    
    
    genVariables_load = cPickle.load(openFile_fineTuning)

    W_list = []
    b_list = []

    for i in xrange(len(hidden_layers_sizes) + 1):
        W_list.append(cPickle.load(openFile_fineTuning))
        b_list.append(cPickle.load(openFile_fineTuning))
        
    return activate2, W_list, b_list


def predictOutput(test_data, activate2, W_list, b_list):
    '''To predict output from the given test image'''
    if len(W_list) is 4:
        layers2 = activate2(test_data,W_list[0],W_list[1],W_list[2],W_list[3],b_list[0],b_list[1],b_list[2],b_list[3])
    elif len(W_list) is 5:
        layers2 = activate2(test_data,W_list[0],W_list[1],W_list[2],W_list[3],W_list[4],b_list[0],b_list[1],b_list[2],b_list[3],b_list[4])
    elif len(W_list) is 5:
        layers2 = activate2(test_data,W_list[0],W_list[1],W_list[2],W_list[3],W_list[4],W_list[5],b_list[0],b_list[1],b_list[2],b_list[3],b_list[4],b_list[5])
    return layers2

def classify_test_data(activate2, W_list, b_list, path, patch_size, prefix, recon_flag, slice_num=1):
    Flair = []
    T1 = []
    T2 = []
    T_1c = []
    Recon = []
    Folder = []
    Truth=[]
    Subdir_array = []
    label_num = 5

    for subdir, dirs, files in os.walk(path):
        # if len(Flair) is 1:
        #     break
        for file1 in files:
            #print file1
            if file1[-3:]=='nii' and ( 'Flair' in file1):
                Flair.append(file1)
                Folder.append(subdir+'/')
                Subdir_array.append(subdir[-5:])
            elif file1[-3:]=='nii' and ('T1' in file1 and 'T1c' not in file1):
                T1.append(file1)
            elif file1[-3:]=='nii' and ('T2' in file1):
                T2.append(file1)
            elif file1[-3:]=='nii' and ('T1c' in file1 or 'T_1c' in file1):
                T_1c.append(file1)
            elif file1[-3:]=='mha' and 'OT' in file1:
                Truth.append(file1)            
            elif file1[-3:]=='mha' and 'Recon' in file1:
                Recon.append(file1)
    number_of_images = len(Flair)
    
    for image_iterator in range(number_of_images):
        print 'Iteration : ',image_iterator+1
        print 'Folder : ', Folder[image_iterator]
        print 'T2: ', T2[image_iterator]
        print '... predicting'

        Flair_image = nib.load(Folder[image_iterator]+Flair[image_iterator])
        T1_image = nib.load(Folder[image_iterator]+T1[image_iterator])
        T2_image = nib.load(Folder[image_iterator]+T2[image_iterator])
        T_1c_image = nib.load(Folder[image_iterator]+T_1c[image_iterator])
        if recon_flag is True:
            Recon_image = mha.new(Folder[image_iterator]+Recon[image_iterator])
        Flair_image = Flair_image.get_data()
        T1_image = T1_image.get_data()
        T2_image = T2_image.get_data()
        T_1c_image = T_1c_image.get_data()
        if recon_flag is True:
            Recon_image = Recon_image.data
        print 'Input shape: ', Flair_image.shape
        if slice_num ==2:
            Flair_image = np.swapaxes(Flair_image,0,1)
            Flair_image = np.swapaxes(Flair_image,1,2)
            T1_image = np.swapaxes(T1_image, 0,1)
            T1_image = np.swapaxes(T1_image, 1, 2)
            T2_image = np.swapaxes(T2_image, 0,1)
            T2_image = np.swapaxes(T2_image, 1, 2)
            T_1c_image = np.swapaxes(T_1c_image, 0,1)
            T_1c_image = np.swapaxes(T_1c_image, 1, 2)
        elif slice_num == 3:
            Flair_image = np.swapaxes(Flair_image,0,1)
            Flair_image = np.swapaxes(Flair_image,0,2)
            T1_image = np.swapaxes(T1_image, 0,1)
            T1_image = np.swapaxes(T1_image, 0, 2)
            T2_image = np.swapaxes(T2_image, 0,1)
            T2_image = np.swapaxes(T2_image, 0, 2)
            T_1c_image = np.swapaxes(T_1c_image, 0,1)
            T_1c_image = np.swapaxes(T_1c_image, 0, 2)
        xdim, ydim, zdim = Flair_image.shape
        prediction_image = []
        # if slice_num ==1:
        #     prediction_image = np.zeros([(xdim-patch_size+1)*(ydim-patch_size+1), zdim])
        # elif slice_num==2 or slice_num==3:
        #     prediction_image = np.zeros([zdim, (ydim-patch_size+1)*(xdim-patch_size+1)])
        print 
        print 'shape:',Flair_image.shape
        for i in range(zdim):
            Flair_slice = np.transpose(Flair_image[:,:,i])
            T1_slice = np.transpose(T1_image[:,:,i])
            T2_slice = np.transpose(T2_image[:,:,i])
            T_1c_slice = np.transpose(T_1c_image[:,:,i])
            if recon_flag is True:
                Recon_slice = np.transpose(Recon_image[:,:,i])

            Flair_patch = image.extract_patches_2d(Flair_slice, (patch_size, patch_size))
            F_P=Flair_patch.reshape(len(Flair_patch),patch_size*patch_size)
            T1_patch = image.extract_patches_2d(T1_slice, (patch_size, patch_size))
            T1_P=T1_patch.reshape(len(Flair_patch),patch_size*patch_size)
            T2_patch = image.extract_patches_2d(T2_slice, (patch_size, patch_size))
            T2_P=T2_patch.reshape(len(Flair_patch),patch_size*patch_size)
            T_1c_patch = image.extract_patches_2d(T_1c_slice, (patch_size, patch_size))
            T1c_P=T_1c_patch.reshape(len(Flair_patch),patch_size*patch_size)
            temp_patch = np.concatenate([F_P,T1_P,T2_P,T1c_P],axis=1)
        
            prediction_slice = predictOutput(temp_patch, activate2, W_list, b_list)
            # print 'Shape of slice : ', prediction_slice.shape
            prediction_image.append(prediction_slice)

        prediction_image = np.array(prediction_image)
        print 'Prediction shape : ', prediction_image.shape
        #
        print 'Slice num : ',slice_num
        if slice_num==1:
            prediction_image = np.transpose(prediction_image,(1,0,2))
            print 'Look here : ', prediction_image.shape 
            prediction_image = prediction_image.reshape([xdim-patch_size+1, ydim-patch_size+1, zdim, label_num])
            print 'Look after : ', prediction_image.shape 
            output_image = np.zeros([xdim,ydim,zdim,label_num])
            
            output_image[1+((patch_size-1)/2):xdim-((patch_size-1)/2)+1,1+((patch_size-1)/2):ydim-((patch_size-1)/2)+1,:] = prediction_image 
            print 'output shape', output_image.shape
        elif slice_num==2:
            prediction_image = prediction_image.reshape([zdim, ydim-patch_size+1, xdim-patch_size+1, label_num])
            output_image = np.zeros([xdim,ydim,zdim,label_num])
            output_image[:,1+((patch_size-1)/2):ydim-((patch_size-1)/2)+1,1+((patch_size-1)/2):xdim-((patch_size-1)/2)+1] = prediction_image

            output_image = np.swapaxes(output_image,2,1)
            output_image = np.swapaxes(output_image,1,0)
        elif slice_num == 3:
            prediction_image = prediction_image.reshape([zdim, ydim-patch_size+1, xdim-patch_size+1, label_num])
            output_image = np.zeros([zdim,ydim,xdim,label_num])
            output_image[:,1+((patch_size-1)/2):ydim-((patch_size-1)/2)+1,1+((patch_size-1)/2):xdim-((patch_size-1)/2)+1,:] = prediction_image
# print 
#        np.save(,output_image)#TODO: save it in meaningful name in corresponding folder
#        break
            
        print 'Output shape : ', output_image.shape  
        np.save(Folder[image_iterator]+Subdir_array[image_iterator]+prefix+'.npy', output_image)
        output_image = np.argmax(output_image,axis = 3)  

        a = output_image
        for j in xrange(a.shape[2]):
            a[:,:,j] = np.transpose(a[:,:,j])
        print 'a shape here: ', a.shape
        output_image = a
        # save_image = np.zeros()
        # output_image = itk_py_converter.GetImageFromArray(a.tolist())
        # writer.SetFileName(Folder[image_iterator]+Subdir_array[image_iterator]+'_'+prefix+'_.mha')
        # writer.SetInput(output_image)
        # writer.Update()

        affine=[[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
        img = nib.Nifti1Image(output_image, affine)
        img.set_data_dtype(np.int32)
        nib.save(img,Folder[image_iterator]+Subdir_array[image_iterator]+prefix+'xyz.nii')
        print '########success########'

def axis_test_network(root,prefix, path, patch_size_x, patch_size_y, patch_size_z, recon_flag, patch_shape=3, slice_num=1):
    
    print 'Loading Network....'
    activate2, W_list, b_list = initializeNetwork(root+prefix)
    print 'Network Loaded!'    
    pred = []
    print '#################'
    #print 'itk instantiated'
    print '#################'        
    print 'Entering loop...'
    print 'Predicting outputs....'
    start_time = time.clock()    
    print 'classify_test_data'
    if patch_shape == 2:
        classify_test_data(activate2, W_list,b_list,path,patch_size_x, prefix,recon_flag,slice_num)
    else:
        classify_test_data_3d(activate2, W_list, b_list, path, patch_size_x, patch_size_y, patch_size_z, prefix, recon_flag)
    end_time = time.clock()
    print 'Classification Time Taken: ', ((end_time - start_time)/60.)
    
if __name__ == '__main__':

    prefix = 'rms_9x9x9_perfect_5000-2000-500_M111'
    root = '../../varghese/Recon_2013_data/'
    test_root = '/media/brain/1A34723D34721BC7/BRATS/codes/results/test_258_rms_9x9x9_perfect_5000-2000-500_M111/'
    test_path = root+'testing'
#    test_path = '/home/bmi/varghese/sample/'
    for i in xrange(1):
        test_network(test_root,prefix + '_' + str(i) + 'dropout_',test_path,9,9,9,False,3)
    
#    convert_mha(root+'testing', prefix, 3) 
        
