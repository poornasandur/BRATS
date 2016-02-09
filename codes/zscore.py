import scipy
import os
import numpy as np
from mha import *
import nibabel as nib
import shutil
# import itk
# print 'setting up itk'
# image_type = itk.Image[itk.F,3]
# writer = itk.ImageFileWriter[image_type].New()
# itk_py_converter = itk.PyBuffer[image_type]
# print 'done'
input_path = '/media/bmi/MyPassport/mover/post_n4_training_hist_match'
output_path = '/media/bmi/MyPassport/mover/n4_training_hist_match_z_score'

# if os.path.exists(output_path):
# 	os.makedirs(output_path)
Flair = []
T1 = []
T2 = []
T1c = []
Truth = []
Folder = []

for subdir, dirs, files in os.walk(input_path):
    for file1 in files:
        if file1[-3:]=='mha' and 'Flair' in file1:
            Flair.append(file1)
            Folder.append(subdir+'/')
        elif file1[-3:]=='mha' and ('T1' in file1 and 'T1c' not in file1):
            T1.append(file1)
        elif file1[-3:]=='mha' and ('T2' in file1):
            T2.append(file1)
        elif file1[-3:]=='mha' and ('T1c' in file1 or 'T_1c' in file1):
            T1c.append(file1)
        elif file1[-3:]=='mha' and 'OT' in file1:
            Truth.append(file1)
number_of_images = len(Flair)
print 'Number of Patients : ', number_of_images

for image_iterator in range(number_of_images):
    print 'Image number : ',image_iterator+1
    print 'Folder : ', Folder[image_iterator]
    Flair_image = new(Folder[image_iterator]+Flair[image_iterator])
    T1_image = new(Folder[image_iterator]+T1[image_iterator])
    T2_image = new(Folder[image_iterator]+T2[image_iterator])
    T1c_image = new(Folder[image_iterator]+T1c[image_iterator])
    Truth_image = new( Folder[image_iterator] + Truth[image_iterator] )
    
    Flair_image = Flair_image.data
    T1_image = T1_image.data
    T2_image = T2_image.data
    T1c_image = T1c_image.data
    Truth_image = Truth_image.data

    Flair_mean = np.mean(Flair_image[Flair_image!=0])
    Flair_std = np.std(Flair_image[Flair_image!=0])
    T1_mean = np.mean(T1_image[Flair_image!=0])
    T1_std = np.std(T1_image[Flair_image!=0])
    T2_mean = np.mean(T2_image[Flair_image!=0])
    T2_std = np.std(T2_image[Flair_image!=0])
    T1c_mean = np.mean(T1c_image[Flair_image!=0])
    T1c_std = np.std(T1c_image[Flair_image!=0])

    print('Flair mean: ', Flair_mean)
    print('Flair Std: ', Flair_std)

    Flair_image = (Flair_image-Flair_mean)/Flair_std
    T1_image = (T1_image-T1_mean)/T1_std
    T2_image = (T2_image-T2_mean)/T2_std
    T1c_image = (T1c_image-T1c_mean)/T1c_std

    print('Size: ', np.size(Flair_image))
    print 'writing the images'

    affine=[[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
    img_flair = nib.Nifti1Image(Flair_image, affine)
    img_flair.set_data_dtype(np.int32)
    patient_id=Folder[image_iterator].split('/')

    print 'patient id:',patient_id[-2]

    ids=patient_id[-2]
    if not os.path.exists(output_path+'/'+ids):
    	os.mkdir(output_path+'/'+ids)

    name=Flair[image_iterator].split('mha')
    print name[0]
    nib.save(img_flair,output_path+'/'+ids+'/'+name[0]+'nii')

   
    img_t1 = nib.Nifti1Image(T1_image, affine)
    img_t1.set_data_dtype(np.int32)
    name=T1[image_iterator].split('mha')
    print name[0]
    nib.save(img_t1,output_path+'/'+ids+'/'+name[0]+'nii')

    img_t1c = nib.Nifti1Image(T1c_image, affine)
    img_t1c.set_data_dtype(np.int32)
    name=T1c[image_iterator].split('mha')
    print name[0]
    nib.save(img_t1c,output_path+'/'+ids+'/'+name[0]+'nii')


    img_t2 = nib.Nifti1Image(T2_image, affine)
    img_t2.set_data_dtype(np.int32)
    name=T2[image_iterator].split('mha')
    print name[0]
    nib.save(img_t2,output_path+'/'+ids+'/'+name[0]+'nii')

    # name=Truth[image_iterator].split('mha')
    # print name[0]
    source=input_path+'/'+ids+'/'+Truth[image_iterator]
    print 'source',source
    print '####'
    destination=output_path+'/'+ids
    print destination
    shutil.copy(source,destination)
    # break

    # output_FLAIR = itk_py_converter.GetImageFromArray(Flair_image.tolist())
    # print 'output name:',output_path+'/'+Folder[image_iterator]+Flair[image_iterator]
    # writer.SetFileName(output_path+'/'+Folder[image_iterator]+Flair[image_iterator])
    # writer.SetInput(output_FLAIR)
    # writer.Update()


    # output_T1 = itk_py_converter.GetImageFromArray(T1_image.tolist())
    # print 'output name:',output_path+'/'+Folder[image_iterator]+T1[image_iterator]
    # writer.SetFileName(output_path+'/'+Folder[image_iterator]+T1[image_iterator])
    # writer.SetInput(output_FLAIR)
    # writer.Update()


    # output_T2 = itk_py_converter.GetImageFromArray(T2_image.tolist())
    # print 'output name:',output_path+'/'+Folder[image_iterator]+T2[image_iterator]
    # writer.SetFileName(output_path+'/'+Folder[image_iterator]+T2[image_iterator])
    # writer.SetInput(output_FLAIR)
    # writer.Update()

    # output_T1C = itk_py_converter.GetImageFromArray(T1c_image.tolist())
    # print 'output name:',output_path+'/'+Folder[image_iterator]+T1c[image_iterator]
    # writer.SetFileName(output_path+'/'+Folder[image_iterator]+T1c[image_iterator])
    # writer.SetInput(output_FLAIR)
    # writer.Update()
    # break

