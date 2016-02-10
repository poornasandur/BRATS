import nibabel as nib
import numpy as np
import pandas as pd
import os
from mha import *


def npanalysis(path, predict_prefix, out_path, file_name=''):

    # path = '/media/bmi/varkey/new_n4/Recon_2013_data/N4_zscore_testing_t1_t1c_hist_match/'
    # predict_prefix = 'new_n4_ub_zc_log'
    if file_name=='':
        file_name=predict_prefix+'.csv'
    keyword = 'Masked_RawOutput'

    Folder = []
    Truth = []
    Prediction = []

    t = pd.DataFrame(columns=['Patient','Whole Tumor Dice', 'Tumor Core Dice', 'Active Tumor Dice'])

    for subdir, dirs, files in os.walk(path):
    #     if len(Folder)==1:
    #         break
        for file1 in files:
            if file1[-3:] == 'mha' and 'OT' in file1:
                Truth.append(file1)
                Folder.append(subdir+'/')
            elif file1[-3:] == 'nii' and predict_prefix in file1 and keyword in file1 and 'AT' not in file1 and 'WT' not in file1 and 'TC' not in file1:
                Prediction.append(file1)

    num_of_patients = len(Folder)
    print 'Number of Patients : ', num_of_patients
    print 'Number of Truth Images : ', len(Truth)
    print 'Number of Prediction Images : ', len(Prediction)
    for iterator in range(num_of_patients):
        print 'Iteration : ', iterator+1
        predict = nib.load(Folder[iterator]+Prediction[iterator])
        predict = predict.get_data()
        truth = new(Folder[iterator]+Truth[iterator])
        truth = truth.data
        WT_truth = np.copy(truth)
        WT_truth[WT_truth>0] = 1
        WT_predict = np.copy(predict)
        WT_predict[WT_predict>0] = 1
        WT_dice = np.sum(WT_predict[WT_truth==1])*2.0/(np.sum(WT_predict)+np.sum(WT_truth))
        del WT_truth
        del WT_predict
        
        TC_truth = np.copy(truth)
        TC_truth[TC_truth==2] = 0
        
        TC_truth[TC_truth>0] = 1
        TC_predict = np.copy(predict)
        TC_predict[TC_predict==2] = 0
        TC_predict[TC_predict>0] = 1
        TC_dice = np.sum(TC_predict[TC_truth==1])*2.0/(np.sum(TC_predict)+np.sum(TC_truth))
        del TC_truth
        del TC_predict
        
        AT_truth = np.copy(truth)
        AT_truth[AT_truth<=3] = 0
        AT_truth[AT_truth>0] = 1
        AT_predict = np.copy(predict)
        AT_predict[AT_predict<=3]=0
        AT_predict[AT_predict>0] = 1
        AT_dice = np.sum(AT_predict[AT_truth==1])*2.0/(np.sum(AT_predict)+np.sum(AT_truth))
        del AT_truth
        del AT_predict
        
        
        t.loc[len(t)] = [Folder[iterator].split('/')[-2],WT_dice,TC_dice,AT_dice]
    print t.describe()
    t.to_csv(out_path+file_name,index=False)


if __name__ == '__main__':
    path = '/media/bmi/MyPassport/n4_entire/n4_t1_t1c_hist_match_z_score/'
    predict_prefix = 'new_n4_ub_zc_log'
    out_path = '/media/bmi/varkey/new_n4/analysis/'
    file_name = 'n4_entire.csv'
    npanalysis(path, predict_prefix, out_path, file_name)



