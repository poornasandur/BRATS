
import os
import subprocess
new_prefixs=['new_n4_axis2dropout_xyz','new_n4_axis3dropout_xyz']

roots= ['/media/bmi/varkey/new_n4/Recon_2013_data/log_training','/media/bmi/varkey/new_n4/Recon_2013_data/log_validation']


name_of_text_files=['log_training_axis_2.txt','log_training_axis_3.txt','log_validation_axis_2.txt','log_validation_axis_3.txt']
counter=0
for i in range(2):
	root=roots[i]
	for j in range(2): 
		new_prefix=new_prefixs[j]
		txt_file=name_of_text_files[counter]
		counter=counter+1
		print 'root:',root
		print 'new_prefix',new_prefix
		print 'txt_file:',txt_file
		callString = 'python analysis.py > ' + txt_file
		print callString
		subprocess.call(callString, shell = True)
