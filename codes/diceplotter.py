# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:17:41 2015

@author: kiran
"""

import re
import numpy
# from segment import new_prefix, testingImages

def calculateDice(filename, testingImages):
	
	# filename = new_prefix + '.txt'

	print 'Filename: ', filename

	numberOfClasses = 1
	jump = numberOfClasses + 5

	with open(filename) as f:
	    contents = f.readlines()

	endline = testingImages * jump

	mean, std = performCalc(2, endline, jump, contents)

	print

	print 'WT - Dice: ', mean
	print 'WT - STD : ', std

	mean, std = performCalc(endline+2, 2*endline, jump, contents)

	print

	print 'TC - Dice: ', mean
	print 'TC - STD : ', std

	mean, std = performCalc(2*endline+2, 3*endline, jump, contents)

	print

	print 'AT - Dice: ', mean
	print 'AT - STD : ', std

def performCalc(startline, stopline, jump, contents):

	dice = []
	falsePositives = []
	falseNegatives = []

	for i in range(startline, stopline, jump):
		x = re.split(' |\n',contents[i])
		y = [float(k) for k in x if k!='']
		dice.append(y[2])
		falsePositives.append(y[4])
		falseNegatives.append(y[5])

	# print 'AT - Dice: ', numpy.mean(dice)
	# print 'AT - STD : ', numpy.std(dice)
	return numpy.mean(dice), numpy.std(dice)


if __name__ == '__main__':
	calculateDice('new_n4_mse_WT2_log.txt',35)
    
    
    
