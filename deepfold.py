#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
import sys
import numpy as np
import scipy as sp
import h5py
import os,glob
import argparse

parser = argparse.ArgumentParser()
if len(sys.argv) < 2:
    parser.print_usage()
    sys.exit(1)

group = parser.add_mutually_exclusive_group()
group.add_argument("-S", "--subset", action="store_true",help=" Use subset data that sequence similarity < 60 percentage as reference data" )
group.add_argument("-F", "--fullset", action="store_true", help="Use fullset data as reference data") 

parser.add_argument('input', help='Input directory, RNA ct file')
parser.add_argument('output', help='Ouput directory')
args = parser.parse_args()
if not(args.subset or args.fullset):
    parser.print_usage()
    sys.exit(1)


import keras
from keras.models import Sequential, Model, model_from_json
from keras.utils import np_utils, generic_utils
from six.moves import range
from sklearn.externals import joblib
from Functions_data import seq_to_mat, seq_to_mat_2D, fill_window, prep_data
Winsize = 801


#----------------

def DeepFold_predict(SeqArr, model_1, path_prefix, current_file):
    
    print ("Predict 1D prob:")
    seqlen = len(SeqArr)
    Test_1 = np.zeros((seqlen, 6 ,Winsize), dtype="float32")
    for j in range(seqlen):
        Arr = fill_window(j, SeqArr, Winsize)
        Test_1[j,:,:] = seq_to_mat(Arr)

    Data = Test_1 # Test_1 need to be reshaped, so just copy it for further usage.
    Test_1 = Test_1.reshape(Test_1.shape[0], Test_1.shape[1], Test_1.shape[2], 1)
    proba_1 = model_1.predict(Test_1, verbose=0)

    Thr1 = 0.25
    Pair = []     # Nucleotides that are paired 
    for j in range(seqlen):
        if (proba_1[j,1] > Thr1):
            Pair.append(j)
    
    Pattern = []
    pairnum = len(Pair)
    total = int((1+pairnum-1)*(pairnum-1)/2)
    Test = np.zeros((total,9,Winsize), dtype="float32")
    
    k=0
    for i in range(pairnum-1):
        for j in range(i+1,pairnum):
            if(   (SeqArr[Pair[i]]=="A" and (SeqArr[Pair[j]]=="U" or SeqArr[Pair[j]]=="T"))
               or (SeqArr[Pair[i]]=="C" and SeqArr[Pair[j]]=="G")
               or (SeqArr[Pair[i]]=="G" and SeqArr[Pair[j]]=="C")
               or (SeqArr[Pair[i]]=="G" and SeqArr[Pair[j]]=="U")
               or (SeqArr[Pair[i]]=="U" and SeqArr[Pair[j]]=="G")
               or ((SeqArr[Pair[j]]=="U" or SeqArr[Pair[j]]=="T") and SeqArr[Pair[j]]=="A") ):
                sequence1 = fill_window(Pair[i], SeqArr, Winsize)
                sequence2 = fill_window(Pair[j], SeqArr, Winsize)
                Test[k] = seq_to_mat_2D(sequence1, sequence2)
                Pattern.append([Pair[i], Pair[j]])
                k += 1

    print ("Predict 2D prob:")
    Test_2 = np.zeros((k,9,Winsize), dtype="float32")
    Test_2 = Test[0:k]

    Test_2 = Test_2.reshape(Test_2.shape[0], Test_2.shape[1], Test_2.shape[2], 1)

    model_list = get_file_list(path2, ['h5'] )
    model_num = len(model_list)
    model_c = model_from_json(open(path2 + "0_DeepFold_2D_architecture.json").read())
    for n in range(model_num):
	model_c.load_weights(model_list[n])
        proba_c = model_c.predict(Test_2, verbose=0)
        if (n == 0):
            proba_2 = proba_c
        if (n > 0):
            proba_2 = proba_2 + proba_c
    proba_2 = proba_2/model_num

    
    print ("Create final:")
    Thr2 = []
    for i in range(50):
        Thr2.append(0.99-i*0.01)
    Final={}

    for K in range(len(Thr2)):
        for i in range(k):
            if (proba_2[i,1] > Thr2[K] and ((Pattern[i][0] in Final)==False)):
                Final[Pattern[i][0]]=Pattern[i][1]
                Final[Pattern[i][1]]=Pattern[i][0]
                if (Pattern[i][1] in Final):
                    if  (Final[Pattern[i][1]]==Pattern[i][0]+1):
                        Final[Pattern[i][0]]=Pattern[i][1]+1
                        Final[Pattern[i][1]+1]=Pattern[i][0]
                    elif(Final[Pattern[i][1]]==Pattern[i][0]-1):
                        Final[Pattern[i][0]]=Pattern[i][1]-1
                        Final[Pattern[i][1]-1]=Pattern[i][0]
                else:
                    Final[Pattern[i][0]]=Pattern[i][1]
                    Final[Pattern[i][1]]=Pattern[i][0]

    for j in range(seqlen):
        if ((j in Final) and (j-1 in Final) and (j+1 in Final)):
            if(abs(Final[j-1]-Final[j+1])==2 and Final[j] != int((Final[j-1]+Final[j+1])/2)):
                Final[j]=int((Final[j-1]+Final[j+1])/2)

    print("Create ct file: ") 
    fh = open(path_prefix+current_file,"w")
    fh.write(str(seqlen)+"\t"+current_file+"\n")
    for j in range(seqlen):
        if j in Final:
            fh.write(str(j+1)+" "+SeqArr[j]+"\t"+str(j)+"\t"+str(j+2)+"\t"+str(Final[j]+1)+"\t"+str(j+1)+"\n")
        else:    
            fh.write(str(j+1)+" "+SeqArr[j]+"\t"+str(j)+"\t"+str(j+2)+"\t"+"0"+"\t"+str(j+1)+"\n")
    fh.close()

#----------------

def get_file_list(dir_path, extension_list):
    '''
        fuction: get_file_list(dir_path,extension_list)
        parms:
        dir_path : a string of directory full path. eg. 'user/scientist'
        extension_list : a list of file extension. eg. ['ct']
        The function returns a list of file full path.
        '''
    '''
    os.chdir(dir_path)
    file_list = []
    for extension in extension_list:
        extension = '*.' + extension
        file_list += [os.path.realpath(e) for e in glob.glob(extension)]
    '''
    file_list = []
    for extension in extension_list:
        file_list += [dir_path+f for f in os.listdir(dir_path) if f.endswith(extension)]
    return file_list


#################
#     Main      #
#################

#path_prefix = "/Share/home/huboqin/project/deepfold_all_data/final/result/"
path_prefix= args.output+"/"
if not os.path.exists(path_prefix):
    os.mkdir( path_prefix, 0755 );

# Load 1D and 2D DeepFold model 
if (args.subset):
    path2= "./model/2D_S/"
    model_1 = model_from_json(open( "./model/1D_S/DeepFold_1D_architecture.json").read())
    model_1.load_weights("./model/1D_S/DeepFold_1D_weight.h5")
if (args.fullset):
    path2= "./model/2D_F/"
    model_1 = model_from_json(open( "./model/1D_F/DeepFold_1D_architecture.json").read())
    model_1.load_weights("./model/1D_F/DeepFold_1D_weight.h5")

#dir_path="./test/"
dir_path = args.input+"/"
extension_list = ['ct']
seqlist = get_file_list(dir_path, extension_list)



for file in seqlist:
    fh = open(file,'r')
    headline = fh.readline()
    Headline = headline.strip().split()
    filename = str(file).split("/") 
    print("Read file: "+ filename[-1])
    SeqArr = []
    for line in fh.readlines():
        Arr=line.strip().split()
        SeqArr.append(Arr[1])

    DeepFold_predict(SeqArr, model_1, path_prefix, filename[-1])
