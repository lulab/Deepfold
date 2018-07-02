#!/usr/bin/env python

import numpy as np
import os,glob
import random

def get_file_list(dir_path, extension_list):
    '''
        fuction: get_file_list(dir_path,extension_list)
        parms:
        dir_path : a string of directory full path. eg. 'user/scientist'
        extension_list : a list of file extension. eg. ['ct']
        The function returns a list of file full path.
    '''
    os.chdir(dir_path)
    file_list = []
    for extension in extension_list:
        extension = '*.' + extension
        file_list += [os.path.realpath(e) for e in glob.glob(extension) ]
    return file_list

def seq_to_mat(SeqArr):
    '''
    Convert an length-m RNA sequence in to a 4*m matrix.
    
    e.g. "ACGUAN" ->  [1 0 0 0 1 0
                       0 1 0 0 0 0
                       0 0 1 0 0 0
                       0 0 0 1 0 0]
    '''
    SeqDict = { 'A' : np.array([1,0,0,0], dtype="float32"),
                'C' : np.array([0,1,0,0], dtype="float32"),
                'G' : np.array([0,0,1,0], dtype="float32"),
                'U' : np.array([0,0,0,1], dtype="float32"),
                'T' : np.array([0,0,0,1], dtype="float32"),
                'N' : np.array([0,0,0,0], dtype="float32")}

    data = np.zeros((4, len(SeqArr)), dtype="float32")
    for i in range(len(SeqArr)):
        data[:,i] = SeqDict[SeqArr[i]]

    #Process the 4*m matrix into a 6*m matrix due to the adding of other features.
    data_more = np.zeros((6,len(SeqArr)), dtype="float32")
    data_more[0:4,:] = data
    md_num = (len(SeqArr)-1)/2
    data_more[4,md_num] = 1.0

    # Check if the neighborhoods are not empty.
    if (SeqArr[md_num+1] != 'N'):
        data_more[4, md_num+1] = 0.5
    if (SeqArr[md_num+2] != 'N'):
        data_more[4, md_num+2] = 0.25
    if (SeqArr[md_num-1] != 'N'):
        data_more[4, md_num-1] = 0.5
    if (SeqArr[md_num-2] != 'N'):
        data_more[4, md_num-2] = 0.25

    # Indicate all possible pairing patterners.
    if (SeqArr[md_num] == 'A'):
        for index in range(0, len(SeqArr)):
            if (SeqArr[index] == 'U' or SeqArr[index] == 'T'):
                data_more[5, index] = 1.0
                
    if (SeqArr[md_num] == 'C'):
        for index in range(0, len(SeqArr)):
            if (SeqArr[index] == 'G'):
                data_more[5, index] = 1.0

    if (SeqArr[md_num] == 'G'):
        for index in range(0, len(SeqArr)):
            if (SeqArr[index] == 'C' or (SeqArr[index] == 'U' or SeqArr[index] == 'T')):
                data_more[5, index] = 1.0

    if (SeqArr[md_num] == 'U' or SeqArr[md_num] == 'T'):
        for index in range(0, len(SeqArr)):
            if (SeqArr[index] == 'G' or SeqArr[index] == 'A'):
                data_more[5, index] = 1.0

    return data_more


def seq_to_mat_2D(SeqArr1, SeqArr2):

    '''
    Convert two length-m RNA sequence into a 9*m matrix.
    (1) The central nucleotide whose rank is smaller on an RNA sequence 
        will be placed above. The sequence placed below is reversed.
    (2) The 5th row indicate the possible pairing partners and there 
        neighborhoods that the network need to decide.
    '''

    if(len(SeqArr1) != len(SeqArr2)):
        print("Error in seq_to_mat_2D! Two input sequences has different lengths.")
        
    SeqDict = { 'A' : np.array([1,0,0,0], dtype="float32"),
                'C' : np.array([0,1,0,0], dtype="float32"),
                'G' : np.array([0,0,1,0], dtype="float32"),
                'U' : np.array([0,0,0,1], dtype="float32"),
                'T' : np.array([0,0,0,1], dtype="float32"),
                'N' : np.array([0,0,0,0], dtype="float32")}

    data = np.zeros((9,len(SeqArr1)), dtype="float32")
    for i in range(len(SeqArr1)):
        data[0:4,i]      = SeqDict[SeqArr1[i]]
        data[5:9,-(i+1)] = SeqDict[SeqArr2[i]]

    md_num = (len(SeqArr1)-1)/2
    data[4, md_num] = 1.0
    if ((SeqArr1[md_num+1] != 'N') and (SeqArr2[md_num-1] != 'N')):
        data[4, md_num+1] = 0.5
    if ((SeqArr1[md_num+2] != 'N') and (SeqArr2[md_num-2] != 'N')):
        data[4, md_num+2] = 0.25
    if ((SeqArr1[md_num-1] != 'N') and (SeqArr2[md_num+1] != 'N')):
        data[4, md_num-1] = 0.5
    if ((SeqArr1[md_num-2] != 'N') and (SeqArr2[md_num+2] != 'N')):
        data[4, md_num-2] = 0.25

    return data


def fill_window(j, SeqArr, Winsize):
    newSeq = ['N']*Winsize
    mi = int((Winsize-1)/2)
    Seqlen = len(SeqArr)

    if Seqlen-j>mi+1: #Exceed the right end of the window
        newSeq[mi-j:mi]=SeqArr[0:j]
        newSeq[mi:Winsize]=SeqArr[j:j+mi+1]
        newSeq[0:Seqlen-(j+mi+1)] =SeqArr[j+mi+1:Seqlen]
    else:
        if j>mi:      #Exceed the left end of the window
            newSeq[mi:mi+Seqlen-j]=SeqArr[j:Seqlen]
            newSeq[0:mi]=SeqArr[j-mi:j]
            newSeq[Winsize-(j-mi):Winsize] =SeqArr[0:j-mi]
        else:    
            newSeq[mi-j:mi+Seqlen-j]=SeqArr[0:Seqlen]

    return newSeq


def prep_data(Winsize, dir_path, RNAclass=''):
    #Prepare the training data and labels accroding to the Window size
    extension_list = ['ct']
    seqlist = get_file_list(dir_path, extension_list, RNAclass)

    Arr = []
    Data = []
    label = []
    fileInfo = []
    Pos = 0
    Neg = 0

    for file in seqlist:
        fileName = os.path.split(file)
        #print(fileName[-1])
        fh = open(file,'r')
        headline = fh.readline()
        Headline = headline.strip().split()
        Headline[0] = int(Headline[0])
        #if Headline[0] < Winsize:
        '''
        The window size is always an odd, for the purpose of symmetry.
        The length of the RNA sequence that can be loaded into the "Training Box"
        is at most (Winsize-1), because at least one empty cell should exit to re-
        present the differences between the beginning nucleotide and the end. 
        '''
        SeqArr = []
        for line in fh.readlines():
            Arr=line.strip().split()
            SeqArr.append(Arr[1])
            Arr[4]=int(Arr[4])
            if(Arr[4]!=0):
                Pos += 1
                label.append(1)
            else:
                Neg += 1
                label.append(0)

            fileInfo.append([fileName[-1],Arr[0],Arr[1]])
            #Finish the labeling of training samples
        Data.append(SeqArr)
        
   # num = max(Pos,Neg)
    num = Pos+Neg
    Pos_Arr = np.empty((Pos,4,Winsize), dtype="float32")
    Neg_Arr = np.empty((Neg,4,Winsize), dtype="float32")
    training_data_new = np.empty((num,4,Winsize), dtype="float32")
    n_pos=0
    n_neg=0
    k=0 # k is the index of array "label"

    for i in range(len(Data)):
        for j in range(len(Data[i])):
	    if (len(Data[i])<Winsize):
                Arr = fill_window(j, Data[i], Winsize)
	    else:
                Arr = fill_window_small(j, Data[i], Winsize)

	    training_data_new[k] = seq_to_mat(Arr)

            if label[k]==0:
                Neg_Arr[n_neg] = seq_to_mat(Arr)
                n_neg += 1
            elif label[k]==1:
		'''
		print("k:",k)
		print("j:",j)
		print("n_pos:",n_pos)
		'''
                Pos_Arr[n_pos] = seq_to_mat(Arr)
                n_pos += 1
            k += 1

    np.random.shuffle(Neg_Arr)
    np.random.shuffle(Pos_Arr)    
    training_data = np.empty((num,4,Winsize), dtype="float32")
    training_label= np.empty((num,1), dtype="int8")
    n_pos=0
    n_neg=0

    for i in range(Pos):
        training_data[i] = Pos_Arr[i]
        training_label[i] = 1

    for i in range(Neg):
        training_data[i+Pos] = Neg_Arr[i]
        training_label[i+Pos] = 0

    '''
    ran_list=random.sample(range(num*2), num)
    for i in ran_list:
        training_data[i] = Pos_Arr[n_pos]
        training_label[i]= 1
        n_pos += 1
    
    All = [i for i in range(num*2)]
    ran_list=list(set(All).difference(set(ran_list)))
    for i in ran_list:
        training_data[i] = Neg_Arr[n_neg]
        training_label[i]= 0
        n_neg += 1
	if (n_neg==len(Neg_Arr)):
		n_neg = 0
    '''

    #return training_data, training_label
    return training_data_new, label, fileInfo
