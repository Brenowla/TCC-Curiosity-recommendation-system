#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:56:55 2018

@author: alexandre
"""
import os
import numpy as np
import pandas as pd
from DataAnalysis.logger import Logger
import glob
#from scipy import stats as st
#from util import plot_cdf_list_curves

from multiprocessing import Process

def s_f(df,pa,pi):
    return ((np.exp(df.C*(-1)*pa)+np.exp(df.D*(-1)*pi))/2)

def s_r(df,pt):
    return ((np.exp((-1)*df.E*pt)+np.exp((-1)*df.F*pt))/2)

def diss(df, ptag, pt):
    count = 0
    if type(df.O) == int or type(df.K) == int:
        count += np.exp((-1)*ptag*(df.O))+np.exp((-1)*pt*(df.K))
    else:
        for i in range(df.K.size):
            count += np.exp((-1)*ptag*(df.O[i]))+np.exp((-1)*pt*(df.K[i]))
    if type(df.L) == int:
        aux = 2
    else:
        aux = 2*(len(df.L))
    if aux > 0:
        count = count/aux
    else:
        count = 0
    return count

def compute_fraction_time(arr, time,TIME_WINDOW):
    for i in range(arr.size):
        if time-arr[i] >= TIME_WINDOW:
            arr[i] = TIME_WINDOW
        else:
            arr[i] = (TIME_WINDOW-(time - arr[i]))/TIME_WINDOW
    return arr
        
def working_process_bits(fname,path,STORE_PATH,version,maior,menor):
    count=0
    ldata,ldata2,ldata10,ldata11,ldata8,ldata9=[],[],[],[],[],[]
    cols1=list('BCDEFGHI')
    cols1+=['J_avg','K_avg','P_avg','Sf','Sr','Dis']
    cols_corrs=['USER']
    for row in range(1,len(cols1)):
        for col in range(0,row):
            cols_corrs.append('%s&%s'%(cols1[row],cols1[col]))

    col_st=['user','avg','std','min','10%','50%','90%','max']
    for i in range(len(cols1)):
        ldata.append([])
    for i in range(len(cols1)):
        ldata2.append([])
    for i in range(len(cols_corrs)):
        ldata10.append([])
        ldata11.append([])
    for i in range(len(cols_corrs)):
        ldata8.append([])
        ldata9.append([])         
    TIME_WINDOW=999999999#24*3600
    count=records=0
    RECORDS_PERIOD=1
    pa = 0.1
    pi = 0.1
    pt = 0.01
    ptag = 0.1
    colsv = ['C','D','G','I','J_avg']
    new_colsv = ['C_n','D_n','G_n','I_n','J_avg_n']
    count+=1
    cols=['user']
    cols+=list("ABCDEFGHIJKLMNOP")
    cols.append('timestamp')
    cols.append('track')
    df = pd.read_csv(path,sep='\t',names=cols,comment='#')
    if df.shape[0] >= 5:          
        if df.P.dtype != np.float64:
            df.P = df.P.str.split(',').apply(np.asarray,args=(np.float,))
        df.K = df.K.astype(str)
        df.K = df.K.str.split(',').apply(np.asarray,args=(np.float,))
        if df.J.dtype != np.int64:
            df.J = df.J.str.split(',').apply(np.asarray,args=(np.int,))
        if df.O.dtype != np.int64:
            df.O = df.O.str.split(',').apply(np.asarray,args=(np.int,))
                
        df['J_avg']=df.J.apply(np.mean)

        for index,row in df.iterrows():
            row.K = compute_fraction_time(row.K,row.timestamp,TIME_WINDOW)

        df['J_avg']=df.J.apply(np.mean)
        df['K_avg']=df.K.apply(np.mean)
        df['O_sum']=df.O.apply(np.sum)
        
        df['K_avg']=df.K.apply(np.mean)

        df['O_avg']=df.O.apply(np.mean)

        df['P_avg']=df.P.apply(np.mean)

        df['Sf'] = df.apply(s_f, axis =1, args=(pa,pi,))
        df['Sr'] = df.apply(s_r, axis = 1, args=(pt,))
        df['Dis'] = df.apply(diss, axis = 1, args = (ptag, pt,))
                
        df.C=df.C/df.B
        df.D=df.D/df.B
        df.J_avg=df.J_avg/df.O_sum
        df.J_avg.fillna(0,inplace=True)
        df.I=df.I/df.B
        df.H=df.H/df.B

        df.loc[df.B == 0,'B']=1.
        df.loc[df.C == 0,'C']=1.
        df.loc[df.D == 0,'D']=1.
        df.loc[df.H == 0,'H']=1.
        df.loc[df.I == 0,'I']=1.
        df.loc[df.J_avg == 0,'J_avg']=1.
        """
        Changing the probabilities of metrics in bits via surprisal
        """
        df.B=np.log2(df.B)
        df.C=-np.log2(df.C)
        df.D=-np.log2(df.D)
        df.H=-np.log2(df.H)
        df.I=-np.log2(df.I)
        df.J_avg=-np.log2(df.J_avg)
                
        for i in range(5):
            if maior[fname][i]==-1:
                menor[fname][i] = df[colsv[i]].min()
                maior[fname][i] = df[colsv[i]].max()
            if maior[fname][i] == 0:
                df[new_colsv[i]]=df[colsv[i]]
            elif maior[fname][i] != menor[fname][i]:
                df[new_colsv[i]]=(df[colsv[i]]-menor[fname][i])/(maior[fname][i]-menor[fname][i])
            else:
                df[new_colsv[i]]=df[colsv[i]]/maior[fname][i]
                
        df['C_n'] = 1. - df['C_n'].values
        df['D_n'] = 1. - df['D_n'].values
        df['J_avg_n'] = 1. - df['J_avg_n'].values

        df['S_b'] = sum(df[c].values for c in colsv)/len(cols)
        df['S_n'] = sum(df[c].values for c in new_colsv)/len(new_colsv)
        df['Nov'] = (df.Sf+df.Sr+df.Dis)/3

        dt = df[['user','C','D','E','F','G','H','I','J_avg','Sf','Sr','Dis','Nov','C_n','D_n','G_n','I_n','J_avg_n','S_b','S_n','timestamp','track']].copy()
        dt[['user','C','D','E','F','G','H','I','J_avg','Sf','Sr','Dis','Nov','C_n','D_n','G_n','I_n','J_avg_n','S_b','S_n','timestamp','track']].\
            to_csv(STORE_PATH,sep = '\t', index = False, header = True, float_format = '%.2f')
    return maior, menor

def main(window,raiz,tipo,user): 
    path= raiz+'runs'+str(window)
    path_bits =  raiz+'runs'+str(window)+'/'
    STORE_PATH='da.tsv'
    
    version=1
    maior = {}
    menor = {}
    maior[user] = [-1,-1,-1,-1,-1]
    menor[user] = [-1,-1,-1,-1,-1]
    maior,menor = working_process_bits(user,path+'/pcl.txt',path_bits+STORE_PATH,version,maior,menor)

    path= raiz+'runs'+str(window)
    path_bits =  raiz+'runs'+str(window)+'/'
    STORE_PATH='da_all.tsv'

    if os.path.exists(path_bits+STORE_PATH):
        os.remove(path_bits+STORE_PATH)
    working_process_bits(user,path+'/pcl_all.txt',path_bits+STORE_PATH,version,maior,menor)

    