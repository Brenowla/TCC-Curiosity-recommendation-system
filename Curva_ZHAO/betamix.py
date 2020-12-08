#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 15:48:52 2018

@author: alexandre
"""
import time
import numpy as np
import scipy as sc
import pandas as pd
from scipy.stats import beta # Beta distribution
from scipy.stats import ks_2samp
from matplotlib import pyplot as plt
#from scipy.special.cython_special import beta as Beta # Beta function that use gamma function
from statsmodels.distributions.empirical_distribution import ECDF

def D_weight_square(x,Y):
    return np.min(np.abs(x-Y))**2

def beta_distribution_estimation(values):
    avg=np.mean(values)
    std=np.std(values)
    righ_side=(((avg*(1-avg))/std**2)-1)
    a=avg*righ_side
    b=(1-avg)*righ_side
    return a,b

def getPointsY(C, values):
    X=values.copy()
    y=np.random.choice(X,1)
    Y=np.array(y)
    X=X[np.logical_not(X == y)]
    
    while Y.size != C:
        p=np.array([])
        for x in X:
            p=np.insert(p,p.size,D_weight_square(x,Y))
        p=p/p.sum()
        y=np.random.choice(X,1,p=p)
        Y=np.insert(Y,Y.size,y)
        X=X[np.logical_not(X == y)]
    return Y

def plot_U_shaped(values,C,A,B,Pi,path_save_fig):
    values.sort()
    x_values=values.copy()
    plt.xlabel('Stimulus Degree')
    plt.ylabel('Distribution')
    xticks=np.arange(0,1,0.1)
    yticks=np.arange(0,10.1,1)
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlim(0,1)
    plt.ylim(0,10.1)
    plt.grid(True,axis='both',linestyle=':')
    yvalues = sum(Pi[c]*beta.pdf(x_values,A[c],B[c]) for c in range(C))
    plt.plot(x_values,yvalues,'k--',lw=1.5)
    plt.hist(x_values,bins=50,density=True,alpha=0.25,facecolor='green')
    plt.draw()
    plt.savefig(path_save_fig,bbox_inches='tight')
    #plt.show()
    plt.clf()

def data_from_beta_dist(a,b):
    mod=(a-1)/(a+b-2)
    k=np.sqrt(((a-1)*(b-1))/(a+b-3))/(a+b-2)
    sd_max=beta.pdf(mod,a,b)
    sd_inf=-1
    sd_sup=-1
    if a > 1 and a < 2 and b > 2:
        sd_sup=beta.pdf(mod+k,a,b)
    if b > 1 and b < 2 and a > 2:
        sd_inf=beta.pdf(mod-k,a,b)
    if a > 2 and b > 2:
        sd_sup=beta.pdf(mod+k,a,b)
        sd_inf=beta.pdf(mod-k,a,b)
    return mod, k, sd_max, sd_inf, sd_sup

def init(C, values):
    np.random.seed(np.int(time.time()))
    #Y=getPointsY(C,values)
    lower=values.min()
    upper=values.max()
    Y=np.linspace(lower,upper,C+2)
    #Y=Y[:-2]
    Y=Y[1:Y.size-1]
    #Y.sort()
    
    SLACK=(values.max()-values.min())/2.
    #print(SLACK)
    A=np.array([])
    B=np.array([])
    Pi=np.array([])
    for y in Y:
        if y-SLACK < 0.:
            lower_bound=0
        else:
            lower_bound=y-SLACK
        if y+SLACK > 1.:
            upper_bound=1.
        else:
            upper_bound=y+SLACK
        v=values[np.logical_and(values>=lower_bound,values<=upper_bound)]
        a,b=beta_distribution_estimation(v)
        A=np.insert(A,A.size,a)
        B=np.insert(B,B.size,b)
        Pi=np.insert(Pi,Pi.size,v.size)
    
    Pi=Pi/Pi.sum()

    return A,B,Pi    

def estimate(A,B,Pi,values,C,niter=1000,precision=1E-5):
    maximum=1E-5
    k=10**5
    count=0
    W=np.zeros((values.size,C))
    A_new=np.zeros(A.shape)
    B_new=np.zeros(A.shape)
    Pi_new=np.zeros(A.shape)
    mean=np.zeros(A.shape)
    variance=np.zeros(A.shape)
    while k > precision and count < niter:
        count+=1
        W[:,:]=0
        for j in range(C):
            #denom=Beta(A[j],B[j])
            #assert denom!=0,(A[j],B[j])
            W[:,j]=Pi[j]*beta.pdf(values,A[j],B[j])#((values**(A[j]-1)*(1-values)**(B[j]-1))/denom)
        assert np.isfinite(W.any())
        S=np.sum(W,1).reshape((values.size,1))
        W=W/S
        SLACK=(values.max()-values.min())/2
        
        wfirst=np.array([1.]+(C-1)*[0])
        wlast=np.array((C-1)*[0]+[1.])
        bad=(~np.isfinite(W)).any(axis=1)
        badfirst = np.logical_and(bad, values < SLACK)
        badlast = np.logical_and(bad, values >= SLACK)
        W[badfirst,:] = wfirst
        W[badlast,:] = wlast
        assert np.isfinite(W.any())
        
        Pi_new[:]=0.
        for j in range(C):
            Pi_new[j]=W[:,j].sum()/values.size    
        
        mean[:]=0.
        variance[:]=0.
        A_new[:]=0.
        B_new[:]=0.
        
        for j in range(C):
            denom=values.size*Pi_new[j]
            mean[j]=(W[:,j].dot(values))/denom
            variance[j]=(W[:,j].dot((values-mean[j])**2))/denom
            if np.isnan(mean[j]) or np.isnan(variance[j]):
                mean[j]=0.5; variance[j]=1/12.
                A_new[j]=1.
                B_new[j]=1.
                assert denom == 0
            else:
                assert np.isfinite(mean[j]) and np.isfinite(variance[j]), (j,mean[j],variance[j],denom)
                pi=(mean[j]*(1-mean[j]))/variance[j] - 1
                A_new[j]=mean[j]*pi
                B_new[j]=(1-mean[j])*pi   
        
        maximum=max(abs(A.max()),abs(A_new.max()),abs(B.max()),\
                    abs(B_new.max()),abs(Pi.max()),abs(Pi_new.max()))
        if maximum != 0:
            a=np.abs(A_new-A)/maximum
            b=np.abs(B_new-B)/maximum
            p=np.abs(Pi_new-Pi)/maximum
            k=max(a.max(),b.max(),p.max())
        else:
            k=0

        A=A_new.copy()
        B=B_new.copy()
        Pi=Pi_new.copy()

    return A,B,Pi,k,count,W

def test(values,A,B,Pi,C):
    Fn=ECDF(values)
    F=Fn(values)
    F0=np.zeros(F.shape)
    beta_mix=np.array([])
    for j in range(C):
        F0+=Pi[j]*beta.cdf(values,A[j],B[j])
        beta_mix=np.concatenate((beta_mix,beta.rvs(A[j],B[j],\
                                 size=np.int(round(values.size*Pi[j])))))
    n=values.size
    ks2=1.63*np.sqrt((2*n)/n**2)
    ks1=1.63/np.sqrt(n)
    p=0.01
    dmax=np.abs(F0-F).max()
    res=ks_2samp(beta_mix,values)
    d,pvalue=res[0],res[1]
    return n,ks1,ks2,dmax,d,p,pvalue

def amplitude(x,a,b,values):
    max_value=np.max(values)
    min_value=np.min(values)
    k = np.sqrt(((a-1)*(b-1))/(a+b-3))/(a+b-2)
    s = np.sqrt((a*b)/(((a+b)**2)*(a+b+1)))
    x_left=0.
    x_right=0.
    if a > 2. and b > 2.:  # Bell-shaped form
        x_left= x - k
        x_right= x + k
    elif a == 2. and b > 2.: # Right-tailed
        x_left = x - (x - min_value)/2.
        x_right = 2/b
    elif (a > 1) and (a < 2) and (b > 2): # Right-tailed Unimodal
        x_left = x_left = x - (x - min_value)/2.
        x_right = x_right= x + k
    elif (a > 2) and (b == 2):  # Left-tailed
        x_left = x_left = 1 - 2/a
        x_right = x_right= x + (max_value - x)/2.
    elif (a > 2.) and (b > 1) and (b < 2):   # Left-tailed Unimodal     
        x_left = x_left = x - k
        x_right = x_right= x + (max_value - x)/2.
    elif (a > 1) and (a < 2) and (b > 1) and (b < 2): # Inverse U-Shaped form
        x_left  = x_left = x - (x - min_value)/2.
        x_right = x_right= x + (max_value - x)/2.
        
    return x_left,x_right

def curve_profile(A,B,Pi,C,values):
    C_score=[]
    SD_score=[]
    Mode=(A - 1)/(A + B -2)
    for j in range(C):
        a=A[j]
        b=B[j]
        p=Pi[j]
        x=Mode[j]
        x_left, x_right = amplitude(x,a,b,values)
        c_score = p * beta.pdf(x,a,b)
        c_score_left = p * beta.pdf(x_left,a,b)
        c_score_right = p * beta.pdf(x_right,a,b)
        C_score.append([c_score_left, c_score, c_score_right])
        SD_score.append([x_left, x, x_right])
    
    C_score=np.array(C_score)
    SD_score=np.array(SD_score)
    return C_score,SD_score

def read_data(path,fname):
    cols=list("ABCDEFGHIJKLMNOP")
    df = pd.read_csv(path+str(fname)+'.txt',sep='\t',names=cols,comment='#')    
                     
    if df.K.dtype != np.float64:
        df.K = df.K.str.split(',').apply(np.asarray,args=(np.float,))
    if df.J.dtype != np.int64:
        df.J = df.J.str.split(',').apply(np.asarray,args=(np.int,))
    if df.O.dtype != np.int64:
        df.O = df.O.str.split(',').apply(np.asarray,args=(np.int,))
        
    df['J_avg']=df.J.apply(np.mean)
    df['K_avg']=df.K.apply(np.mean)
    df['O_sum']=df.O.apply(np.sum)    
    """
    Changing the metrics for probabilities
    """
    df.C=df.C/df.B
    df.D=df.D/df.B
    df.J_avg=df.J_avg/df.O_sum
    df.J_avg.fillna(0,inplace=True)
    df.I=df.I/20.#df.B
    df.H=df.H/20.#df.B
    
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
    
    df.loc[df.B == 0,'B']=0.
    df.loc[df.C == 0,'C']=0.
    df.loc[df.D == 0,'D']=0.
    df.loc[df.H == 0,'H']=0.
    df.loc[df.I == 0,'I']=0.
    df.loc[df.J_avg == 0,'J_avg']=0.
    
    df.J=df.J_avg 
    cols=list('CDGIJ')
      
    for key in cols:
        df[key]=(df[key]-df[key].min())/(df[key].max()-df[key].min())
    """
    The novelty is inversely proportional to reduntancy, frequency and dissimilarity:
    """
    df.C = 1. - df.C.values
    df.D = 1. - df.D.values
    df.J = 1. - df.J.values
    
    df['S']=(df.C.values+df.D.values+df.J.values+df.G.values+df.I.values)/5.
    return df['S']

# Input a pandas series 
def data_entropy(data, ):
    p_data= data.value_counts()/len(data) # calculates the probabilities
    entropy=sc.stats.entropy(p_data)  # input probabilities to get the entropy 
    return entropy

    
if __name__ == '__main__':
    #9142215     32804681    42603266
    # 46873279  7760751     6367988     31795803    30339296    9966267  
    # 45869907  12287978    18849887
    path='/home/alexandre/Documentos/Curiosity/window1/stimulus-filter-new/'
    #path='/media/alexandre/Dell USB Portable HDD/BKP-Optiplex Micro 12-06-18/Documentos/Curiosity/window1/stimulus/2110860.txt'
    #path='/home/alexandre/Dropbox/Researchs Options/Analysis/window1/34668056.txt'
    fname=5036203
    df=read_data(path,fname)
    #df=pd.read_csv(path,sep='\t',comments='#')
    #for i in range(1):
    i=0
    for key in []:#df.keys():
        np.random.seed(np.int(time.time()))#np.random.randint(df.shape[0]
        C=1
        values=df[key].values
        #values=df['A'].values
        #values=(df['D'].values+df['SF'].values+df['SR'].values+df['Hnorm'].values)/4.
        #values=df['SD'].values
        
        A,B,Pi=init(C,values)
        #"""
        print('\nInitialization: %s\n'%key)
        print('\t\tPi:',Pi)
        print()
        print('\t\tAlpha:',A)
        print()
        print('\t\tBeta:',B)
        print()
        #"""
        
        niter=1000
        precision=1E-6
        
        begin=time.time()
        A,B,Pi,k,count,W=estimate(A,B,Pi,values,C,niter=niter,precision=precision)
        end=time.time()
        runtime=end-begin
        n,ks1,ks2,dmax,d,p,pvalue=test(values,A,B,Pi,C)
        C_score,SD_score=curve_profile(A,B,Pi,C,values)
                
        print('\nResults of experiment:\n')
        print('\t\tPi:',Pi)
        print()
        print('\t\tAlpha:',A)
        print()
        print('\t\tBeta:',B)
        print()
        print('\t\tNumber of iterations: %d'%count)
        print('\t\tPrecision: %.12f\n'%(k))
        print('\t\tData Entropy: %.12f\n'%(data_entropy(pd.Series(values))))
        
        print("\t\tDmax: %.6f, KS-value: %.6f, p-value: %.3f" %(dmax,ks1,p))
        print("\t\tD   : %.6f, KS-value: %.6f, p-value: %.3f" %(d,ks2,pvalue))
        
        print("\n\t\tCuriosity Score:\n")
        for j in range(C):
            print("\t\t",C_score[j])
        print("\n\t\tStimulus Degree:\n")
        for j in range(C): 
            print("\t\t",SD_score[j])
        print("\nIteration:[ %d ] \tRuntime: %.9f secs...\n"%(i,runtime))
    
        """
        Defining the clusters according to posterior probability given by W
        """
        cl=W.argmax(axis=1)
        i=i+1