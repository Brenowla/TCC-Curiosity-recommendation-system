import numpy as np
import pandas as pd
import util as ut
from matplotlib import pyplot as plt
import os

def main(path)
    K = [5,10,20,50]

    if not os.path.exists(path+'Performance_Zhao/'):
        os.makedirs(path+'Performance_Zhao/')

    if not os.path.exists(path+'Performance_Alexandr/'):
        os.makedirs(path+'Performance_Alexandr/')
    
    fout = open(path+'Performance_Alexandr/'+'performance.txt','w+')
    for i in range(1):
        if i != 4:
            fin = open(path+'runs'+str(i)+'/Performance_Alexandre/performance.txt','r')
            for line in fin:
                fout.write(line)
                fout.flush()

    fout = open(path+'Performance_Zha/'+'performance_zhao.txt','w+')
    for i in range(1):
        if i != 4:
            fin = open(path+'runs'+str(i)+'/Performance/performance.txt','r')
            for line in fin:
                fout.write(line)
                fout.flush()
    
    fout = open(path+'komogorov_test_Zhao.txt','w+')
    for i in range(9):
        if i != 4:
            fin = open(path+'runs'+str(i)+'/Curva_Zha/komogorov_test.txt','r')
            for line in fin:
                fout.write(line)
                fout.flush()
    '''
    fout = open(path+'komogorov_test_Alexandre.txt','w+')
    for i in range(9):
        if i != 4:
            fin = open(path+'runs'+str(i)+/Curva_Alexandr/komogorov_test.txt','r')
            for line in fin:
                fout.write(line)
                fout.flush()
    '''
    for i in range(9):
        if i != 4:
            fin = open(path+'runs'+str(i)+'/Curva_Zha/komogorov_test.txt','r')
            for line in fin:
                fout.write(line)
                fout.flush()

    fout = open(path+'p_values.txt','w+')
    for i in range(9):
        if i != 4:
            fin = open(path+'runs'+str(i)+'/Curva_Zha/p_values.txt','r')
            for line in fin:
                fout.write(line)
                fout.flush()


    cols_EF = ['EF_collaborative','EF_collaborative_cur','EF_matriz','EF_matriz_cur','EF_popularity','EF_popularity_cur']
    cols_P = ['P_collaborative','P_collaborative_cur','P_matriz','P_matriz_cur','P_popularity','P_popularity_cur']
    cols_IS = ['IS_collaborative','IS_collaborative_cur','IS_matriz','IS_matriz_cur','IS_popularity','IS_popularity_cur']
    cols = cols_EF+cols_P+cols_IS
    sty = ['b-','b--','g-','g--','r-','r--']
    df = pd.read_csv(path+'Performance_Zha/'+'performance_zhao.txt',sep='\t', names = cols+['K','tet'])


    for k in K:
        x = {}
        for c in cols:
            x[c] = []
        for t in range(11):
            for c in cols:
                x[c].append(df.loc[df.K == 50].loc[df.tet == t/10,c].mean())
        if not os.path.exists(path+'Performance_Zha/'+str(k)+/"):
            os.makedirs(path+'Performance_Zha/'+str(k)+/")
        #Estimulation Fitness
        plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})   
        xticks=np.arange(0,1,0.1)
        yticks=np.arange(0,1.1,0.1)
        plt.title('Estimulation Fitness, small better')
        plt.grid(True,axis='both',linestyle=':')
        plt.xticks(xticks)
        plt.xlim(0,1)
        plt.xlabel('\u03B8')
        plt.ylabel('Estimulation Fitness')
        for c,s in zip(cols_EF,sty):
            plt.plot(yticks,x[c],s,lw=1.5,label = c)
        legend = plt.legend(loc='best', fontsize='x-small', framealpha = 0.5 )
        legend.get_frame().set_facecolor('white')
        plt.draw()
        plt.savefig(path+'Performance_Zha/'+str(k)+/Estimulation_Fitness.png", format='png')
        plt.close()

        #Precision
        plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})   
        xticks=np.arange(0,1,0.1)
        yticks=np.arange(0,1.1,0.1)
        plt.title('Precision, large better')
        plt.grid(True,axis='both',linestyle=':')
        plt.xticks(xticks)
        plt.xlim(0,1)
        #plt.ylim(bottom = 0)
        plt.xlabel('\u03B8')
        plt.ylabel('Precision')
        for c,s in zip(cols_P,sty):
            plt.plot(yticks,x[c],s,lw=1.5,label = c)
        legend = plt.legend(loc='best', fontsize='x-small', framealpha = 0.5 )
        legend.get_frame().set_facecolor('white')
        plt.draw()
        plt.savefig(path+'Performance_Zha/'+str(k)+/Precision.png", format='png')
        plt.close()

        #Inter user similarity
        plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})   
        xticks=np.arange(0,1,0.1)
        yticks=np.arange(0,1.1,0.1)
        plt.title('Inter User Similarity, small better')
        plt.grid(True,axis='both',linestyle=':')
        plt.xticks(xticks)
        plt.xlim(0,1)
        #plt.ylim(bottom = 0)
        plt.xlabel('\u03B8')
        plt.ylabel('Inter user similarity')
        for c,s in zip(cols_IS,sty):
            plt.plot(yticks,x[c],s,lw=1.5,label = c)
        legend = plt.legend(loc='best', fontsize='x-small', framealpha = 0.5 )
        legend.get_frame().set_facecolor('white')
        plt.draw()
        plt.savefig(path+'Performance_Zha/'+str(k)+/Inter_User_Similarity.png", format='png')
        plt.close()
    
    df = pd.read_csv(path+'komogorov_test_Zhao.txt',sep='\t', names = ['K_t'])
    fout = open(path+'Performance_Zha/komogorov_result.txt','w+')
    fout.write(str(df.K_t.mean()))
    fout.flush()
    '''
    df = pd.read_csv(path+'komogorov_test_Alexandre.txt',sep='\t', names = ['K_t'])
    fout = open(path+'Performance_Alexandr/komogorov_result.txt','w+')
    print(df.loc[df.K_t == 1,'K_t'].size/df.K_t.size)
    print(df.loc[df.K_t == 2,'K_t'].size/df.K_t.size)
    print(df.loc[df.K_t == 3,'K_t'].size/df.K_t.size)
    print(df.loc[df.K_t == 0,'K_t'].size/df.K_t.size)
    '''
    #fout.flush()

    fin = open(path+'p_values.txt','r')
    pvalues = []
    for line in fin:
        pvalues.append(float(line[:-1]))
    ut.CDF(pvalues,"P_values")
    fout = open(path+'Performance_Zha/p_value_result.txt','w+')
    fout.write(str(df.K_t.mean()))
    fout.flush()
    
    cols = cols_P+cols_IS
    df = pd.read_csv(path+'Performance_Alexandr/performance.txt',sep='\t', names = cols+['K','tet'])
    for k in K:
        x = {}
        for c in cols:
            x[c] = []
        for t in range(11):
            for c in cols:
                x[c].append(df.loc[df.K == 50].loc[df.tet == t/10,c].mean())
        if not os.path.exists(path+'Performance_Alexandr/'+str(k)+/"):
            os.makedirs(path+'Performance_Alexandr/'+str(k)+/")
        
        #Estimulation Fitness
        '''
        plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})   
        xticks=np.arange(0,1,0.1)
        yticks=np.arange(0,1.1,0.1)
        plt.title('Estimulation Fitness, small better')
        plt.grid(True,axis='both',linestyle=':')
        plt.xticks(xticks)
        plt.xlim(0,1)
        plt.xlabel('\u03B8')
        plt.ylabel('Estimulation Fitness')
        for c,s in zip(cols_EF,sty):
            plt.plot(yticks,x[c],s,lw=1.5,label = c)
        legend = plt.legend(loc='best', fontsize='x-small', framealpha = 0.5 )
        legend.get_frame().set_facecolor('white')
        plt.draw()
        plt.savefig(path+'Performance_Alexandr/'+str(k)+/Estimulation_Fitness.png", format='png')
        plt.close()
        '''
        #Precision
        plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})   
        xticks=np.arange(0,1,0.1)
        yticks=np.arange(0,1.1,0.1)
        plt.title('Precision, large better')
        plt.grid(True,axis='both',linestyle=':')
        plt.xticks(xticks)
        plt.xlim(0,1)
        #plt.ylim(bottom = 0)
        plt.xlabel('\u03B8')
        plt.ylabel('Precision')
        for c,s in zip(cols_P,sty):
            plt.plot(yticks,x[c],s,lw=1.5,label = c)
        legend = plt.legend(loc='best', fontsize='x-small', framealpha = 0.5 )
        legend.get_frame().set_facecolor('white')
        plt.draw()
        plt.savefig(path+'Performance_Alexandr/'+str(k)+/Precision.png", format='png')
        plt.close()

        #Inter user similarity
        plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})   
        xticks=np.arange(0,1,0.1)
        yticks=np.arange(0,1.1,0.1)
        plt.title('Inter User Similarity, small better')
        plt.grid(True,axis='both',linestyle=':')
        plt.xticks(xticks)
        plt.xlim(0,1)
        #plt.ylim(bottom = 0)
        plt.xlabel('\u03B8')
        plt.ylabel('Inter user similarity')
        for c,s in zip(cols_IS,sty):
            plt.plot(yticks,x[c],s,lw=1.5,label = c)
        legend = plt.legend(loc='best', fontsize='x-small', framealpha = 0.5 )
        legend.get_frame().set_facecolor('white')
        plt.draw()
        plt.savefig(path+'Performance_Alexandr/'+str(k)+/Inter_User_Similarity.png", format='png')
        plt.close()