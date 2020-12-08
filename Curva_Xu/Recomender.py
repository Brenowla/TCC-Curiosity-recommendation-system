import os
import numpy as np
import pandas as pd
import math
import glob
from Optimization.util import data_from_beta_dist,integration_beta_dist
from scipy.stats import beta
import pulp as p
import statsmodels.api as sm
from operator import itemgetter

def isNaN(num):
    return num != num

def get_new_recomentations_xu(path,path_collaborative_recomendation,K,curvas,user):


    for tet in [0.1,0.5,0.9]:
        if not os.path.exists(path+"Final_Collaborative_Filtering/"+str(K)+'/'+str(tet)+"/SECM/"):
            os.makedirs(path+"Final_Collaborative_Filtering/"+str(K)+'/'+str(tet)+"/SECM/")

    df = pd.read_csv(path+'pcl_all.txt',sep='\t',comment='#')
    tracks = df.track.values
    cur = df.cur.values
        
    curiosity = {}
    for i in range(len(cur)):
        curiosity[int(tracks[i])] = float(cur[i])

    curiosity = sorted(curiosity.items(), key=itemgetter(1),reverse = True)

    df = pd.read_csv(path_collaborative_recomendation+user,sep='\t',comment='#',names = ['track','rel'])
    tracks = df.track.values
    rel = df.rel.values
    recomendations = {}
    for i in range(len(rel)):
        recomendations[int(tracks[i])] = float(rel[i])

    recomendations = sorted(recomendations.items(), key=itemgetter(1),reverse = True)

    cur = {}
    rel = {}
    tam = len(curiosity)
    for i in range(tam):
        cur[curiosity[i][0]] =  tam-i
        rel[recomendations[i][0]] = tam-i
            
    for tet in [0.1,0.5,0.9]:
        lista_rec = {}
        for track in cur.keys():
            lista_rec[track] = (1-tet)*rel[track] + tet*cur[track]
            
        lista_rec = sorted(lista_rec.items(), key=itemgetter(1),reverse = True)
        fout = open(path+"Final_Collaborative_Filtering/"+str(K)+'/'+str(tet)+"/SECM/"+user,'w+')
        for i in lista_rec:
            line = str(i[0])+'\n'
            fout.write(line)
            fout.flush()
        fout.close

			

def main(window,K,curvas,raiz,user): 
    path= raiz+'runs'+str(window)+'/'
    
    path_collaborative_recomendation = path+'CollaborativeFiltering/'
    
    get_new_recomentations_xu(path,path_collaborative_recomendation,K,curvas,user)