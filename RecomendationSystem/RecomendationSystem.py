# Pacotes para análise dos dados
import pandas as pd
import numpy as np
import os

#import CollaborativeFiltering as cf
from RecomendationSystem.MatrixFactorization import MF
import RecomendationSystem.Calculate_Rates as cr
import RecomendationSystem.CollaborativeFiltering as cf
import glob

# For our 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from random import randint
import random
from operator import itemgetter
from sklearn.decomposition import NMF

def add_dict(d,key):
    if key in d.keys():
        d[key]+=1
    else:
        d[key]=1

def add_dict_musicas(d,key,info):
    d[key]=info

def main(window,raiz): 
    # set output to three decimals
    pd.set_option('display.float_format',lambda x: '%.5f' %x)

    common_users = 20

    path=raiz

    fin=open(path+'LFM-1b_artists_genres.txt','r')
    artist_genre={}
    for line in fin:
        row=line.split('\t')
        artist=int(row[0])
        genre=int(row[1][:-1])
        if not artist in artist_genre.keys():
            artist_genre[artist]=set([genre])
        else:
            artist_genre[artist].add(genre)
    fin.close()

    fin=open(path+'runs'+str(window)+'/tracks_artists/tracks_artists.txt','r')
    dic_tracks={}
    for line in fin:
        row=line.split('\t')
        track=int(row[0])
        artist=int(row[1])
        dic_tracks[track] = artist
    fin.close()

    list_musics = []
    for k in dic_tracks.keys():
        list_musics.append(k)

    sim_users = {}
    users=glob.glob(path+'runs'+str(window)+"/history/*.txt")
    for user in users:
        sim_users[user.split('/')[-1][:-4]] = {}
    
    if not os.path.exists(path+'runs'+str(window)+"/CollaborativeFiltering/"):
        os.makedirs(path+'runs'+str(window)+"/CollaborativeFiltering/") 

    for user in users:
        dic_tracks_user_atual={}
        fin = open(user,'r')
        for line in fin:
            row=line.split('\t')
            add_dict(dic_tracks_user_atual,int(row[3]))
        fin.close()
        dic_users_ranking = cf.calc_users_similarity(user,dic_tracks_user_atual,sim_users,users)
        cf.collaborative_filtering(dic_users_ranking, user.split('/')[-1][:-4], list_musics,path+'runs'+str(window), common_users, path+'runs'+str(window)+"/history/")
        del(dic_tracks_user_atual)