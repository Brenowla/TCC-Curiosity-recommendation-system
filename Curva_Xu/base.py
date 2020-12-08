import numpy as np
import os
import glob

def add_dict(d,key):
    if key in d.keys():
        d[key]+=1
    else:
        d[key]=1

def main(raiz,window,users):
    if not os.path.exists(raiz+'runs'+str(window)+"/Tracks_history/"):
        os.makedirs(raiz+'runs'+str(window)+"/Tracks_history/") 

    dic_songs = {}
    for user in users:
        dic_tracks_user_atual={}
        fin = open(raiz+'runs'+str(window)+"/history/"+user,'r')
        for line in fin:
            row=line.split('\t')
            add_dict(dic_tracks_user_atual,int(row[3]))
            add_dict(dic_songs,row[3])
        for track in dic_tracks_user_atual.keys():
            fout = open(raiz+'runs'+str(window)+"/Tracks_history/"+str(track)+'.txt','a+')
            fout.write(user.split('/')[-1][:-4]+'\t'+str(dic_tracks_user_atual[track])+'\n')
            fout.flush()
            fout.close()
    
    soma = 0
    for i in dic_songs.keys():
        soma+=dic_songs[i]
    
    fout = open(raiz+'runs'+str(window)+"/media.txt",'w')
    fout.write(str(soma/len(dic_songs.keys())))
        