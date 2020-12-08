# Pacotes para análise dos dados
import pandas as pd
import numpy as np
import os

import glob

from sklearn.decomposition import NMF

def calc_nov(path_nov, track, last_tracks):
    fin = open(path_nov+str(track)+'.txt','r')
    track_hist = []
    for line in fin:
        track_hist.append(float(line))
    fin.close()
    nov = 0
    for other_track in range(len(last_tracks)):
        other_track_hist = []
        fin = open(path_nov+str(last_tracks[other_track])+'.txt','r')
        for line in fin:
            other_track_hist.append(float(line))
        pc = np.corrcoef(track_hist,other_track_hist)
        dissim = (1-pc[0][1])/2
        nov += np.exp(-other_track)*dissim
    return nov
        

def main(window,raiz):
    path=raiz

    fin=open(path+'runs'+str(window)+'/tracks_artists/tracks_artists.txt','r')
    list_tracks = []
    for line in fin:
        row=line.split('\t')
        track=int(row[0])
        list_tracks.append(track)
    fin.close()

    users=glob.glob(path+'runs'+str(window)+"/stimulus-bits-new-iii-All_Tracks/users/*.tsv")
    path_songs_hist = path+'runs'+str(window)+"/hist_tracks/"
    path_conflict = path+'runs'+str(window)+"/Social_conflict/"
    path_hist = path+'runs'+str(window)+"/history/"
    path_estimulus = path+'runs'+str(window)+"/stimulus-bits-new-iii/users/"

    for user in users:
        conflito = {}
        fin = open(path_conflict+user.split('/')[-1][:-4]+'.txt')
        for line in fin:
            row = line.split('\t')
            conflito[int(row[0])] = float(row[1])
        last_tracks = []
        fout = open(path_estimulus+user.split('/')[-1][:-4]+'.txt','w+')
        fin = open(path_hist+user.split('/')[-1][:-4]+'.txt','r')
        for line in fin:
            row = line.split('\t')
            track = int(row[3])
            nov = calc_nov(path_songs_hist,track,last_tracks)
            fout.write(str(track)+'\t'+str(conflito[track])+'\t'+str(nov)+'\n')
            fout.flush()
            last_tracks.append(track)
            if len(last_tracks)>7:
                del(last_tracks[0])
    

    matriz_estimulus = []
    for user in users:
        df = pd.read_csv(path_estimulus+str(user.split("/")[-1][:-4]+'.txt'),sep='\t',comment='#', names=['track', 'Conf','Nov'])
        estimulus = {}
        for i in df.itertuples():
            estimulus[int(i.track)] = i.Nov
        list_estimulus = []
        for track in list_tracks:
            if track in estimulus.keys():
                list_estimulus.append(estimulus[track])
            else:
                list_estimulus.append(0)
        matriz_estimulus.append(list_estimulus)

    model = NMF(n_components=5).fit(matriz_estimulus)
    matriz_fac = model.inverse_transform(model.transform(matriz_estimulus))
    for i in range(len(users)):
        fout = open(path+'runs'+str(window)+"/stimulus-bits-new-iii-All_Tracks/users/"+users[i].split("/")[-1][:-4]+'.txt','w')
        fout.write('user\ttrack\tNov\n')
        fout.flush()
        user = users[i].split("/")[-1][:-4]
        for j in range(len(list_tracks)):
            fout.write(user+'\t'+str(list_tracks[j])+'\t'+str(matriz_fac[i][j])+'\n')
            fout.flush()



