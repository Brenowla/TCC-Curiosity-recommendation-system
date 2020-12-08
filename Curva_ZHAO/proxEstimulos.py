# Pacotes para análise dos dados
import pandas as pd
import numpy as np
import os

import glob

from sklearn.decomposition import NMF

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
    path_estimulus = path+'runs'+str(window)+"/stimulus-bits-new-iii/users/"

    matriz_estimulus = []
    for user in users:
        df = pd.read_csv(path_estimulus+str(user.split("/")[-1]),sep='\t',comment='#')
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
        fout = open(path+'runs'+str(window)+"/stimulus-bits-new-iii-All_Tracks/users/"+users[i].split("/")[-1],'w')
        fout.write('user\ttrack\tNov\n')
        fout.flush()
        user = users[i].split("/")[-1][:-4]
        for j in range(len(list_tracks)):
            fout.write(user+'\t'+str(list_tracks[j])+'\t'+str(matriz_fac[i][j])+'\n')
            fout.flush()
        