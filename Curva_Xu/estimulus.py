import numpy as np
import os
import glob
from operator import itemgetter

def add_dict(d,key):
    if key in d.keys():
        d[key]+=1
    else:
        d[key]=1

def add_dict_musicas(d,key,info):
    d[key]=info

def calc_users_similarity_pearson(user,users, dic_tracks,dic_tracks_user_atual):
    scores_users={}
    list_user_atual = []
    for m in dic_tracks.keys():
        if m in dic_tracks_user_atual.keys():
            list_user_atual.append(dic_tracks_user_atual[m])
        else:
            list_user_atual.append(0)
    for other_user in users:
        if other_user != user:
            fin = open(other_user,'r')
            dic_tracks_user_other={}
            for line in fin:
                row=line.split('\t')
                add_dict(dic_tracks_user_other,int(row[3]))
            fin.close()
            list_other_user = []
            for m in dic_tracks.keys():
                if m in dic_tracks_user_other.keys():
                    list_other_user.append(dic_tracks_user_other[m])
                else:
                    list_other_user.append(0)
            n = np.corrcoef(list_user_atual,list_other_user)
            add_dict_musicas(scores_users,other_user,n[0][1])
            del(dic_tracks_user_other)
    dic_users_ranking = sorted(scores_users.items(), key=itemgetter(1),reverse = True)
    return dic_users_ranking

def social_conflict(dic_users_ranking,user, dic_tracks,dic_tracks_user_atual,media_musicas, dic_estimulus):
	#Social Conflict
    dic_tracks_user_other = {}
    i = 0
    for n in range(5):
        if dic_users_ranking[n][1] > 0:
            i=n+1
    for other_user in range(i):
        fin = open(dic_users_ranking[other_user][0],'r')
        dic_tracks_user_other[dic_users_ranking[other_user][0]] ={}
        for line in fin:
            row=line.split('\t')
            add_dict(dic_tracks_user_other[dic_users_ranking[other_user][0]],int(row[3]))
        fin.close()
    for track in dic_tracks.keys():
        po = 0
        ne = 0
        pco = 0
        pcn = 0
        for other_user in range(i):
            if track in dic_tracks_user_other[dic_users_ranking[other_user][0]].keys():
                if dic_tracks_user_other[dic_users_ranking[other_user][0]][track]>media_musicas:
                    po += dic_users_ranking[other_user][1]*(dic_tracks_user_other[dic_users_ranking[other_user][0]][track] - media_musicas)
                    pco += dic_users_ranking[other_user][1]
                else:
                    ne += dic_users_ranking[other_user][1]*(media_musicas - dic_tracks_user_other[dic_users_ranking[other_user][0]][track] )
                    pcn += dic_users_ranking[other_user][1]
            else:
                ne += dic_users_ranking[other_user][1]*(media_musicas - 0)
                pcn += dic_users_ranking[other_user][1]
        if po>0:
            po = po/pco
        if ne>0:
            ne = ne/pcn
        conf = 1 - abs((po-ne)/(po+ne))
        dic_estimulus[track] = conf

def calc_nov(raiz,last_tracks,track,window,users):
    fin = open(raiz+'runs'+str(window)+"/Tracks_history/"+str(track)+'.txt')
    dic_track = {}
    for line in fin:
        row=line.split('\t')
        dic_track[row[0]] = int(row[1])
    nov = 0
    for t in range(len(last_tracks)):
        fin = open(raiz+'runs'+str(window)+"/Tracks_history/"+last_tracks[t]+'.txt')
        dic_other_track = {}
        for line in fin:
            row=line.split('\t')
            dic_other_track[row[0]] = int(row[1])
        list_track = []
        list_other_track = []
        for i in users:
            k = i.split('/')[-1][:-4]
            if k in dic_track.keys():
                list_track.append(dic_track[k])
            else:
                list_track.append(0)
            if k in dic_other_track.keys():
                list_other_track.append(dic_other_track[k])
            else:
                list_other_track.append(0)
        pcc = np.corrcoef(list_track,list_other_track)
        pcc = pcc[0][1]
        nov += np.exp(-0.1*len(last_tracks)-t)*((1-pcc)/2)
    return nov
    


def main(raiz,window,user,track_artists,users):
    fin = open(raiz+'runs'+str(window)+"/media.txt")
    media = float(fin.read())
    print(user)
    users=glob.glob(raiz+'runs'+str(window)+"/history/*.txt")
    dic_tracks_user_atual={}
    dic_estimulus = {}
    fin = open(raiz+'runs'+str(window)+"/history/"+user,'r')
    for line in fin:
        row=line.split('\t')
        add_dict(dic_tracks_user_atual,int(row[3]))
    fin.close()
    #dic_users_ranking = calc_users_similarity_pearson(user,users,track_artists,dic_tracks_user_atual)
    #social_conflict(dic_users_ranking,user,track_artists,dic_tracks_user_atual,media,dic_estimulus)
    fin = open(raiz+'runs'+str(window)+"/history/"+user,'r')
    fout = open(raiz+'runs'+str(window)+"/pcl.txt",'w')
    fout.write('track\tcur\n')
    fout.flush()
    last_songs = []
    for line in fin:
        row=line.split('\t')
        nov = calc_nov(raiz,last_songs,row[3],window,users)
        fout.write(row[3]+'\t'+str(nov)+'\n')
        fout.flush()
        last_songs.append(row[3])
        if len(last_songs)>3:
            last_songs = last_songs[1:]
    fin = open(raiz+'runs'+str(window)+"/CollaborativeFiltering/"+user)
    fout = open(raiz+'runs'+str(window)+"/pcl_all.txt",'w')
    fout.write('track\tcur\n')
    fout.flush()
    for line in fin:
        row=line.split('\t')
        nov = calc_nov(raiz,last_songs,row[0],window,users)
        fout.write(row[0]+'\t'+str(nov)+'\n')
        fout.flush()
