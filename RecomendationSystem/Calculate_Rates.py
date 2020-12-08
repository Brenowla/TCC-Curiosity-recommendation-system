import pandas as pd
from operator import itemgetter
import os

def add_dict_musicas(d,key,info):
    d[key]=info

def add_dict(d,key,value):
    if key in d.keys():
        d[key]+=value
    else:
        d[key]=value

def sort_scores(e):
    return e[1]

def save(user, path, list_music,list_matriz):
	if not os.path.exists(path+"/MatrizRec/"):
	    os.makedirs(path+"/MatrizRec/")
	dic_tracks={}
	for track in range(len(list_matriz)):
		dic_tracks[list_music[track]] = list_matriz[track]
	dic_ranking = sorted(dic_tracks.items(), key=itemgetter(1),reverse = True)
	fout = open(path+"/MatrizRec/"+user+'.txt',"w+")
	max = dic_ranking[0][1]
	for i in range(len(dic_ranking)):
		line = str(dic_ranking[i][0])+'\t'+str(dic_ranking[i][1]/max)+'\n'
		fout.write(line)
		fout.flush()
	fout.close()
	
def generate_list_tracks(dic_users_ranking,list_musics,matriz_music,list_users):
	for other_user in range(50):
	   	fin = open(dic_users_ranking[other_user][0],'r')
	   	list_users.append(dic_users_ranking[other_user][0])
	   	list_music_user=[]
	   	dic_t_user={}
	   	for line in fin:
	   		row=line.split('\t')
	   		add_dict_musicas(dic_t_user,int(row[3]),int(row[1]))
	   	fin.close()
	   	for t in list_musics:
	   		if t in dic_t_user.keys():
	   			list_music_user.append(1)
	   		else:
	   			list_music_user.append(0)
	   	matriz_music.append(list_music_user)