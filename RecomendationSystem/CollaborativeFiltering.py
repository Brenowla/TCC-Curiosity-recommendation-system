import pandas as pd
from scipy.sparse import csr_matrix 
from scipy import spatial
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

def calc_users_similarity(user,list_music_user,sim_users,users):
	name = user.split('/')[-1][:-4]
	for other_user in users:
		if other_user != user:
			name_other = other_user.split('/')[-1][:-4]
			if name_other not in sim_users[name].keys():
				dic_tracks_other_user = {}
				fin = open(other_user,'r')
				for line in fin:
					row=line.split('\t')
					add_dict_musicas(dic_tracks_other_user,int(row[3]),1)
				fin.close()
				sim_users[name][name_other] = 0
				sim_users[name_other][name] = 0
				for track in dic_tracks_other_user.keys():
					if track in list_music_user.keys():
						sim_users[name][name_other] += 1
						sim_users[name_other][name] += 1
	dic_users_ranking = sorted(sim_users[name].items(), key=itemgetter(1),reverse = True)
	del(sim_users[name])
	return dic_users_ranking

def collaborative_filtering(dic_users_ranking, user, list_musics, path, common_users, path_open):
	#Collaborative filtering
	dic_ranking={}
	for other_user in range(common_users):
		fin = open(path_open+str(dic_users_ranking[other_user][0])+'.txt','r')
		dic_tracks_other_user = {}
		for line in fin:
			row=line.split('\t')
			add_dict_musicas(dic_tracks_other_user,int(row[3]),1)
		fin.close()
		for track in dic_tracks_other_user.keys():
			add_dict(dic_ranking,track,1)
	dic_ranking = sorted(dic_ranking.items(), key=itemgetter(1),reverse = True)
	max = dic_ranking[0][1]
	fout = open(path+"/CollaborativeFiltering/"+user+'.txt',"w+")
	for i in range(len(dic_ranking)):
		line = str(dic_ranking[i][0])+'\t'+str(dic_ranking[i][1]/max)+'\n'
		fout.write(line)
		fout.flush()
	fout.close()
		