import os
import numpy as np
import pandas as pd
import glob
from PerformanceMetrics.util import data_from_beta_dist,integration_beta_dist
import PCurioLinear.PCurioModelLinear as PC
import DataAnalysis.DataAnalysis as DA
import Curva_Xu.estimulus as CXE
from scipy.stats import beta
import pulp as p
import math
import numba

def calc_cur(bv,value):
	return sum(bv[3][c]*beta.pdf(value,bv[1][c],bv[2][c]) for c in range(bv[0]))

def get_history(history):
	list_musics = []
	for line in history:
		row=line.split('\t')
		list_musics.append(int(row[3]))
	lista = np.array(list_musics)
	return lista
        
def get_list_musics_recomendation(arq,K):
	list_musics = []
	i = 0
	for line in arq:
		if i == K:
			break
		row=line.split('\t')
		list_musics.append(int(row[0]))
		i+=1
	lista = np.array(list_musics)
	return lista

def get_list_musics_final_recomendation(arq,K):
	list_musics = []
	for line in arq:
		row=line.split('\t')
		list_musics.append(int(row[0]))
	lista = np.array(list_musics)
	return lista

def estimulation_fitness_b(list_recomendations,curiosity,curva,K):
	soma = 0
	if curva[5] == 0:
		return 0
	for music in list_recomendations:
		soma += float(curiosity[music])/curva[5]
	if math.isnan(soma):
		soma = 0
	return soma/K

def estimulation_fitness_k(list_recomendations,curiosity,curva,K):
	soma = 0
	if curva[1] == 0:
		return 0
	for music in list_recomendations:
		soma += float(curiosity[music])/curva[1]
	if math.isnan(soma):
		soma = 0
	return soma/K

def estimulation_fitness_x(list_recomendations,curiosity,curva,K):
	soma = 0
	if curva[2] == 0:
		return 0
	for music in list_recomendations:
		soma += float(curiosity[music])/curva[2]
	if math.isnan(soma):
		soma = 0
	return soma/K

@numba.jit(nopython=True)
def recomendation_precision(list_recomendations,list_history,K):
	sum = 0
	for i in list_recomendations:
		for j in list_history:
			if i == j:
				sum += 1
				break
	return sum/K

@numba.jit(nopython=True)
def inter_user_similarity(list_recomendations,list_recomendations_other_user,K):
	sum = 0
	for i in list_recomendations:
		for j in list_recomendations_other_user:
			if i == j:
				sum += 1
	#print(sum/K)
	return sum/K

def main(window,K,curvas,tipo,raiz,recomendation_precision_colaborative,inter_user_similarity_colaborative):
	if tipo == 0:
		t = 'CBRS_UB'
		c = 'S_n'
	elif tipo == 1:
		t = 'CBRS_Z'
		c = 'Nov'
	elif tipo == 2:
		t = 'CBRS_UK'
		c = 'S_b'
	elif tipo == 3:
		t = 'New_UB'
		c = 'S_n'
	elif tipo == 4:
		t = 'New_UK'
		c = 'S_b'
	elif tipo == 5:
		t = 'SeCM'
		c = 'cur'
	
	artist_genre = {}
	fin=open(raiz+'/LFM-1b_artists_genres.txt','r')
	for line in fin:
		row=line.split('\t')
		artist=int(row[0])
		genre=int(row[1][:-1])
		if not artist in artist_genre.keys():
			artist_genre[artist]=set([genre])
		else:
			artist_genre[artist].add(genre)
	fin.close()

	fin=open(raiz+"runs"+str(window)+'/tracks_artists/tracks_artists.txt','r')
	track_artist = {}
	for line in fin:
		row=line.split('\t')
		track=int(row[0])
		artist=int(row[1][:-1])
		track_artist[track] = artist
	fin.close()
	
	path = raiz+'runs'+str(window)+'/'
	
	path_collaborative_recomendation = path+'CollaborativeFiltering/'

	if tipo == 5:
		path_estimulus = path+'pcl_all.txt'
	else:
		path_estimulus = path+'da_all.tsv'

	path_verify = raiz+'runs'+str(window)+'/test/'

	if not os.path.exists(path+"Performance/"):
		os.makedirs(path+"Performance/")

	fout = open(path+"Performance/performance.txt",'a')

	users_idx=curvas.keys()

	if recomendation_precision_colaborative == 0 or tipo == 5:
		recomendation_precision_colaborative = 0

		inter_user_similarity_colaborative = 0

	estimulation_fitness_recomendations_colaborative = 0

	estimulation_fitness_recomendations_colaborative_final = {}

	recomendation_precision_colaborative_final = {}

	inter_user_similarity_colaborative_final = {}

	for tet in [0.1,0.5,0.9]:
		estimulation_fitness_recomendations_colaborative_final[tet] = 0
		recomendation_precision_colaborative_final[tet] = 0
		inter_user_similarity_colaborative_final[tet] = 0
	
	sum_inter = {}
	sum_inter['base'] = {}
	for tet in [0.1,0.5,0.9]:
		sum_inter[tet] = {}
	for user in users_idx:
		sum_inter['base'][user] = 0
		for tet in [0.1,0.5,0.9]:
			sum_inter[tet][user] = 0
	for user in users_idx:
		if tipo == 0 or tipo == 2 or tipo ==3 or tipo == 4:
			PC.main(window,raiz,0,user,artist_genre,track_artist,path+'CollaborativeFiltering/')
			DA.main(window,raiz,0,user)
		if tipo == 1:
			PC.main(window,raiz,1,user,artist_genre,track_artist,path+'CollaborativeFiltering/')
			DA.main(window,raiz,1,user)
		if tipo == 5:
			CXE.main(raiz,window,user,track_artist,users_idx)
		curiosity = {}
		df = pd.read_csv(path_estimulus,sep='\t',comment='#')
		estimulus = df[c].values
		music = df['track'].values
		for i in range(len(music)):
			if tipo == 0 or tipo == 1 or tipo == 3:
				aux = calc_cur(curvas[user],estimulus[i])
				if math.isnan(aux):
					aux = 0
				curiosity[int(music[i])]= aux
			elif tipo== 2 or tipo == 4:
				aux = float(curvas[user][0].evaluate(estimulus[i]))
				if math.isnan(aux):
					aux = 0
				curiosity[int(music[i])]= aux
			else:
				curiosity[int(music[i])] = (1/(1+np.exp(-20*(estimulus[i]-curvas[user][0])))-1/(1+np.exp(-20*(estimulus[i]-curvas[user][1]))))
			
		list_history_actual_user = get_history(open(path_verify+user,'r'))
		
		list_recomendations = get_list_musics_recomendation(open(path_collaborative_recomendation+user,'r'),K)

		if tipo == 0 or tipo == 1 or tipo == 3:
			estimulation_fitness_recomendations_colaborative += estimulation_fitness_b(list_recomendations, curiosity,curvas[user],K)
		elif tipo== 2 or tipo == 4:
			estimulation_fitness_recomendations_colaborative += estimulation_fitness_k(list_recomendations, curiosity,curvas[user],K)
		else:
			estimulation_fitness_recomendations_colaborative += estimulation_fitness_x(list_recomendations, curiosity,curvas[user],K)

		if tipo == 0 or tipo == 5:
			recomendation_precision_colaborative += recomendation_precision(list_recomendations,list_history_actual_user,K)
			
			aux = 1
			for other_user in users_idx:
				if other_user == user:
					aux = 0
					continue
				if aux == 1:
					continue
				list_recomendations_other_user = get_list_musics_recomendation(open(path_collaborative_recomendation+other_user,'r'),K)
				aux1 = inter_user_similarity(list_recomendations,list_recomendations_other_user,K)
				sum_inter['base'][user] += aux1
				sum_inter['base'][other_user] += aux1

			inter_user_similarity_colaborative += sum_inter['base'][user]/(len(users_idx)-1)

		for tet in [0.1,0.5,0.9]:
			#------------------------------------------------------------------------------------------------------------#
			path_collaborative_recomendation_final = path+'Final_Collaborative_Filtering/'+str(K)+'/'+str(tet)+'/'+t+'/'
			
			list_recomendations_final =  get_list_musics_final_recomendation(open(path_collaborative_recomendation_final+user,'r'),K)

			if tipo == 0 or tipo == 1 or tipo == 3:
				estimulation_fitness_recomendations_colaborative_final[tet] +=estimulation_fitness_b(list_recomendations_final,curiosity,curvas[user],K)
			elif tipo== 2 or tipo == 4:
				estimulation_fitness_recomendations_colaborative_final[tet] +=estimulation_fitness_k(list_recomendations_final,curiosity,curvas[user],K)
			else:
				estimulation_fitness_recomendations_colaborative[tet] += estimulation_fitness_x(list_recomendations, curiosity,curvas[user],K)

			recomendation_precision_colaborative_final[tet] += recomendation_precision(list_recomendations_final,list_history_actual_user,K)
					
			aux = 1
			for other_user in users_idx:
				if other_user == user:
					aux = 0
					continue
				if aux == 1:
					continue
				list_recomendations_other_user = get_list_musics_recomendation(open(path_collaborative_recomendation_final+other_user,'r'),K)
				aux1 = inter_user_similarity(list_recomendations,list_recomendations_other_user,K)
				sum_inter[tet][user] += aux1
				sum_inter[tet][other_user] += aux1

			inter_user_similarity_colaborative_final[tet] += sum_inter[tet][user]/(len(users_idx)-1)

	estimulation_fitness_recomendations_colaborative = estimulation_fitness_recomendations_colaborative/len(curvas.keys())

	if tipo == 0 or tipo ==1:
		recomendation_precision_colaborative = recomendation_precision_colaborative/len(users_idx)
		inter_user_similarity_colaborative = inter_user_similarity_colaborative/len(users_idx)			
	
	for tet in [0.1,0.5,0.9]:
		estimulation_fitness_recomendations_colaborative_final[tet] = estimulation_fitness_recomendations_colaborative_final[tet]/len(users_idx)

		recomendation_precision_colaborative_final[tet] = recomendation_precision_colaborative_final[tet]/len(users_idx)

		inter_user_similarity_colaborative_final[tet] = inter_user_similarity_colaborative_final[tet]/len(users_idx)
	
		store_line = str(estimulation_fitness_recomendations_colaborative)+'\t'+str(estimulation_fitness_recomendations_colaborative_final[tet])+'\t'
			
		store_line += str(recomendation_precision_colaborative)+'\t'+str(recomendation_precision_colaborative_final[tet])+'\t'

		store_line += str(inter_user_similarity_colaborative)+'\t'+str(inter_user_similarity_colaborative_final[tet])+'\t'

		store_line += t+'\t'
			
		store_line += str(K)+'\t'+str(tet)+'\n'

		fout.write(store_line)
		fout.flush()
		
	return recomendation_precision_colaborative,inter_user_similarity_colaborative
