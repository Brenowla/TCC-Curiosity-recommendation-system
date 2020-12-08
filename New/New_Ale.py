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

def calc_cur(bv,value):
	return sum(bv[3][c]*beta.pdf(value,bv[1][c],bv[2][c]) for c in range(bv[0]))

def get_new_recomentations_alexandre(c,tipo,path,path_collaborative_recomendation,path_estimulus,K,curvas,user):
	if tipo == 0:
		t = 'New_UB'
	elif tipo == 1:
		t = 'New_UK'
	for tet in [0.1,0.5,0.9]:
		if not os.path.exists(path+"Final_Collaborative_Filtering/"+str(K)+'/'+str(tet)+'/'+t+'/'):
			os.makedirs(path+"Final_Collaborative_Filtering/"+str(K)+'/'+str(tet)+'/'+t+'/')
	if user not in curvas.keys():
		return
	if not os.path.exists(path_estimulus):
		curvas.pop(user,1)
		return
	df = pd.read_csv(path_estimulus,sep='\t',comment='#')
	recomendations = {}
	fin = open(path_collaborative_recomendation+user,'r')
	count = 0
	curiosity = {}
	for line in fin:
		row = line.split('\t')
		recomendations[int(row[0])] = float(row[1][:-1])
		curiosity[int(row[0])] = 0
		count += 1
	for i in df.itertuples():
		if int(i.track) in curiosity.keys():
			if tipo == 0:
				aux = calc_cur(curvas[user],float(getattr(i, c)))
				if math.isnan(aux):
					aux = 0
				curiosity[int(i.track)] += aux
			else:
				aux = curvas[user][0].evaluate(getattr(i, c))
				if math.isnan(aux):
					aux = 0
				curiosity[int(i.track)] = aux

	curiosity = sorted(curiosity.items(), key=itemgetter(1),reverse = True)
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
		fout = open(path+"Final_Collaborative_Filtering/"+str(K)+'/'+str(tet)+'/'+t+'/'+user,'w+')
		for i in range(K):
			line = str(lista_rec[i][0])+'\n'
			fout.write(line)
			fout.flush()
		fout.close
			

def main(window,K,curvas,tipo,raiz,user): 
	path= raiz+'runs'+str(window)+'/'

	if tipo == 0:
		c = 'S_n'
	elif tipo == 1:
		c = 'S_b'


	path_estimulus =  raiz+'runs'+str(window)+'/da_all.tsv'

	path_collaborative_recomendation = path+'CollaborativeFiltering/'

	get_new_recomentations_alexandre(c,tipo,path,path_collaborative_recomendation,path_estimulus,K,curvas,user)