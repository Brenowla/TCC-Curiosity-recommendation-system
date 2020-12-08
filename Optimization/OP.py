import os
import numpy as np
import pandas as pd
import math
import glob
from Optimization.util import data_from_beta_dist,integration_beta_dist
from scipy.stats import beta
import pulp as p
import statsmodels.api as sm

def isNaN(num):
    return num != num

def calc_cur(bv,value):
	return sum(bv[3][c]*beta.pdf(value,bv[1][c],bv[2][c]) for c in range(bv[0]))

def init_alexandre(t,df,user,path2,curvas,m):
	curiosity = {}
	recomendations = {}
	estimulus = {}
	tracks = []
	fin = open(path2,'r')
	count = 0
	
	if t == 'CBRS_UB':
		c = 'S_n'
		curva = 0
	elif t == 'CBRS_Z':
		c = 'Nov'
		curva = 0
	else:
		curva = 1
		c = 'S_b'
	for line in fin:
		row = line.split('\t')
		aux = float(row[1][:-1])
		if math.isnan(aux):
			aux = 0
		recomendations[int(row[0])] = aux
		curiosity[int(row[0])] = 0
		estimulus[int(row[0])] = 0
		if len(tracks)<100:
			tracks.append(int(row[0]))
		count += 1
		if count == m:
			break
	for i in df.itertuples():
		if int(i.track) in curiosity.keys():
			estimulus[int(i.track)] += getattr(i, c)
			if curva == 0:
				aux = calc_cur(curvas[user],float(getattr(i, c)))
				if math.isnan(aux):
					aux = 0
				if aux>9999:
					aux = 9999
				curiosity[int(i.track)] += aux
			else:
				aux = curvas[user][0].evaluate(getattr(i, c))
				if math.isnan(aux):
					aux = 0
				curiosity[int(i.track)] += aux
	return tracks,curiosity,recomendations,estimulus

def get_new_recomentations_alexandre(t,path,path_collaborative_recomendation,path_estimulus,K,curvas,user):
	m = K+50
	if t == 'CBRS_UB':
		mod = 4
	elif t == 'CBRS_Z':
		mod = 4
	else:
		mod = 2
	if user not in curvas.keys():
		return
	tol = 0.02*K*curvas[user][mod]
	if math.isnan(tol):
		tol = 0.5
	if not os.path.exists(path_estimulus):
		curvas.pop(user,1)
		return
	df = pd.read_csv(path_estimulus,sep='\t',comment='#')
	tracks,curiosity,recomendations,estimulus = init_alexandre(t,df,user,path_collaborative_recomendation+str(user),curvas,m)
		
	for tet in [0.1,0.5,0.9]:
		if  tet==1 and sum([float(x) for x in curiosity.values()])== 0:
			curiosity = recomendations

		resolution_alexandre(tracks,curvas[user][mod],recomendations,curiosity,estimulus,tet,tol,K,path+"Final_Collaborative_Filtering/"+str(K)+'/'+str(tet)+'/'+t+'/'+user)

		

def resolution_alexandre(tracks,mod,recomentations,curiosity,estimulus,tet,tol,K,path):
	Lp_prob = p.LpProblem('Zhao', p.LpMaximize)  
	y = p.LpVariable.dicts("y",tracks,cat='Binary')
	Lp_prob += p.lpSum((1-tet)*y[i]*recomentations[i]+tet*y[i]*curiosity[i] for i in tracks)
	Lp_prob += p.lpSum(y[i] for i in tracks)==K
	status = Lp_prob.solve()
	obj = p.value(Lp_prob.objective)
	fout = open(path,'w+')
	for v in Lp_prob.variables():
		if v.varValue>0:
			fout.write(v.name[2:]+'\n')
			fout.flush()


def main(window,K,curvas,raiz,tipo,user): 
	path= raiz+'runs'+str(window)+'/'

	path_estimulus =  raiz+'runs'+str(window)+'/da_all.tsv'

	path_collaborative_recomendation = path+'CollaborativeFiltering/'

	t = ''
	if tipo == 0:
		t = 'CBRS_UB'
	elif tipo == 1:
		t = 'CBRS_Z'
	else:
		t = 'CBRS_UK'
	for tet in [0.1,0.5,0.9]:
		if not os.path.exists(path+"Final_Collaborative_Filtering/"+str(K)+'/'+str(tet)+'/'+t+'/'):
			os.makedirs(path+"Final_Collaborative_Filtering/"+str(K)+'/'+str(tet)+'/'+t+'/')

	get_new_recomentations_alexandre(t,path,path_collaborative_recomendation,path_estimulus,K,curvas,user)