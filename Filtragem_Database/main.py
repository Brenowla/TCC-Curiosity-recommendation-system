# Pacotes para análise dos dados
import pandas as pd
import numpy as np
import os
#import CollaborativeFiltering as cf
import glob

# For our 
import seaborn as sns
from random import randint
import random

#Abrindo dados
def add_dict_musicas(d,key,info):
    d[key]=info

def main(path):
	for i in range(0,10):	
		if not os.path.exists(path+'runs'+str(i)+'/'):
			os.makedirs(path+'runs'+str(i)+'/')
		if not os.path.exists(path+'runs'+str(i)+'/'+'history/'):
			os.makedirs(path+'runs'+str(i)+'/'+'history/')
		if not os.path.exists(path+'runs'+str(i)+'/'+'test/'):
			os.makedirs(path+'runs'+str(i)+'/'+'test/')

	users=glob.glob(path+"runs/*.txt")

	#Encontrar os timestamps limite

	'''
	menor_tempo = 999999999999
	maior_tempo = 0
	total_erro = 0
	for user in users:
		contador = 0
		time_count = 0
		correto = 1
		fin = open(user,'r')
		for line in fin:
			contador+=1
			row=line.split('\t')
			if int(row[4]) > maior_tempo:
				maior_tempo = int(row[4])
			if int(row[4]) < menor_tempo:
				menor_tempo =  int(row[4])
		fin.close()
	
	
	'''

	'''
	#Gerar valores aleátorios para a semana
	maior_tempo = 1388541599
	menor_tempo = 1357005600

	week = 7*24*3600
	day = 24*3600
	random.seed()
	
	for i in range(10):
		print(random.randint(menor_tempo,maior_tempo-(week+day)))
	'''
	
	week = 7*24*3600
	day = 24*3600	
	#Vetor com 10 tempos selecionados aleatoriamente
	timestamps = [1366826976,1359226052,1369017287,1363608690,1371244390,1357309082,1377697649,1375260569,1363467892,1367931237]
	#Filtrar usuários com 1000 eventos por janela de tempo
	for user in users:
		history = []
		test = []
		for i in range(10):
			history.append([])
			test.append([])
		fin = open(user,'r')
		for line in fin:
			row=line.split('\t')
			for time in range(10):
				if int(row[4]) >= timestamps[time] and int(row[4]) < timestamps[time]+week:
					history[time].append(line)
				if int(row[4]) >= timestamps[time]+week and int(row[4]) < timestamps[time]+week+day:
					test[time].append(line)
		fin.close()
		for time in range(10):
			if len(history[time]) > 100 and len(test[time]) >50:
				fout = open(path+'runs'+str(time)+'/history/'+user.split("/")[-1],'w+')
				for line in history[time]:
					fout.write(line)
					fout.flush()
				fout.close()
				fout = open(path+'runs'+str(time)+'/test/'+user.split("/")[-1],'w+')
				for line in test[time]:
					fout.write(line)
					fout.flush()
				fout.close()
	
	'''
	#Dividir os dados em 10 janelas - Antiga
	for user in users:
		if user not in rejecteds:
			time_count = 0
			fin = open(user,'r')
			fout = open(path+'runs'+str(time_count)+'/'+user.split("/")[8],'w+')
			for line in fin:
				row=line.split('\t')
				if int(row[4]) > timestamps[time_count]:
					time_count+=1
					fout.close()
					fout = open(path+'runs'+str(time_count)+'/'+user.split("/")[8],'w+')
				fout.write(line)
				fout.flush()
			fout.close()
			fin.close()
	'''
	
	#Gerando arquivo de músicas para cada janela de tempo
	for i in range(10):
		dic_t_user={}
		users=glob.glob(path+"runs"+str(i)+"/history/*.txt")
		for user_data in users:
			fin = open(path+'runs'+str(i)+'/history/'+user_data.split("/")[-1],'r')
			for line in fin:
				row=line.split('\t')
				add_dict_musicas(dic_t_user,int(row[3]),int(row[1]))
			fin.close()
		if not os.path.exists(path+'runs'+str(i)+"/tracks_artists/"):
			os.makedirs(path+'runs'+str(i)+"/tracks_artists/") 
		fout = open(path+'runs'+str(i)+"/tracks_artists/tracks_artists.txt",'w+')
		for k in dic_t_user.keys():
			line ='%d\t%d\n'%\
			(k,dic_t_user[k])
			fout.write(line)
			fout.flush()
		fout.close()