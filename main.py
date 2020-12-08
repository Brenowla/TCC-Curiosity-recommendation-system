import sys
import glob
from multiprocessing import Process
import time
import warnings
import os
'''
import pandas as pd
import Filtragem_Database.main as FD
import Curva_ZHAO.Curva_ZHAO as CZ
import Curva_ZHAO.proxEstimulos as PE
import DataAnalysis.DataAnalysis as DA 
import Optimization.OP as OP
import New.New_Ale as NA
import PCurioLinear.PCurioModelLinear as PCL 
import RecomendationSystem.RecomendationSystem as RS
import PerformanceMetrics.PerformanceMetrics as PM
import Curva_Alexandre.Curva_Alexandre_Beta as CAB
import Curva_Alexandre.Curva_Alexandre_KDE as CAK
import Curva_Xu.base as CXB
import Curva_Xu.estimulus as CXE
import Curva_Xu.Curva_Xu as CX
import Curva_Xu.Recomender as CXR
'''
'''
def frameworks_ale(window,raiz,K):
	cols = ['EF','EFC','RP','RPC',"IU",'IUC','type','K','tet']
	
	if os.path.exists(raiz+"runs"+str(window)+"/Performance/performance.txt"):
		os.remove(raiz+"runs"+str(window)+"/Performance/performance.txt")
	
	p1,p2 = {},{}
	curvas = {}
	users=glob.glob(raiz+"runs"+str(window)+"/history/*.txt")
	for i in range(len(users)):
		users[i] = users[i].split("/")[-1]
	artist_genre={}
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
	#Beta
	
	print("CBRS_UB - New_UB")
	for user in users:
		PCL.main(window,raiz,0,user,artist_genre,track_artist,raiz+'runs'+str(window)+'/CollaborativeFiltering/')
		DA.main(window,raiz,0,user)
		CAB.main(window,raiz,curvas,user)
		for k in K:
			OP.main(window,k,curvas,raiz,0,user)
			NA.main(window,k,curvas,0,raiz,user)
	print('Performances - CBRS_UB - New_UB')
	for k in K:
		p1[k],p2[k] = PM.main(window,k,curvas,0,raiz,-1,-1)
		PM.main(window,k,curvas,3,raiz,p1[k],p2[k])
	
	#KDE
	print('CBRS_UK- NEW_UK')
	curvas = {}
	for user in users:
		PCL.main(window,raiz,0,user,artist_genre,track_artist,raiz+'runs'+str(window)+'/CollaborativeFiltering/')
		DA.main(window,raiz,0,user)
		CAK.main(window,raiz,curvas,user)
		for k in K:
			OP.main(window,k,curvas,raiz,2,user)
			NA.main(window,k,curvas,1,raiz,user)
	print('Performances - CBRS_UK- NEW_UK')
	for k in K:
		PM.main(window,k,curvas,2,raiz,p1[k],p2[k])
		PM.main(window,k,curvas,4,raiz,p1[k],p2[k])
	
	#Zhao
	print('CBRS_Z')
	users=glob.glob(raiz+"runs"+str(window)+"/Final_Collaborative_Filtering/5/0.1/CBRS_UB/*.txt")
	for i in range(len(users)):
		users[i] = users[i].split("/")[-1]
	for user in users:
		PCL.main(window,raiz,1,user,artist_genre,track_artist,raiz+'runs'+str(window)+'/CollaborativeFiltering/')
		DA.main(window,raiz,1,user)
		CZ.main(window,raiz,curvas,user)
		for k in K:
			OP.main(window,k,curvas,raiz,1,user)
	for k in K:
		PM.main(window,k,curvas,1,raiz,p1[k],p2[k])
	
	print('XU')
	#CXB.main(raiz,window,users)
	for user in users:
		CXE.main(raiz,window,user,track_artist,users)
		CX.main(window,raiz,curvas,user)
		for k in K:
			CXR.main(window,k,curvas,raiz,user)
	for k in K:
		PM.main(window,k,curvas,5,raiz,p1[k],p2[k])
		#break
	

'''	
	

if __name__ == '__main__':
	warnings.filterwarnings('ignore')
	max_window = 9
	raiz = 'C:\\Users\\breno\\OneDrive\\Documentos\\Database_TCC\\window1'
	K = [5,20,50]
	inic = [0]
	end = [4,6,10]
	
	print('Divisão Dataset')
	#FD.main(raiz)
	'''
	proc = []
	print("Recomendation System")
	for i in range(3):
		proc = []
		for window in range(inic[i],end[i]):
			p = Process(target=RS.main,args = (window,raiz,))
			proc.append(p)
			p.start()
		for p in proc:
			p.join()
	
	print('Frameworks')
	for i in range(1):
		proc = []
		for window in inic:
			p = Process(target=frameworks_ale,args = (window,raiz,K,))
			proc.append(p)
			p.start()
		for p in proc:
			p.join()
	'''
	fout = open(raiz+'\\performances.txt','w')
	for window in range(10):
		fin = open(raiz+"\\performance"+str(window)+".txt",'r')
		for line in fin:
			fout.write(line)
			fout.flush()
	fout.close()
	
	
	
	
	

	

		
	
	
