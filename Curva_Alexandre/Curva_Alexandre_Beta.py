import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import beta
from matplotlib import pyplot as plt
import Curva_Alexandre.betamix as bm
import scipy.integrate as integrate
import glob


def pdf_Alexandre(path,path_estimulus,user,curvas):
	if not os.path.exists(path):
		return
	if not os.path.exists(path_estimulus):
		return
	df = pd.read_csv(path,sep='\t',comment='#')
	if len(df.index) < 30:
		return
	C=0
	for C in range (1,4):
		A,B,Pi = bm.init(C, df['S_n'])
		A,B,Pi,k,count,W = bm.estimate(A,B,Pi,df['S_n'],C,niter=100)
		n,ks1,ks2,dmax,d,p,pvalue = bm.test(df['S_n'],A,B,Pi,C)
		if pvalue>=p and dmax<d:
			break
	maior = 0
	for c in range(C):
		m,k,sd_max,sd_inf,sd_sup = bm.data_from_beta_dist(A[c],B[c])
		v_k = sum(Pi[c]*beta.pdf(m,A[c],B[c]) for c in range(C))
		if v_k > maior:
			maior = v_k
	curvas[user] = [C,A,B,Pi,m,maior]

def main(window,raiz,curvas,user):
	path_estimulus =  raiz+'runs'+str(window)+'/da_all.tsv'
	path = raiz+'runs'+str(window)+'/da.tsv'
	pdf_Alexandre(path,path_estimulus,user,curvas)
