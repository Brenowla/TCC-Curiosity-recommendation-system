import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import beta
from matplotlib import pyplot as plt
import Curva_ZHAO.util as ut
import Curva_ZHAO.betamix as bm
import scipy.integrate as integrate
import glob

def pdf_ZHAO(path,curvas,user):
	if not os.path.exists(path):
		curvas.pop(user,1)
		return
	df = pd.read_csv(path,sep='\t',comment='#')
	if len(df.index.values) <30:
		return
	A,B,Pi = bm.init(1, df['Nov'])
	A,B,Pi,k,count,W = bm.estimate(A,B,Pi,df['Nov'],1,niter=100)
	maior = 0
	m,k,sd_max,sd_inf,sd_sup = bm.data_from_beta_dist(A[0],B[0])
	maior = Pi[0]*beta.pdf(m,A[0],B[0])
	curvas[user] = [1,A,B,Pi,m,maior]


def main(window,raiz,curvas,user): 
	path =  raiz+'runs'+str(window)+'/da.tsv'
	pdf_ZHAO(path,curvas,user)

