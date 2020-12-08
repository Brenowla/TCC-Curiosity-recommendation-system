import os
import numpy as np
import pandas as pd
import math
from sympy import *
import numba
from scipy.stats import norm
from scipy.stats import beta
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import glob

#0.349066
#and abs(new-ant)>0.001

@numba.jit(nopython=True)
def Training_process_Xu(count,values,max_iterations,alpha):
	sum_s = values.size
	sr = 0
	sp = 1
	iter = 0
	ant = 0
	new = np.sum((-count/3076 + 1/(np.exp(-20*values + 20*sr) + 1) - 1/(np.exp(-20*values+ 20*sp) + 1))**2 )
	while new > 0 and ant!=new and iter<max_iterations:
		ant = new
		sr = sr - alpha*np.sum((-40*(-count/sum_s + 1/(np.exp(-20*values + 20*sr) + 1) - 1/(np.exp(-20*values + 20*sp) + 1))*np.exp(-20*values + 20*sr)/(np.exp(-20*values + 20*sr) + 1)**2))
		sp = sp - alpha*np.sum((40*(-count/sum_s + 1/(np.exp(-20*values + 20*sr) + 1) - 1/(np.exp(-20*values+ 20*sp) + 1))*np.exp(-20*values + 20*sp)/(np.exp(-20*values + 20*sp) + 1)**2))
		iter += 1
		new = np.sum((-count/3076 + 1/(np.exp(-20*values + 20*sr) + 1) - 1/(np.exp(-20*values+ 20*sp) + 1))**2)
	return sr,sp,new

def plot_U_shaped(values,vsr,vsp,count,path_save_fig):
	values.sort()
	sum_s = sum(c for c in count)
	s,sp,sr,count_s = symbols('s sp sr count_s')
	P = -1/(1+exp(-20*(s-sp)))
	P = lambdify([s,sp],P)
	R = 1/(1+exp(-20*(s-sr)))
	R = lambdify([s,sr],R)
	Cx_u = 1/(1+exp(-20*(s-sr)))-1/(1+exp(-20*(s-sp)))
	Cx_u = lambdify([s,sr,sp],Cx_u)
	C_u =  count_s/sum_s
	C_u = lambdify(count_s,C_u)
	x_values=values.copy()
	plt.xlabel('Stimulus Degree')
	plt.ylabel('Distribution')
	xticks=np.arange(0,1,0.02)
	yticks=np.arange(-1,1,0.1)
	plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
	plt.xticks(np.arange(0,1,0.1))
	plt.yticks(yticks)
	plt.grid(True,axis='both',linestyle=':')
	plt.xlim(0,1)
	yvalues = []
	for c in np.arange(0,1,0.02):
		yvalues.append(C_u(count[math.floor(c/0.02)]))
	plt.bar(xticks,yvalues,width = 0.02)
	#plt.ylim(-1,1)
	yvalues = [] 
	for c in np.arange(0,1,0.02):
		yvalues.append(Cx_u(c,vsr,vsp))
	plt.plot(np.arange(0,1,0.02),yvalues,'k--',lw=1.5)
	yvalues = []
	for c in np.arange(0,1,0.02):
		yvalues.append(P(c,vsp))
	plt.plot(np.arange(0,1,0.02),yvalues,'b--',lw=1.5)
	yvalues = []
	for c in np.arange(0,1,0.02):
		yvalues.append(R(c,vsr))
	plt.plot(np.arange(0,1,0.02),yvalues,'b--',lw=1.5)
	plt.draw()
	plt.savefig(path_save_fig,bbox_inches='tight')
	plt.close()

def pdf_Xu(path,path_save,user,curva):
	count = []
	df = pd.read_csv(path+'/pcl.txt',sep='\t')	
	for i in range(51):
		count.append(0)
	for s in df['cur'].values:
		count[math.floor(float(s)/0.02)] += 1
			
	count_s = []
	for s in df['cur'].values:
		count_s.append(count[math.floor(s/0.02)])

	sr,sp,erro = Training_process_Xu(np.array(count_s,dtype =  np.dtype(float)),np.array(df.cur.values, dtype =  np.dtype(float)),1000000,0.00001)
	maior = 0
	for i in df.cur.values:
		atual = 1/(1+np.exp(-20*(i-sr)))-1/(1+np.exp(-20*(i-sp)))
		if atual > maior:
			maior = atual
	curva[user] = [sr,sp,maior]

	plot_U_shaped(df.cur.values,sr,sp,count_s,path+'/'+user+'.png')



def main(window,raiz,curvas,user):
	path = raiz+'runs'+str(window)
	path_save =raiz+'runs'+str(window)+'/Curva_Xu/'

	if not os.path.exists(path_save+str(beta)+'/'):
		os.makedirs(path_save+str(beta)+'/')
	
	
	pdf_Xu(path,path_save,user,curvas)