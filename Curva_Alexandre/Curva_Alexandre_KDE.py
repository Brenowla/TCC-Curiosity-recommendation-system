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
    kde = bm.KDE(df['S_b'])
    maior = 0
    atp = 0
    for v in df['S_b'].values:
        v_k = kde.evaluate(v)
        if v_k > maior:
            maior = float(v_k)
            atp = float(v)
    curvas[user] = [kde,maior,atp]
    return curvas


def main(window,raiz,curvas,user): 
    path_estimulus =  raiz+'runs'+str(window)+'/da_all.tsv'
    path =  raiz+'runs'+str(window)+'/da.tsv'
    pdf_Alexandre(path,path_estimulus,user,curvas)
