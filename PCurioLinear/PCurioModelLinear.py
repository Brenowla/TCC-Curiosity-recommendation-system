#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:26:02 2018

@author: alexandre
"""

import os
import glob
import numpy as np
import time as Time
import pandas as pd
from PCurioLinear.logger import Logger
from PCurioLinear.util import binary_search
from multiprocessing import Process

def add_dict(d,key):
    if key in d.keys():
        d[key]+=1
    else:
        d[key]=1


def add_dict_time(d,d2,key,value):
    if key in d.keys():
        d2[key]=d[key]
    else:
        d2[key]=value
    d[key]=value

def calculateStimulus(count,line,artist_genre,user_id,artist,track,time,TIME_WINDOW,LOWER_BOUND_EVENTS,last_time,queue_art,queue_track,queue_genres,queue_time,dict_art,dict_track,dict_genres,dart,dtrack,dgenres,dart_time,dtrack_time,dgenres_time,ldart,ldtrack,ldgenres,dict_art_time,dict_track_time,dict_genres_time,ldart_time,ldtrack_time,ldgenres_time):
    count+=1
    store_line =''
    size=len(queue_time)
    if size > 0:
        last_time=queue_time[-1]
                
    if size == 0 or time - last_time <= TIME_WINDOW:
        queue_art.append(artist)
        queue_track.append(track)
        queue_time.append(time)
        queue_genres.append(artist_genre[artist])
                
        add_dict(dict_art,artist)
        add_dict(dict_track,track)
        for genre in artist_genre[artist]:
            add_dict(dict_genres,genre)

        '''
        ----------------------------------------------------------------
        '''

        add_dict_time(dict_art_time,dart_time,artist,time)
        add_dict_time(dict_track_time,dtrack_time,track,time)
        for genre in artist_genre[artist]:
            add_dict_time(dict_genres_time,dgenres_time,genre,time)

        ldart_time.append(dict_art_time.copy())
        ldtrack_time.append(dict_track_time.copy())
        ldgenres_time.append(dict_genres_time.copy())
        '''
        ----------------------------------------------------------------
        '''
                    
        ldart.append(dict_art.copy())
        ldtrack.append(dict_track.copy())
        ldgenres.append(dict_genres.copy())
                
        if size > 2:
            i=binary_search(queue_time,TIME_WINDOW) 
            if i > 0:
                del queue_art[:i]
                del queue_track[:i]
                del queue_time[:i]
                del queue_genres[:i]
                        
                for key in ldart[i-1].keys():
                    if key in dict_art.keys():
                                dart[key]=ldart[i-1][key]
                                    
                for key in ldtrack[i-1].keys():
                    if key in dict_track.keys():
                                dtrack[key]=ldtrack[i-1][key]
                                    
                for key in ldgenres[i-1].keys():
                    if key in dict_genres.keys():
                                dgenres[key]=ldgenres[i-1][key]
                #----------------------------------------------------
                for key in ldart_time[i-1].keys():
                    if key in dict_art_time.keys():
                        dart_time[key]=ldart_time[i-1][key]
                                    
                for key in ldtrack_time[i-1].keys():
                    if key in dict_track_time.keys():
                        dtrack_time[key]=ldtrack_time[i-1][key]
                                    
                for key in ldgenres_time[i-1].keys():
                    if key in dict_genres_time.keys():
                        dgenres_time[key]=ldgenres_time[i-1][key]
                #----------------------------------------------------


                del ldart[:i]
                del ldtrack[:i]
                del ldgenres[:i]

                #----------------------------------------------------
                del ldart_time[:i]
                del ldtrack_time[:i]
                del ldgenres_time[:i]
                #----------------------------------------------------

    else:
        queue_time=[time]
        queue_art=[artist]
        queue_track=[track]
        queue_genres=[artist_genre[artist]]
                
        dict_art={}
        dict_track={}
        dict_genres={}

        #--------------------------------------------------------------
        dict_art_time={}
        dict_track_time={}
        dict_genres_time={}
        #--------------------------------------------------------------
                
        dart={}
        dtrack={}
        dgenres={}
                
        #--------------------------------------------------------------
        dart_time={}
        dtrack_time={}
        dgenres_time={}
        #--------------------------------------------------------------

        ldart=[]
        ldtrack=[]
        ldgenres=[]

        #--------------------------------------------------------------
        ldart_time=[]
        ldtrack_time=[]
        ldgenres_time=[]
        #--------------------------------------------------------------
                
        add_dict(dict_art,artist)
        add_dict(dict_track,track)
        for genre in artist_genre[artist]:
            add_dict(dict_genres,genre)

        #--------------------------------------------------------------
        add_dict(dict_art_time,artist)
        add_dict(dict_track_time,track)
        for genre in artist_genre[artist]:
            add_dict(dict_genres_time,genre)
        #--------------------------------------------------------------

        ldart.append(dict_art.copy())
        ldtrack.append(dict_track.copy())
        ldgenres.append(dict_genres.copy())

        #--------------------------------------------------------------
        ldart_time.append(dict_art_time.copy())
        ldtrack_time.append(dict_track_time.copy())
        ldgenres_time.append(dict_genres_time.copy())
        #--------------------------------------------------------------

    size=len(queue_time)
    if size > LOWER_BOUND_EVENTS:# storing the records in the file!!!
        A=count
        B=size
        if artist in dict_art.keys():
            if not artist in dart.keys():
                C=dict_art[artist]-1 #len(set(queue_art))
            else:
                C=dict_art[artist]-dart[artist]-1
        else:
            C=0
        if track in dict_track.keys():
            if not track in dtrack.keys():
                D=dict_track[track]-1#len(set(queue_track))
            else:
                D=dict_track[track]-dtrack[track]-1
        else:
            D=0
        arr_art=np.array(queue_art)
        #----------------------------------------------------------
        if artist in dict_art_time.keys():
            if artist in dart_time.keys():
                if time != dart_time[artist]:
                    E=dart_time[artist] 
                else:
                    E=0
            else:
                E=0
        else:
            E=0

        if track in dict_track_time.keys():
            if track in dtrack_time.keys():
                if time != dtrack_time[track]:
                    F=dtrack_time[track]
                else:
                    F=0
            else:
                F=0
        else:
            F=0
        #----------------------------------------------------------
        arr_track=np.array(queue_track)
            
        I=len(artist_genre[artist])
        J=[str(dict_genres[k]-dgenres[k]-1)
                if k in dgenres.keys()
            else str(dict_genres[k]-1)
            for k in artist_genre[artist]
                if k in dict_genres.keys()]
        #-------------------------------------------------------------------------
        K=[]
        for genre in artist_genre[artist]:
            if genre in dgenres_time.keys() and genre in dict_genres_time.keys() and time != dgenres_time[genre]:
                K.append(dgenres_time[genre])
            else:
                K.append(0)
        #-------------------------------------------------------------------------
        L=[str(e) for e in artist_genre[artist]]
        N=[]
        for k in dict_genres.keys():
            if not k in dgenres.keys():
                if dict_genres[k] != 0:
                    N.append(str(k))
            elif dict_genres[k]-dgenres[k] != 0:
                N.append(str(k))
                        
        O=[dict_genres[k]-dgenres[k]
                if k in dgenres.keys()
            else dict_genres[k]
                for k in dict_genres.keys()]
        P=np.array(O,dtype=np.int)
        P=P[P!=0]
        O=P
        H=P.size
        M=H
        P=P/P.sum()
        h=-P*np.log2(P)
        G=h.sum()
        store_line+='%s\t%d\t%d\t%d\t%d\t%d\t%d\t%.2f\t%d\t%d\t'%\
                    (user_id[:-4],A,B,C,D,E,F,G,H,I)
        store_line+=','.join(J)+'\t'
        store_line+=','.join(('%d'%(k)) for k in K)+'\t'
        store_line+=','.join(L)+'\t'
        store_line+='%d\t'%(M)
        store_line+=','.join(N)+'\t'
        store_line+=','.join(('%d'%(o)) for o in O)+'\t'
        store_line+=','.join(('%.2f'%(p)) for p in P)+'\t'
        store_line+='%d\t%s\n'%(time,track)
    return store_line,count,last_time
                    
def computeStimulusDegree(fin, fout, artist_genre,track_artist,user_id,path,window,tipo):
    if not os.path.exists(os.getcwd() + "/logs/"):
        os.makedirs(os.getcwd() + "/logs/")
    logger = Logger(os.getcwd() + "/logs")
    logger.add_log({'empty':'key.empty'})
    fout.write("#user\tA\tB\tC\tD\tE\tF\tG\tH\tI\tJ\tK\tL\tM\tN\tO\tP\ttimestamp\ttrack\n")               
    count=0
    if tipo == 0:
        TIME_WINDOW=24*3600
    else:
        TIME_WINDOW = 3.5*24*3600
    LOWER_BOUND_EVENTS=10
    last_time=0
    queue_art=[]
    queue_track=[]
    queue_genres=[]
    queue_time=[]
    dict_art={}
    dict_track={}
    dict_genres={}
    dart={}
    dtrack={}
    dgenres={}
    ldart=[]
    ldtrack=[]
    ldgenres=[]
    '''
    --------------------------------------------------------------------------
    '''
    dict_art_time={}
    dict_track_time={}
    dict_genres_time={}

    dart_time={}
    dtrack_time={}
    dgenres_time={}

    ldart_time=[]
    ldtrack_time=[]
    ldgenres_time=[]
    '''
    -------------------------------------------------------------------------
    '''
    store_line=''
    for line in fin:
        row=line.split('\t')
        artist,track,time=int(row[1]),int(row[3]),int(row[4][:-1]) 
        add_line,count,last_time = calculateStimulus(count,line,artist_genre,user_id,artist,track,time,TIME_WINDOW,LOWER_BOUND_EVENTS,last_time,queue_art,queue_track,queue_genres,queue_time,dict_art,dict_track,dict_genres,dart,dtrack,dgenres,dart_time,dtrack_time,dgenres_time,ldart,ldtrack,ldgenres,dict_art_time,dict_track_time,dict_genres_time,ldart_time,ldtrack_time,ldgenres_time)
        store_line += add_line
    
    fout.write(store_line)
    fout.flush()
    
    
    fout=open(path+'/runs'+str(window)+"/pcl_all.txt",'w+')

    fout.write("#user\tA\tB\tC\tD\tE\tF\tG\tH\tI\tJ\tK\tL\tM\tN\tO\tP\ttimestamp\ttrack\n")
    fout.flush()
    store_line = ''

    for track in track_artist.keys():
        artist=track_artist[track]
        add_line,c,l = calculateStimulus(count,line,artist_genre,user_id,artist,track,time,TIME_WINDOW,LOWER_BOUND_EVENTS,last_time,queue_art.copy(),queue_track.copy(),queue_genres.copy(),queue_time.copy(),dict_art.copy(),dict_track.copy(),dict_genres.copy(),dart.copy(),dtrack.copy(),dgenres.copy(),dart_time.copy(),dtrack_time.copy(),dgenres_time.copy(),ldart.copy(),ldtrack.copy(),ldgenres.copy(),dict_art_time.copy(),dict_track_time.copy(),dict_genres_time.copy(),ldart_time.copy(),ldtrack_time.copy(),ldgenres_time.copy())
        store_line += add_line
    fout.write(store_line)
    fout.flush()

    logger.log_done(str(user_id)) 

def main(window,raiz,tipo,user,artist_genre,track_artist,path_source):
    path_artists=raiz
    path=path_artists
    _PATH=path
    
    tracks = {}
    fin = open(path_source+user)
    for line in fin:
        row=line.split('\t')
        tracks[int(row[0])] = track_artist[int(row[0])]

    if not os.path.exists(path):
        os.makedirs(path)  

    fin=open(path+'/runs'+str(window)+'/history/'+user,'r')
    fout=open(path+'/runs'+str(window)+"/pcl.txt",'w+')
    
    computeStimulusDegree(fin, fout, artist_genre,tracks,user,path,window,tipo)

    fin.close()
    fout.close()