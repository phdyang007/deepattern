import pandas as pd 
import numpy as np
import sys
import os
import random as rd
import matplotlib
matplotlib.use('Agg')
import multiprocessing as mtp
import matplotlib.pyplot as plt
from progress.bar import Bar
import time
import itertools as itr
from sklearn.utils import shuffle

def squishImage_mtp(zip_args, maxX=16, maxY=16, nchannels=3, scale_delta=96.0):
    df=zip_args[1]
    index=zip_args[0]
    df_row=df.iloc[index]
    m = np.uint8(df_row['cX'])
    n = np.uint8(df_row['cY'])

    Imarr  = np.array(list(df_row['topoSig'])).astype(np.uint8)
    Imarr  = Imarr.reshape((n,m),order='C')
    
    topo=np.zeros((maxX, maxY)).astype(float)
    sx=int((maxX-m)/2.0)
    sy=int((maxY-n)/2.0)

    topo[sy:sy+n,sx:sx+m]=Imarr
    try:
        assert topo.shape[0]==maxY
        assert topo.shape[1]==maxX
    except:
        print (topo.shape)
        quit()
    return np.expand_dims(topo, axis=-1)

class EUV:
    def __init__(self, reduce_nhs=False, sample_train_from_test=False):

        self.cpu_count=8

        self.getDataDP()

        self.test_batch_pointer=0

        self.p=mtp.Pool(self.cpu_count)
        self.train_batch_pointer=0
    def saveImage(self, type, id, path):
        if type =='hs':
            topo=self.test_hs[id,:,:,0]
            dx=self.test_hs[id,:,:,1]*self.scale_delta
            dy=self.test_hs[id,:,:,2]*self.scale_delta
            plt.imshow(topo,interpolation="none", cmap='gray')
            plt.savefig(path+'topo.pdf')
            plt.clf()
            plt.close()
            plt.imshow(dx,interpolation="none", cmap='gray')
            plt.savefig(path+'dx.pdf')
            plt.clf()
            plt.close()
            plt.imshow(dy,interpolation="none", cmap='gray')
            plt.savefig(path+'dy.pdf')
            plt.clf()
            plt.close()
    def printExactPattern(self, df_row):
        m = int(df_row['cX'])
        n = int(df_row['cY'])
        Imarr  = np.array(list(df_row['topoSig'])).astype(int)
        Imarr  = Imarr.reshape((n,m),order='C')
        
        sx = int(n)
        sy = int(m)
        Delarr_x = np.flipud(np.array(df_row['yDelta'].split(' ')).astype(int))
        Delarr_y = np.array(df_row['xDelta'].split(' ')).astype(int)
        
        nx = int(sum(Delarr_x))
        ny = int(sum(Delarr_y))

        s  = (nx,ny)
        
        Dx = Delarr_x
        Dy = Delarr_y
        #print("Imarr size",Imarr.shape)
        b = np.tile(Imarr[0,:],(Dx[0],1))
        for i in np.arange(1,len(Dx)):
            b = np.concatenate((b,np.tile(Imarr[i,:],(Dx[i],1))),axis=0)
        #print("b size",b.shape)
        c = np.tile(b[:,0],(Dy[0],1)).T
        for i in np.arange(1,len(Dy)):
            c = np.concatenate((c,np.tile(b[:,i],(Dy[i],1)).T),axis=1)
        #print(c.shape)
        fig = plt.figure()
        img = plt.imshow(c,interpolation="none")
        img.set_cmap('nipy_spectral')
        #outDir = os.path.abspath("./imageDb")
        #if not os.path.isdir(outDir):
        #    os.mkdir(outDir)
        #fig.savefig(outDir+'/'+'label_'+str(df_row[labelName])+'_'+str(df_row['ruleHash'])+'.png')
        #plt.close(fig)
        
        print("Size of pattern - ",c.shape)
        print("Kbytes for pattern - ",c.nbytes/1024.)

    def squishImage(self, df_row,maxX=64,maxY=64,nchannels=3):
        m = np.uint8(df_row['cX'])
        n = np.uint8(df_row['cY'])

        Imarr  = np.array(list(df_row['topoSig'])).astype(np.uint8)
        Imarr  = Imarr.reshape((n,m),order='C')
        Delarr_y = np.flipud(np.array(df_row['yDelta'].split(' ')).astype(np.float32)).reshape(n,1)
        Delarr_x = np.array(df_row['xDelta'].split(' ')).astype(np.float32).reshape(1,m)

        delta_x = Delarr_x.flatten()
        delta_y = Delarr_y.flatten()

        assert m<=maxX and n<=maxY
        while len(delta_x)<maxX or len(delta_y)<maxY:
            if len(delta_x)<maxX:
                dup_x = np.ones(len(delta_x)).astype(np.int8)
                split_idx = np.argmax(delta_x)
                delta_x[split_idx]/=2.0
                dup_x[split_idx]+=1
                Imarr=np.repeat(Imarr, dup_x, axis=1)
                delta_x= np.repeat(delta_x, dup_x, axis=0)
                
            if len(delta_y)<maxY:
                dup_y = np.ones(len(delta_y)).astype(np.int8) 
                split_idx = np.argmax(delta_y)
                delta_y[split_idx]/=2.0
                dup_y[split_idx]+=1
                Imarr=np.repeat(Imarr, dup_y, axis=0)
                delta_y= np.repeat(delta_y, dup_y, axis=0)
                      

        assert len(delta_x)==maxX
        assert len(delta_y)==maxY
        assert np.sum(delta_x)==192
        assert np.sum(delta_y)==192


        delta_x= delta_x.reshape((1,len(delta_x)))
        delta_y= delta_y.reshape((len(delta_y),1))
        

        DelXrep = np.tile(delta_x,(maxY,1))/self.scale_delta
        DelYrep = np.tile(delta_y,(1,maxX))/self.scale_delta  

        squish = np.dstack((Imarr,DelXrep,DelYrep))        

        return squish

    def getDataDP(self):
        train_path = './data/train.msgpack'
        test_path = './data/test.msgpack'
        self.train_df = pd.read_msgpack(train_path)
        self.test_df = pd.read_msgpack(test_path)
        self.train_length = self.train_df.shape[0]
        self.test_length = self.test_df.shape[0]
    
    def getTrainBatchDP(self, batch_size):
        idx=rd.sample(list(np.arange(self.train_df.shape[0])), batch_size)

        train=self.df2tensor(self.train_df.iloc[idx])[0]


        return train
    def getTestBatchDP(self, batch_size):

        idx=rd.sample(list(np.arange(self.test_df.shape[0])), batch_size)

        test=self.df2tensor(self.test_df.iloc[idx])[0]


        return test

    def df2tensor(self, df_rows):
        indices=range(df_rows.shape[0])
        df_rows=df_rows.reset_index(drop=True)
        squishArrList=[]
        squishArrList.append(self.p.map(squishImage_mtp, zip(indices, itr.repeat(df_rows))))
        return np.stack(squishArrList)
    





