import pandas as pd 
import numpy as np
import sys
import os
import tensorflow as tf 
import random as rd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

class ICCAD:
    def __init__(self):
        self.scale_delta=600.0
        self.val_hs_size=200
        self.getData()
        self.test_batch_pointer=0
        self.val_batch_pointer=0

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

    def squishImage(self, df_row,maxX,maxY,nchannels=3):
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
        assert np.sum(delta_x)==1200
        assert np.sum(delta_y)==1200


        delta_x= delta_x.reshape((1,len(delta_x)))
        delta_y= delta_y.reshape((len(delta_y),1))
        

        DelXrep = np.tile(delta_x,(maxY,1))/self.scale_delta
        DelYrep = np.tile(delta_y,(1,maxX))/self.scale_delta  

        squish = np.dstack((Imarr,DelXrep,DelYrep))        

      

        """
        if padX%2 > 0:
            padX = (padX+1)/2
            offX = 1
        else:
            padX = padX/2
            offX = 0
        if padY%2 > 0:
            padY = (padY+1)/2
            offY = 1
        else:
            padY = padY/2
            offY = 0
        DelX = np.zeros((maxY,1),dtype=np.uint16)
        DelY = np.zeros((1,maxX),dtype=np.uint16).T
        
        DelX[padX:maxY-padX+offX] = Delarr_x
        DelY[padY:maxX-padY+offY] = Delarr_y.T
        
    
        Topo = np.zeros((maxY,maxX),dtype=np.uint8)
        Topo[padX:maxY-padX+offX,padY:maxX-padY+offY] = Imarr
        
        if nchannels == 2:
            Del  = np.multiply(DelX,DelY.T).astype(np.uint16)
            squish = np.dstack((Topo,Del))
        elif nchannels == 3:
            DelXrep = np.tile(DelX,(1,maxX))/600.
            DelYrep = np.tile(DelY.T,(maxY,1))/600.  
            squish = np.dstack((Topo,DelXrep,DelYrep))
        """
        return squish

    def getData(self, channel=3):
        mX=64
        mY=64
        self.udf_train_hs = pd.DataFrame()
        self.udf_test_hs = pd.DataFrame()
        for i in xrange(2,6):
            hs1_train_path = './data/iccad/train'+str(i)+ '_hs1.msgpack'
            hs2_train_path = './data/iccad/train'+str(i)+ '_hs2.msgpack'
            hs1_test_path = './data/iccad/test'+str(i)+ '_hs1.msgpack'
            hs2_test_path = './data/iccad/test'+str(i)+ '_hs2.msgpack'
            for j in [hs1_train_path, hs2_train_path]:
                try:
                    tmpdf = pd.read_msgpack(j)
                except:
                    print("file %s not found, ignored"%j)
                    continue
                self.udf_train_hs = self.udf_train_hs.append(tmpdf)
            for j in [hs1_test_path, hs2_test_path]:
                try:
                    tmpdf = pd.read_msgpack(j)
                except:
                    print("file %s not found, ignored"%j)
                    continue
                self.udf_test_hs = self.udf_test_hs.append(tmpdf)

        squishArrList = []
        for i in np.arange(self.udf_train_hs.shape[0]):
            squishArrList.append(self.squishImage(self.udf_train_hs.iloc[i],mX,mY,channel))
        self.train_hs = np.stack(squishArrList)
        for i in np.arange(self.udf_test_hs.shape[0]):
            squishArrList.append(self.squishImage(self.udf_test_hs.iloc[i],mX,mY,channel))
        self.test_hs = np.stack(squishArrList)
        np.random.shuffle(self.train_hs)
        self.val_hs=self.train_hs[0:self.val_hs_size]
        self.train_hs=self.train_hs[self.val_hs_size:]
        

        self.udf_train_nhs = pd.DataFrame()
        for i in xrange(2,6):
            nhs_train_path = './data/iccad/train'+str(i)+ '_nhs.msgpack'

            try:
                tmpdf = pd.read_msgpack(nhs_train_path)
            except:
                print("file %s not found, ignored"%nhs_train_path)
                continue
            self.udf_train_nhs = self.udf_train_nhs.append(tmpdf)
        squishArrList = []
        for i in np.arange(self.udf_train_nhs.shape[0]):
            squishArrList.append(self.squishImage(self.udf_train_nhs.iloc[i],mX,mY,channel))
        self.train_nhs = np.stack(squishArrList)

        self.udf_test_nhs = pd.DataFrame()
        for i in xrange(2,6):
            nhs_test_path = './data/iccad/test'+str(i)+ '_nhs.msgpack'
            try:
                tmpdf = pd.read_msgpack(nhs_test_path)
            except:
                print("file %s not found, ignored"%nhs_test_path)
                continue
            self.udf_test_nhs = self.udf_test_nhs.append(tmpdf)
        squishArrList = []
        for i in np.arange(self.udf_test_nhs.shape[0]):
            try:
                squishArrList.append(self.squishImage(self.udf_test_nhs.iloc[i],mX,mY,channel))
            except:
                continue
        self.test_nhs = np.stack(squishArrList)


        self.train_hs_length=len(self.train_hs)
        self.train_nhs_length=len(self.train_nhs)
        self.test_hs_length=len(self.test_hs)
        self.test_nhs_length=len(self.test_nhs)
    def get_train_batch(self, half_batch_size, mode='detection', balance=False):
        if mode == 'detection':
            train_hs = rd.sample(self.train_hs, half_batch_size)
            train_nhs = rd.sample(self.train_nhs, half_batch_size)
            train = np.concatenate((train_hs, train_nhs), axis=0)
            train_label=np.concatenate((np.ones(half_batch_size), np.zeros(half_batch_size)), axis=0)
            return train, train_label
    
    def get_test_batch(self, batch_size=300, mode='detection', dtype='hs'):
        if mode == 'detection':
            if dtype=='hs':
                if self.test_batch_pointer+batch_size<self.test_hs_length:
                    batch= self.test_hs[self.test_batch_pointer:self.test_batch_pointer+batch_size]
                    self.test_batch_pointer+=batch_size
                else:
                    batch= self.test_hs[self.test_batch_pointer:self.test_hs_length]
                    self.test_batch_pointer=0
            if dtype=='nhs':
                if self.test_batch_pointer+batch_size<self.test_nhs_length:
                    batch= self.test_nhs[self.test_batch_pointer:self.test_batch_pointer+batch_size]
                    self.test_batch_pointer+=batch_size
                else:
                    batch= self.test_nhs[self.test_batch_pointer:self.test_nhs_length]
                    self.test_batch_pointer=0
            return batch
    
    def get_val_batch(self, batch_size=300, mode='detection', dtype='hs'):
        if mode == 'detection':
            return self.val_hs
            """
            if dtype=='hs':
                if self.val_batch_pointer+batch_size<self.train_hs_length:
                    batch= self.train_hs[self.val_batch_pointer:self.val_batch_pointer+batch_size]
                    self.val_batch_pointer+=batch_size
                else:
                    batch= self.train_hs[self.val_batch_pointer:self.train_hs_length]
                    self.val_batch_pointer=0
            if dtype=='nhs':
                if self.val_batch_pointer+batch_size<self.train_nhs_length:
                    batch= self.train_nhs[self.val_batch_pointer:self.val_batch_pointer+batch_size]
                    self.val_batch_pointer+=batch_size
                else:
                    batch= self.train_nhs[self.val_batch_pointer:self.train_nhs_length]
                    self.val_batch_pointer=0
            return batch
            """




        






