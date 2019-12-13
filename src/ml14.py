import pandas as pd 
import numpy as np
import sys
import os
import random as rd
import matplotlib
matplotlib.use('Agg')
import multiprocessing as mtp
import matplotlib.pyplot as plt
import itertools as itr
from sklearn.utils import shuffle
from copy import *

def splitVec(vec, count):
    
    return 0
def rawImage_mtp(zip_args, maxX=64, maxY=64, nchannels=3, scale_delta=160.0):
    df=zip_args[1]
    index=zip_args[0]
    df_row=df.iloc[index]
    m = np.uint8(df_row['cX'])
    n = np.uint8(df_row['cY'])

    Imarr  = np.array(list(df_row['topoSig'])).astype(np.uint8)
    Imarr  = Imarr.reshape((n,m),order='C')
    Delarr_y = np.flipud(np.array(df_row['yDelta'].split(' ')).astype(np.float32)).reshape(n,1)
    Delarr_x = np.array(df_row['xDelta'].split(' ')).astype(np.float32).reshape(1,m)

    delta_x = Delarr_x.flatten().astype(int)
    delta_y = Delarr_y.flatten().astype(int)
    
    Imarr = np.repeat(Imarr, delta_x, axis=1)
    Imarr = np.repeat(Imarr, delta_y, axis=0)
    try:
        assert Imarr.shape[0]==320
        assert Imarr.shape[1]==320
    except:
        print Imarr.shape


    """
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
    """             

    return np.expand_dims(Imarr, axis=-1)
def AsquishImage_mtp(zip_args, maxX=64, maxY=64, nchannels=3, scale_delta=160.0):
    df=zip_args[1]
    index=zip_args[0]
    df_row=df.iloc[index]
    m = np.uint8(df_row['cX'])
    n = np.uint8(df_row['cY'])

    Imarr  = np.array(list(df_row['topoSig'])).astype(np.uint8)
    Imarr  = Imarr.reshape((n,m),order='C')
    Delarr_y = np.flipud(np.array(df_row['yDelta'].split(' ')).astype(np.float32)).reshape(n,1)
    Delarr_x = np.array(df_row['xDelta'].split(' ')).astype(np.float32).reshape(1,m)

    delta_x = Delarr_x.flatten()
    delta_y = Delarr_y.flatten()
    ave_x = scale_delta*2/maxX
    ave_y = scale_delta*2/maxY
    dup_x = (delta_x/ave_x).astype(int)+1
    dup_y = (delta_y/ave_y).astype(int)+1
    #print np.sum(dup_x)
    #print dup_x
    def post_processing(dup, delta):
        while not np.sum(dup)==maxX:
            old_dup=deepcopy(dup)
            num_to_reduce=np.sum(dup)-maxX
            merge_cdds=np.where(dup>1)[0]
            new_delta=delta/dup
            merge_delta=new_delta[merge_cdds]
            merge_idx=merge_cdds[merge_delta.argsort()[:num_to_reduce]]
            dup[merge_idx]-=1
        try:
            assert np.sum(dup)==maxX
        except:
            print np.sum(dup)
            print dup
            print np.sum(old_dup)
            print old_dup
        return dup

    if not np.sum(dup_x)==64:
        dup_x=post_processing(dup_x, delta_x)
    if not np.sum(dup_y)==64:
        dup_y=post_processing(dup_y, delta_y)

    new_delta_x = delta_x/dup_x
    new_delta_y = delta_y/dup_y
    
    Imarr = np.repeat(Imarr, dup_x, axis=1)
    Imarr = np.repeat(Imarr, dup_y, axis=0)
    delta_x= np.repeat(new_delta_x, dup_x, axis=0)
    delta_y= np.repeat(new_delta_y, dup_y, axis=0)

    assert len(delta_x)==maxX
    assert len(delta_y)==maxY



    delta_x= delta_x.reshape((1,len(delta_x)))
    delta_y= delta_y.reshape((len(delta_y),1))
    

    DelXrep = np.tile(delta_x,(maxY,1))/scale_delta
    DelYrep = np.tile(delta_y,(1,maxX))/scale_delta  

    squish = np.dstack((Imarr,DelXrep,DelYrep))        

    return squish


def squishImage_mtp(zip_args, maxX=64, maxY=64, nchannels=3, scale_delta=160.0):
    df=zip_args[1]
    index=zip_args[0]
    df_row=df.iloc[index]
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
    assert np.sum(delta_x)==320
    assert np.sum(delta_y)==320


    delta_x= delta_x.reshape((1,len(delta_x)))
    delta_y= delta_y.reshape((len(delta_y),1))
    

    DelXrep = np.tile(delta_x,(maxY,1))/scale_delta
    DelYrep = np.tile(delta_y,(1,maxX))/scale_delta  

    squish = np.dstack((Imarr,DelXrep,DelYrep))        

    return squish

class ML14:
    def __init__(self):
        self.scale_delta=160.
        self.threshold=0.85
        self.cpu_count=8
        self.use_image=False
        self.asquish=False
        self.val_hs_size=500
        self.val_nhs_size=5000
        self.getData()
        self.test_batch_pointer=0
        self.val_batch_pointer=0
        self.p=mtp.Pool(mtp.cpu_count())
    def saveImage(self, type, id, path):
        data, label = self.get_train_batch(10, balance=1)
        if type =='hs':
            topo=data[id,:,:,0]
            dx=data[id,:,:,1]*self.scale_delta
            dy=data[id,:,:,2]*self.scale_delta
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

    def squishImage(self, df_row, maxX=64,maxY=64,nchannels=3):
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
        assert np.sum(delta_x)==320
        assert np.sum(delta_y)==320


        delta_x= delta_x.reshape((1,len(delta_x)))
        delta_y= delta_y.reshape((len(delta_y),1))
        

        DelXrep = np.tile(delta_x,(maxY,1))/self.scale_delta
        DelYrep = np.tile(delta_y,(1,maxX))/self.scale_delta  

        squish = np.dstack((Imarr,DelXrep,DelYrep))        

      
        return squish

    def getData(self, channel=3):
        mX=64
        mY=64
        self.udf_train_hs = pd.DataFrame()
        self.udf_test_hs = pd.DataFrame()
        self.udf_train_nhs = pd.DataFrame()
        self.udf_test_nhs=pd.DataFrame()

        train_path = './data/ml14/train/De.msgpack'
        test_path = './data/ml14/test/Dv.msgpack'


        def process(data, threshold=0.85):
            df=pd.read_msgpack(data)
            score=df.score.values
            idx_hs=np.where(score>threshold)[0]
            idx_nhs=np.where(score<=threshold)[0]
            hs_df=df.iloc[idx_hs]
            nhs_df=df.iloc[idx_nhs]

            return hs_df.reset_index(drop=True), nhs_df.reset_index(drop=True)
        
        self.train_hs_df, self.train_nhs_df=process(train_path)
        self.train_hs_df=shuffle(shuffle(self.train_hs_df))
        self.train_nhs_df=shuffle(shuffle(self.train_nhs_df))
      
        self.test_hs_df, self.test_nhs_df=process(test_path)

        self.val_hs_df = self.train_hs_df[:self.val_hs_size].reset_index(drop=True)
        self.train_hs_df=self.train_hs_df[self.val_hs_size:].reset_index(drop=True)
        #print self.val_hs_df, self.train_hs_df, self.test_hs_df, self.test_nhs_df
        self.val_nhs_df = self.train_nhs_df[:self.val_nhs_size].reset_index(drop=True)
        self.train_nhs_df=self.train_nhs_df[self.val_nhs_size:].reset_index(drop=True)

        self.train_hs_length=self.train_hs_df.shape[0]
        self.train_nhs_length=self.train_nhs_df.shape[0]
        self.test_hs_length=self.test_hs_df.shape[0]
        self.test_nhs_length=self.test_nhs_df.shape[0]
        print self.train_hs_df.shape, self.train_nhs_df.shape
        quit()
    def df2tensor(self, df_rows):
        indices=range(df_rows.shape[0])
        df_rows=df_rows.reset_index(drop=True)
        squishArrList=[]
        if self.use_image:
            squishArrList.append(self.p.map(rawImage_mtp, zip(indices, itr.repeat(df_rows))))
        elif self.asquish:
            #AsquishImage_mtp([0, df_rows])
            squishArrList.append(self.p.map(AsquishImage_mtp, zip(indices, itr.repeat(df_rows))))
        else:
            squishArrList.append(self.p.map(squishImage_mtp, zip(indices, itr.repeat(df_rows))))
        a=np.stack(squishArrList)
        return np.stack(squishArrList)
    def get_train_batch(self, half_batch_size, mode='detection', balance=1):
        batch_size=2*half_batch_size
        if mode == 'detection':
                hs_batch_size=batch_size/(balance+1)
                nhs_batch_size=balance*batch_size/(balance+1)
                idx_hs= rd.sample(np.arange(self.train_hs_df.shape[0]), hs_batch_size)
                idx_nhs= rd.sample(np.arange(self.train_nhs_df.shape[0]), nhs_batch_size)
                train_hs = self.df2tensor(self.train_hs_df.iloc[idx_hs])[0]
                train_nhs = self.df2tensor(self.train_nhs_df.iloc[idx_nhs])[0]
                train = np.concatenate((train_hs, train_nhs), axis=0)
                train_label=np.concatenate((np.ones(hs_batch_size), np.zeros(nhs_batch_size)), axis=0)
        return train, train_label
    
    def get_test_batch(self, batch_size=300, mode='detection', dtype='hs'):
        if mode == 'detection':
            if dtype=='hs':
                if self.test_batch_pointer+batch_size<self.test_hs_length:
                    batch_df= self.test_hs_df.iloc[self.test_batch_pointer:self.test_batch_pointer+batch_size]
                    self.test_batch_pointer+=batch_size
                else:
                    batch_df= self.test_hs_df.iloc[self.test_batch_pointer:self.test_hs_length]
                    self.test_batch_pointer=0
            if dtype=='nhs':
                if self.test_batch_pointer+batch_size<self.test_nhs_length:
                    batch_df= self.test_nhs_df.iloc[self.test_batch_pointer:self.test_batch_pointer+batch_size]
                    self.test_batch_pointer+=batch_size
                else:
                    batch_df= self.test_nhs_df.iloc[self.test_batch_pointer:self.test_nhs_length]
                    self.test_batch_pointer=0
            return self.df2tensor(batch_df)[0]

    def get_val_batch(self, batch_size=300, mode='detection', dtype='hs'):
        if dtype=='hs':
            if self.val_batch_pointer+batch_size<self.val_hs_size:
                batch_df= self.val_hs_df[self.val_batch_pointer:self.val_batch_pointer+batch_size]
                self.val_batch_pointer+=batch_size
            else:
                batch_df= self.val_hs_df[self.val_batch_pointer:self.val_hs_size]
                self.val_batch_pointer=0
        if dtype=='nhs':
            if self.val_batch_pointer+batch_size<self.val_nhs_size:
                batch_df= self.val_nhs_df[self.val_batch_pointer:self.val_batch_pointer+batch_size]
                self.val_batch_pointer+=batch_size
            else:
                batch_df= self.val_nhs_df[self.val_batch_pointer:self.val_nhs_size]
                self.val_batch_pointer=0
        
        """
        if dtype=='nhs':
            if self.val_batch_pointer+batch_size<self.train_nhs_length:
                batch= self.train_nhs[self.val_batch_pointer:self.val_batch_pointer+batch_size]
                self.val_batch_pointer+=batch_size
            else:
                batch= self.train_nhs[self.val_batch_pointer:self.train_nhs_length]
                self.val_batch_pointer=0
        """
        return self.df2tensor(batch_df)[0]
