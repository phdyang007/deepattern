import pandas as pd 
import numpy as np
import sys
import os
import random as rd
import matplotlib
matplotlib.use('Agg')
import multiprocessing as mtp
import matplotlib.pyplot as plt
import time
import itertools as itr
from sklearn.utils import shuffle
def squishImage_mtp(zip_args, maxX=32, maxY=32, nchannels=3, scale_delta=96.0):
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
    assert np.sum(delta_x)==192
    assert np.sum(delta_y)==192


    delta_x= delta_x.reshape((1,len(delta_x)))
    delta_y= delta_y.reshape((len(delta_y),1))
    

    DelXrep = np.tile(delta_x,(maxY,1))/scale_delta
    DelYrep = np.tile(delta_y,(1,maxX))/scale_delta  

    squish = np.dstack((Imarr,DelXrep,DelYrep))        

    return squish

class EUV:
    def __init__(self, reduce_nhs=True):
        self.scale_delta=96
        self.cpu_count=8
        self.val_hs_size=500
        self.val_nhs_size=5001
        self.getData()
        if reduce_nhs:
            self.reduce_nhs()
        self.test_batch_pointer=0
        self.val_batch_pointer=0
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

    def getData(self, channel=3):
        mX=64
        mY=64
        self.udf_train_hs = pd.DataFrame()
        self.udf_test_hs = pd.DataFrame()
        self.udf_train_nhs = pd.DataFrame()
        self.udf_test_nhs=pd.DataFrame()

        data3_path = './data/euv/train/df_tl3.msgpack'
        data4_path = './data/euv/train/df_tl4.msgpack'
        data3_label = './data/euv/train/label_tl3.txt'
        data4_label = './data/euv/train/label_tl4.txt'
        data1_path = './data/euv/test/df_tl1.msgpack'
        data2_path = './data/euv/test/df_tl2.msgpack'
        data1_label = './data/euv/test/label_tl1.txt'
        data2_label = './data/euv/test/label_tl2.txt'
        ###Setting 1### 
        #train_path=[train1_path, train2_path]
        #train_label=[train1_label, train2_label]
        #test_path=[test1_path, test2_path]
        #test_label=[test1_label, test2_label]
        ###
        ###Setting 2###
        test_path=[data1_path, data2_path, data3_path]
        test_label=[data1_label, data2_label, data3_label]
        train_path=[data4_path]
        train_label=[data4_label]
        
        ###
        def process(self, path, label):
            udf=pd.DataFrame()
            lbudf=pd.DataFrame()
            for i in xrange(len(path)):
                udf=udf.append(pd.read_msgpack(path[i]))
                lbudf=lbudf.append(pd.read_csv(label[i], sep='\t', skiprows=1, header=None))

            labelf=lbudf[0].to_frame()

            udf=udf.merge(labelf, how='left', left_on='ruleHash', right_on=0)
            tmplabels=udf[0].values
            idx_nhs=np.where(pd.isnull(tmplabels)==True)[0]
            idx_hs=np.where(pd.isnull(tmplabels)==False)[0]

            hs_df=udf.iloc[idx_hs]
            nhs_df=udf.iloc[idx_nhs]



            return hs_df.reset_index(drop=True), nhs_df.reset_index(drop=True)

        self.train_hs_df, self.train_nhs_df=process(self, train_path,train_label)
        self.train_hs_df=shuffle(self.train_hs_df)
        self.train_nhs_df=shuffle(self.train_nhs_df)
      
        self.test_hs_df, self.test_nhs_df=process(self, test_path, test_label)

        self.val_hs_df = self.train_hs_df[:self.val_hs_size].reset_index(drop=True)
        self.train_hs_df=self.train_hs_df[self.val_hs_size:].reset_index(drop=True)
        #print self.val_hs_df, self.train_hs_df, self.test_hs_df, self.test_nhs_df
        self.val_nhs_df = self.train_nhs_df[:self.val_nhs_size].reset_index(drop=True)
        self.train_nhs_df=self.train_nhs_df[self.val_nhs_size:].reset_index(drop=True)


        self.train_hs_length=self.train_hs_df.shape[0]
        self.train_nhs_length=self.train_nhs_df.shape[0]
        self.test_hs_length=self.test_hs_df.shape[0]
        self.test_nhs_length=self.test_nhs_df.shape[0]
        self.train_df=self.train_hs_df.append(self.train_nhs_df)
        tmp_labels=np.concatenate((np.ones(self.train_hs_length), np.zeros(self.train_nhs_length)))
        self.train_df['label']=tmp_labels
        self.train_df=shuffle(self.train_df)
        self.train_length=self.train_df.shape[0]
        print self.train_df, self.train_length
        #print self.train_hs_df.shape, self.test_hs_df.shape, self.val_hs_df.shape
    def df2tensor(self, df_rows):
        indices=range(df_rows.shape[0])
        df_rows=df_rows.reset_index(drop=True)
        squishArrList=[]
        squishArrList.append(self.p.map(squishImage_mtp, zip(indices, itr.repeat(df_rows))))
        return np.stack(squishArrList)
    def reduce_nhs(self):
        new_train_nhs_length=self.train_hs_length
        indexes=np.arange(self.train_nhs_length)
        sample_indexes=rd.sample(indexes, new_train_nhs_length)
        new_train_nhs_df=self.train_nhs_df.iloc[sample_indexes].reset_index(drop=True)
        self.train_nhs_df=new_train_nhs_df
        self.train_nhs_length=new_train_nhs_length

        self.train_df=self.train_hs_df.append(self.train_nhs_df)
        tmp_labels=np.concatenate((np.ones(self.train_hs_length), np.zeros(self.train_nhs_length)))
        self.train_df['label']=tmp_labels
        self.train_df=shuffle(self.train_df)
        self.train_length=self.train_df.shape[0]
        #train_nhs_score=self.train_nhs_df['score'].values.flatten()
        #mean = np.mean(train_nhs_score)
        #sigma= np.std(train_nhs_score) 
        
    def get_train_batch(self, half_batch_size, mode='detection', balance=False):
        batch_size=half_batch_size*2
        
        if mode == 'detection':
            if balance:
                hs_batch_size=batch_size/(balance+1)
                nhs_batch_size=balance*batch_size/(balance+1)
                idx_hs= rd.sample(np.arange(self.train_hs_df.shape[0]), hs_batch_size)
                idx_nhs= rd.sample(np.arange(self.train_nhs_df.shape[0]), nhs_batch_size)
                train_hs = self.df2tensor(self.train_hs_df.iloc[idx_hs])[0]
                train_nhs = self.df2tensor(self.train_nhs_df.iloc[idx_nhs])[0]
                train = np.concatenate((train_hs, train_nhs), axis=0)
                train_label=np.concatenate((np.ones(hs_batch_size), np.zeros(nhs_batch_size)), axis=0)
            else:
                batch_size=half_batch_size*2
                if self.train_batch_pointer+batch_size<self.train_length:
                    batch_df= self.train_df.iloc[self.train_batch_pointer:self.train_batch_pointer+batch_size]
                    self.train_batch_pointer+=batch_size
                else:
                    batch_df= self.train_df.iloc[self.train_batch_pointer:self.train_length]
                    self.train_batch_pointer=0
                train=self.df2tensor(batch_df)[0]
                train_label=batch_df['label'].values.flatten()
                if self.train_batch_pointer==0: self.train_df=shuffle(self.train_df)
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





        






