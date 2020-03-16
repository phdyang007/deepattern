import pandas as pd 
import sys
import random as rd 
import numpy as np 
import os
import multiprocessing as mtp
rd.seed(2)

msgpath=sys.argv[1]
outmsgpath = sys.argv[2]

df = pd.read_msgpack(msgpath)
required_rows = ['topoSig', 'xDelta', 'yDelta', 'cX', 'cY']
byte_rows = ['topoSig', 'xDelta', 'yDelta']


df.columns = ['ruleName', 'topoSig', 'xDelta', 'yDelta', 'cX', 'cY', 'ruleHash']



df = df[required_rows].reset_index()
total_size = df.shape[0]




for row in byte_rows:
    df[row] = df[row].str.decode(encoding='ASCII')

def organize(iloc, df=df):
    swap = 0
    dfrow = df.iloc[iloc]

  
    topoSig = dfrow['topoSig']
    m = np.uint8(dfrow['cX'])
    n = np.uint8(dfrow['cY'])
   
   
    yDelta = np.array(dfrow['yDelta'].split(' ')).astype(int)



    topo2d  = np.array(list(dfrow['topoSig'])).astype(np.uint8)
    topo2d  = topo2d.reshape((n,m),order='C')



    for i in yDelta:
        if not i%16==0:
            swap  = 1
            print("Found misorder at %g"%iloc)
            break
    
    if swap:
        tmpC = dfrow['cX']
        dfrow['cX']=dfrow['cY']
        dfrow['cY']=tmpC

        tmpD = dfrow['xDelta']
        dfrow['xDelta']=dfrow['yDelta']
        dfrow['yDelta']=tmpD


 
        topo2d = topo2d.transpose().flatten()
        dfrow['topoSig'] = ''.join(topo2d.astype(str))
    else:
        topo2d = topo2d.flatten()
        dfrow['topoSig'] = ''.join(topo2d.astype(str))
    
    dfrow['cX'] = int(dfrow['cX'])
    dfrow['cY'] = int(dfrow['cY'])
    return dfrow


#organize(iloc=71559)

p=mtp.Pool(mtp.cpu_count())

df= pd.concat(p.map(organize,range(len(df))), axis=1).transpose()



train_size = int(0.8*total_size)


train_list = rd.sample(range(total_size), train_size)
#print(train_list)

df_train = df.iloc[train_list]
df_test = df.drop(train_list)

print(df_train)
print(df_test)




df_train.to_msgpack(os.path.join(outmsgpath,"train.msgpack"))
df_train.to_msgpack(os.path.join(outmsgpath,"test.msgpack"))
