import pandas as pd 
import sys
import random as rd 
import numpy as np 

msgpath=sys.argv[1]

df = pd.read_msgpack(msgpath)
required_rows = ['topoSig', 'xDelta', 'yDelta', 'cX', 'cY']
byte_rows = ['topoSig', 'xDelta', 'yDelta']


df.columns = ['ruleName', 'topoSig', 'xDelta', 'yDelta', 'cX', 'cY', 'ruleHash']



df = df[required_rows].reset_index()
total_size = df.shape[0]




for row in byte_rows:
    df[row] = df[row].str.decode(encoding='ASCII')


train_size = int(0.8*total_size)


train_list = rd.sample(range(total_size), train_size)
#print(train_list)

df_train = df.iloc[train_list]
df_test = df.drop(train_list)

print(df_train)
print(df_test)




df_train.to_msgpack("train.msgpack")
df_test.to_msgpack("test.msgpack")
