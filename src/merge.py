import pandas as pd
from progress.bar import Bar
import sys
import os
path = sys.argv[1]
which= sys.argv[2]


 


if which=="tcae":
    df1=pd.DataFrame()
    bar=Bar("Merging", max=100)
    for i in range(100):
        try:
            df1=df1.append(pd.read_msgpack(os.path.join(path,'test/noise_data_'+str(i)+'.msgpack')))
        except:
            pass

        bar.next()
    
    bar.finish()
    df1.to_msgpack(os.path.join(path, 'test/noise_data.msgpack'))
elif which=="data":
    df1=pd.DataFrame()
    bar=Bar("Merging", max=100)
    for i in range(6):
        try:
            df1=df1.append(pd.read_msgpack(os.path.join(path,'tc'+str(i+1)+'/train_with_feature.msgpack')))
        except:
            pass

        bar.next()
    
    bar.finish()
    df1.to_msgpack(os.path.join(path, 'tc0/train_with_feature.msgpack')) 
else:
    df1=pd.DataFrame()
    bar=Bar("Merging", max=1000)
    for i in range(1000):
        try:
            df1=df1.append(pd.read_msgpack(os.path.join(path,'noise_data_'+str(i)+'.msgpack')))
        except:
            pass

        bar.next()
    
    bar.finish()
    df1.to_msgpack(os.path.join(path, 'noise_data.msgpack'))