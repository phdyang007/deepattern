import sys
import os
import pandas as pd

def prepare(path):
    df=pd.read_msgpack(path)
    df=df.loc[df['valid']==1]
    df_low=df.loc[df['cX']<7]
    df_mid=df.loc[df['cX']==10]
    df_high=df.loc[df['cX']>13]
    low_path = "./data/csg/low/train.msgpack"
    mid_path = "./data/csg/mid/train.msgpack"
    high_path = "./data/csg/high/train.msgpack"
    print(df_low, df_mid, df_high)
    df_low.to_msgpack(low_path)
    df_mid.to_msgpack(mid_path)
    df_high.to_msgpack(high_path)

if __name__ == '__main__':
    path = "./models/" + sys.argv[1] + "/test/noise_data.msgpack"
    prepare(path)