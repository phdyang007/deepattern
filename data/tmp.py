import pandas as pd
import numpy as np
import sys
import os
inf=sys.argv[1]

for dirname, dirnames, filenames in os.walk(inf):
    for i in filenames:
        dfdir=os.path.join(dirname, i)
        udf=pd.read_msgpack(dfdir)
        hs=np.where(np.logical_not(pd.isnull(udf.dist.values)))[0]
        hsdf=udf.iloc[hs].reset_index(drop=True)
        hsdir=os.path.join(dirname, 'hs_'+i)
        hsdf.to_msgpack(hsdir)

