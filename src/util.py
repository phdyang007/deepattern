import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def hmr(hc_path, mc_path, out, bins=100):
    hc=pd.read_csv(hc_path, usecols=['ruleHash','Count'],delim_whitespace=True)
    mc=pd.read_csv(mc_path, names=['ruleHash','mc'],delim_whitespace=True)
    udf=hc.merge(mc, how='inner', on='ruleHash')
    hs_count=udf.Count.values
    match_count=udf.mc.values
    ratio=hs_count*1.0/match_count
    index=np.arange(bins)*1.0/bins
    bin_value=[]
    for i in xrange(bins):
        l=i*1.0/bins
        u=(i+1)*1.0/bins
        tmp=np.logical_and(np.greater(ratio,l),np.greater(u,ratio))
        bin_value.append(len(np.where(tmp==True)[0]))

    plt.bar(index, bin_value, width=1.0/bins, align='center', alpha=1)
    plt.xlabel('Hotspot Match Count Rate')
    plt.ylabel('Pattern Count')
    plt.savefig(out)
    plt.clf()
    plt.close()
data1=['test', '1']
data2=['test', '2']
data3=['train', '3']
data4=['train', '4']


for data in [data1, data2, data3, data4]:
    hc_path='./data/euv/'+data[0]+'/label_tl'+data[1]+'.txt'
    mc_path='./data/euv/'+data[0]+'/matchCount_tl'+data[1]+'.txt'
    out='./data/euv/'+data[0]+'/hmr'+data[1]+'.pdf'
    udf=hmr(hc_path, mc_path, out)



