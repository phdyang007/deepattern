import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math



def get_div(stat):
    div = 0
    for i in stat:
        pi = i/np.sum(stat)
        div -= pi*math.log(pi, 2)
    
    return div


def analysis(case, dim_max = 24): #case: tc1 tc2 tc3 tc4 tc5 tc6
    
    origin_msg = os.path.join('./data/', case+'/train.msgpack')
    test_msg = os.path.join('./models/', case+'/test/noise_data.msgpack')

    df_origin = pd.read_msgpack(origin_msg)
    df_test = pd.read_msgpack(test_msg)

    stat_origin = np.int8(df_origin[['cX','cY']].values)
    stat_test = df_test.loc[df_test.valid==1][['cX','cY']].values


    stato, stat_count_origin = np.unique(stat_origin, return_counts=True, axis=0)
    statt, stat_count_test = np.unique(stat_test, return_counts=True, axis=0)


    print("Calculating Training Data Diversity.....")
    div_origin= get_div(stat_count_origin)
    div_test = get_div(stat_count_test)
    valid_frac = np.sum(stat_count_test)

    print("The Diversity of Original Training Data in %s is %f"%(case, div_origin))
    print("The Diversity of Noise Perturbed Data of %s is %f"%(case, div_test))
    print("The Total # of Unique Valid Pattern Count of %s is %g"%(case, valid_frac))



def analysis_gan(case, dim_max = 24): #case: tc1 tc2 tc3 tc4 tc5 tc6
    
    origin_msg = os.path.join('./data/', case+'/train.msgpack')
    test_msg = os.path.join('./models/', case+'/gan/test/noise_data.msgpack')

    df_origin = pd.read_msgpack(origin_msg)
    df_test = pd.read_msgpack(test_msg)

    stat_origin = np.int8(df_origin[['cX','cY']].values)
    stat_test = df_test.loc[df_test.valid==1][['cX','cY']].values


    stato, stat_count_origin = np.unique(stat_origin, return_counts=True, axis=0)
    statt, stat_count_test = np.unique(stat_test, return_counts=True, axis=0)


    print("Calculating Training Data Diversity.....")
    div_origin= get_div(stat_count_origin)
    div_test = get_div(stat_count_test)
    valid_frac = np.sum(stat_count_test)

    print("The Diversity of Original Training Data in %s is %f"%(case, div_origin))
    print("The Diversity of Noise Perturbed Data of %s is %f"%(case, div_test))
    print("The Total # of Unique Valid Pattern Count of %s is %g"%(case, valid_frac))

if __name__== "__main__":
    case = sys.argv[1]
    analysis(case)
    analysis_gan(case)








