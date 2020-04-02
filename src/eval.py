import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math
import copy


def get_div(stat):
    div = 0
    for i in stat:
        pi = i/np.sum(stat)
        div -= pi*math.log(pi, np.e)
    
    return div

def get_opt_div(stat, div_ref, case):
    precision = 100
    max_count = np.max(stat)
    stat_2 = copy.copy(stat)
    for threshold_div in range(precision):
        threshold=max_count*(precision-threshold_div)/precision
        stat_2 = np.minimum(stat, np.ones(len(stat))*threshold)
        div = get_div(stat_2)
        if div>=div_ref:
            print("The Optimal Diversity of Noise Perturbed Data of %s is %f"%(case, div))
            print("The Optimal Total # of Unique Valid Pattern Count of %s is %g"%(case, np.sum(stat_2)))
            return 0
    
    print("Optimal Solution Does Not Exist")
    return -1



def analysis(case, dim_max = 24): #case: tc1 tc2 tc3 tc4 tc5 tc6
    
    origin_msg = os.path.join('./data/', case+'/train.msgpack')
    test_msg = os.path.join('./models/', case+'/test/noise_data.msgpack')

    df_origin = pd.read_msgpack(origin_msg)
    df_test = pd.read_msgpack(test_msg)
    df_test = df_test.loc[df_test.valid==1]
    stat_pattern  = df_test[['topoSig']].values
    all_valid = len(stat_pattern)
    uniq_frac, uniq_idx =  np.unique(stat_pattern, return_index=True)

    df_test = df_test.iloc[uniq_idx]
    stat_origin = np.int8(df_origin[['cX','cY']].values)
    stat_test = df_test[['cX','cY']].values

    stato, stat_count_origin = np.unique(stat_origin, return_counts=True, axis=0)
    statt, stat_count_test = np.unique(stat_test, return_counts=True, axis=0)
    
    print("======================TCAE=====================")
    print("Calculating Training Data Diversity.....")
    div_origin= get_div(stat_count_origin)
    div_test = get_div(stat_count_test)
    valid_frac = np.sum(stat_count_test)

    print("The Diversity of Original Training Data in %s is %f"%(case, div_origin))
    print("The Diversity of Noise Perturbed Data of %s is %f"%(case, div_test))
    print("The Total # of Unique Valid Pattern Count of %s is %g"%(case, valid_frac))
    print("The Total # of Valid Pattern Count of %s is %g"%(case, all_valid))



    return div_test, valid_frac

def analysis_gan(case, div_ref, dim_max = 24): #case: tc1 tc2 tc3 tc4 tc5 tc6

    test_msg = os.path.join('./models/', case+'/gan/test/noise_data.msgpack')

    df_test = pd.read_msgpack(test_msg)
    df_test = df_test.loc[df_test.valid==1]

    stat_pattern  = df_test[['topoSig']].values
    all_valid = len(stat_pattern)
    uniq_frac, uniq_idx =  np.unique(stat_pattern, return_index=True)
    df_test = df_test.iloc[uniq_idx]
    stat_test = df_test[['cX','cY']].values


    
    statt, stat_count_test = np.unique(stat_test, return_counts=True, axis=0)

    stat_pattern  = df_test[['topoSig']].values.flatten()
    uniq_frac =  len(list(set(stat_pattern)))
    print("======================G-TCAE=====================")
    print("Calculating Training Data Diversity.....")
    #div_origin= get_div(stat_count_origin)
    div_test = get_div(stat_count_test)
    valid_frac = np.sum(stat_count_test)

    #print("The Diversity of Original Training Data in %s is %f"%(case, div_origin))
    print("The Diversity of Noise Perturbed Data of %s is %f"%(case, div_test))
    print("The Total # of Unique Valid Pattern Count of %s is %g"%(case, valid_frac))
    print("The Total # of Valid Pattern Count of %s is %g"%(case, all_valid))
    get_opt_div(stat_count_test, div_ref, case)
if __name__== "__main__":
    case = sys.argv[1]
    div_test, _ =analysis(case)
    try:
        analysis_gan(case, div_test)
    except:
        print("TCAE-GAN results not avaiable")








