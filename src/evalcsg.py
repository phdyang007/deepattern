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
    
    test_msg = os.path.join('./models/', case+'/test/noise_data.msgpack')


    df_test = pd.read_msgpack(test_msg)
    df_test = df_test.loc[df_test.valid==1]
    stat_pattern  = df_test[['topoSig']].values
    all_valid = len(stat_pattern)
    uniq_frac, uniq_idx =  np.unique(stat_pattern, return_index=True)

    df_test = df_test.iloc[uniq_idx]

    stat_test = df_test[['cX','cY']].values

    statt, stat_count_test = np.unique(stat_test, return_counts=True, axis=0)
    
    print("======================GAN-TCAE CSG=====================")

    div_test = get_div(stat_count_test)
    valid_frac = np.sum(stat_count_test)


    print("The Diversity of Noise Perturbed Data of %s is %f"%(case, div_test))
    print("The Total # of Unique Valid Pattern Count of %s is %g"%(case, valid_frac))
    print("The Total # of Valid Pattern Count of %s is %g"%(case, all_valid))

    cplex = np.average(stat_test, axis=0)
    print("The average complexity of csgs are X : %g, Y: %g"%(cplex[0], cplex[1]))

    df_low = df_test.loc[df_test.cX <5]
    df_mid = df_test.loc[(df_test.cX >=10) & (df_test.cX <12)]
    df_high= df_test.loc[df_test.cX >16]

    lowp = len(df_low)/valid_frac
    midp = len(df_mid)/valid_frac
    highp= len(df_high)/valid_frac

    print("The breakdown of pattern cplexitis of low, mid and high are %f, %f and %f"%(lowp, midp, highp))


if __name__== "__main__":
    case = sys.argv[1]
    analysis(case)




