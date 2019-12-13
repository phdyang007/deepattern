import pandas as pd 
import multiprocessing as mtp
import numpy as np
import sys
import itertools as itr
def check_floating_wire(v):
    state=0
    s=None
    e=None
    intervals=[]

    for i in xrange(len(v)):
        if v[i]==0:
            if state==0 or state==1:
                state=1
                s=i
            elif state==2:
                state=1
                e=i
                intervals.append([s,e])
                s=i

        if v[i]==1:
            if state==0:
                state==0
            elif state==1 or state==2:
                state=2
           

    intervals=np.array(intervals)
    #for i in xrange(len(intervals)):
    #    print v[intervals[i,0]:intervals[i,1]+1]

    return intervals


def check_spacing(v):
    state=0
    s=None
    e=None
    intervals=[]

    for i in xrange(len(v)):
        if v[i]==1:
            if state==0 or state==1:
                state=1
                s=i
            elif state==2:
                state=1
                e=i
                intervals.append([s,e])
                s=i

        if v[i]==0:
            if state==0:
                state==0
            elif state==1 or state==2:
                state=2
           

    intervals=np.array(intervals)
    #for i in xrange(len(intervals)):
    #    print v[intervals[i,0]:intervals[i,1]+1]

    return intervals

def constrains(df_row):

    m = np.uint8(df_row['cX'])
    n = np.uint8(df_row['cY'])
    constrain_string=[]
    Imarr  = np.array(list(df_row['topoSig'])).astype(np.uint8)
    Imarr  = Imarr.reshape((n,m),order='C')    
    
    deltax = np.zeros(m).astype(int).astype(str)
    deltay = np.zeros(n).astype(int).astype(str)

    for i in xrange(n):
        #deltaY constrains: fixed pitch, fixed cd
        y_id = n-1 - i
        if i==0 or i==n-1:
            if np.sum(Imarr[i])>0:
                deltay[y_id] = '1-16'
            else:
                deltay[y_id] = '1-96'
        else:
            deltay[y_id] = '16'
        
        #deltaX constrains
        #t2t
        intervals=check_spacing(Imarr[i])
        for j in xrange(len(intervals)):
            s=intervals[j,0]
            e=intervals[j,1]
            if e-s==2:
                deltax[e]='16-96'
            else:
                constrain_string.append('x'+str(e)+'-x'+str(s+1)+' &gt;= 16')

        #wire length
        intervals=check_floating_wire(Imarr[i])
        for j in xrange(len(intervals)):
            s=intervals[j,0]
            e=intervals[j,1]
            if e-s==2:
                deltax[e]='16-96'
            else:
                constrain_string.append('x'+str(e)+'-x'+str(s+1)+' &gt;= 16')   

    #window constrains
    constrain_string.append('y'+str(n)+'-y0 == 192')
    constrain_string.append('x'+str(m)+'-x0 == 192')

    #Finish Deltas
    for i in xrange(m):
        if deltax[i]=='0':
            deltax[i]='1-96'
    return [','.join(constrain_string), ' '.join(deltax), ' '.join(deltay)]

def main():
    p=mtp.Pool(8)
    df = pd.read_msgpack(sys.argv[1])
    #print df
    indices = range(df.shape[0])
    const=[]
    a=[df.iloc[i] for i in indices][0]
    #print constrains(a)
    #quit()
    const.append(p.map(constrains, [df.iloc[i] for i in indices]))
    print len(const), len(const[0])
    df_constraint=pd.DataFrame(const[0], columns=['constrains','xDelta', 'yDelta'])
 
    udf=pd.concat([df, df_constraint], axis=1, join_axes=[df.index])
    udf.to_msgpack(sys.argv[2])
main()