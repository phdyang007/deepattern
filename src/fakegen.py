import pandas as pd 
import numpy as np 
import sys 
from progress.bar import Bar 

class FAKE:
    def __init__(self,cmin,cmax,count):
        self.complexity = np.arange(cmin,cmax+1)
        self.count=count
        self.level=0
        self.head=['id','cX','cY','topoSig']
        self.data=[]
    def genPattern(self):
        bar = Bar('generating data', max = self.count)
        for i in range(self.count):
            size=np.random.choice(self.complexity, 1)[0]
            topo=np.random.randint(2, size=(size,size))
            for j in range(1,size,2):
                topo[j]=topo[j]*0
            cx=size
            cy=size
            toposig=''.join(topo.flatten().astype(str))
            self.data.append([i,cx,cy,toposig])
            bar.next()

        bar.finish()

    def dump2msgpack(self, path):
        self.df = pd.DataFrame(data=self.data, columns=self.head)
        self.df.to_msgpack(path)



fake_enum = FAKE(3,10,10000)
fake_enum.genPattern()
fake_enum.dump2msgpack("./data/fake/train.msgpack")
fake_enum = FAKE(3,10,10000)
fake_enum.genPattern()
fake_enum.dump2msgpack("./data/fake/test.msgpack")
print(fake_enum.df)

