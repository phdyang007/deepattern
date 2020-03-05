import pandas as pd 
import numpy as np 
import sys
import os 
from progress.bar import Bar 

class FAKE:
    def __init__(self):
        self.data_path=os.getcwd()+"/data/valid_data.csv"
        print("path:", self.data_path)
        self.head=['id','cX','cY','topoSig']
    def genPattern(self, phase):
        input_data = pd.read_csv(self.data_path)
        input_data.columns = self.head
        if phase is "train":
            return input_data.head(int(input_data.shape[0]*0.8))
        if phase is "test":
            return input_data.tail(int(input_data.shape[0]*0.2))

    def dump2msgpack(self, data, path):
        self.df = pd.DataFrame(data=data, columns=self.head)
        self.df.to_msgpack(path)


fake_enum = FAKE()
train_data = fake_enum.genPattern("train")
fake_enum.dump2msgpack(train_data, "./data/train.msgpack")
test_data = fake_enum.genPattern("test")
fake_enum.dump2msgpack(test_data, "./data/test.msgpack")
print(fake_enum.df)

