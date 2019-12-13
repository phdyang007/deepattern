import pandas as pd
from progress.bar import Bar


df1=pd.DataFrame()
df2=pd.DataFrame()
bar=Bar("Merging", max=100)
for i in xrange(100):
    df1=df1.append(pd.read_msgpack('./models/test/noise_data_'+str(i)+'.msgpack'))
    df2=df2.append(pd.read_msgpack('./models/test/span_data_'+str(i)+'.msgpack'))
    bar.next()
   
bar.finish()
df1.to_msgpack('./models/test/noise_data.msgpack')
df2.to_msgpack('./models/test/span_data.msgpack')
