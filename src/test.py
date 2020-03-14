from cdnsgen import *
import sys
data_path = sys.argv[1]
model_path = sys.argv[2]

from euv import *
data=EUV(path=data_path)
detector=hsd('euv', 'detection', model_path=model_path)
detector.model.test(data)