from cdnsgen import *
import sys
model_path = sys.argv[1]
vec_path = sys.argv[2]

from euv import *
#data=EUV(path=data_path)
detector=hsd('euv', 'detection', model_path=model_path)
detector.model.batch_size=10
detector.model.test_csg(vec_path)