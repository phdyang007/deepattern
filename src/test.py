from cdnsgen import *
import sys


from euv import *
data=EUV()
detector=hsd('euv', 'detection')
detector.model.test(data)