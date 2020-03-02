import os
import time

output = print
# output = os.system

data = time.localtime()

output('git add .')
output('git commit -m \"update_{}{:02d}{}\"'.format(data.tm_year, data.tm_mon, data.tm_mday))
output('git push')

