import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
os.chdir(r'D:\Ubuntu_Server_Share')
with open('run.log', 'r', encoding='utf-8') as f:
    res = f.readlines()

imap = []
person = []
bicycle = []
car = []

for i in range(len(res)):
    if 'Index | Class name | AP' in res[i]:
        tmp = res[i+2:i+5]+[res[i+6]]
        imap.append(float(tmp[-1].strip().split(' ')[-1]))
        person.append(float(tmp[0].split(' ')[-2]))
        bicycle.append(float(tmp[1].split(' ')[-2]))
        car.append(float(tmp[2].split(' ')[-2]))

x=range(1,len(imap)+1)
plt.subplot(111)
plt.title('EVAL')
plt.xlabel('epochs')
plt.ylabel('AP')
plt.xlim((0, 100))
plt.ylim((0, 1))
plt.plot(x, imap,'-',label='map')
plt.plot(x,person,label='person')
plt.plot(x,bicycle,label='bicycle')
plt.plot(x,car,label='car')
plt.legend(loc='upper right', frameon=True)
plt.show()