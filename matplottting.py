import matplotlib
from matplotlib import pyplot as plt


start = -5
x = []
y = []
for i in range(11):
    x.append(start)
    y.append(start**2)
    start+=1


plt.plot(x,y)
plt.show()