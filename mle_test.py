import numpy as np
from scipy import special
import matplotlib.pyplot as plt

N = 100
occurs = 50
x = np.linspace(0,1,N+1)
y = [special.comb(N,occurs)*i**occurs*(1-i)**(N-occurs) for i in x]
plt.plot(x,y)
max_y = np.max(y)
max_x = x[np.argmax(y)]
plt.annotate("max=%.3f"%max_y,(max_x,max_y))
plt.show()