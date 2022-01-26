#Draw the lines y = 2x + 1, y = 2x + 2 and y = 2x + 3 in the same figure. 
#Use different drawing colors and line types for your graphs to make them stand out 
# in black and white. 
# Set the image title and captions for the horizontal and vertical axes.

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,5,10)
y = 2*x + 1
y2 = 2*x + 2
y3 = 2*x + 3

plt.plot(x,y,"r--",x,y2,"b:", x,y3, "g-")
plt.title('Exercise 2.1')
plt.xlabel('Values of x')
plt.ylabel('Values of y')
plt.show()
