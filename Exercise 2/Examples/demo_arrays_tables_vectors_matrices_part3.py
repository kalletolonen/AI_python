
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Plotting data points
x = [1,2,3,4]
y = [1,4,9,16]
plt.plot(x,y,marker="+")
plt.show()

# More points => more accurate picture
x = np.linspace(0,7,100)
y = np.sin(x)
plt.plot(x,y,color="black",linestyle="--")
plt.title("Sine curve ",color="red",fontsize=24)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# New picture starting here
y2 = np.cos(x)
plt.plot(x,y,"r--",x,y2,"b:")
plt.plot(x,np.sin(x/2))
plt.legend(['sin(x)','cos(x)','sin(x/2)'])
plt.show()


# Dividing picture to subpictures
plt.subplot(1,2,1)
plt.plot(x,y)
plt.subplot(1,2,2)
plt.plot(x,y2)
plt.suptitle('Commont headline   ')
plt.savefig("Picture_tables_arrays_vectors_matrices.png")
plt.show()


# Bar graph
plt.bar(['2018','2019','2020'],[10000,12000,13000],color="cyan")
plt.title('Sales')
plt.xlabel('Year')
plt.show()


# Histogramme
x = np.random.randn(2000)
plt.hist(x,20) # toinen parametri on pylväiden lkm
plt.show()


# Scatter graph
x = np.random.randn(200)
y = np.random.randn(200)
plt.scatter(x,y)
plt.show()


points = np.arange(-2,2,0.01)
x,y = np.meshgrid(points,points)
z = np.sqrt(x**2 + y**2)
plt.imshow(z)
plt.colorbar()
plt.show()

# Reading picture from file to table 
I = mpimg.imread('cameraman.png')
print(I.shape)
plt.imshow(I)
plt.show()
