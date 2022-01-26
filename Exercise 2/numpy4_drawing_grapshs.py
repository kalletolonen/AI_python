import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 20, 100)  
#plt.plot(x, np.sin(x))       
#plt.show()               

x = np.linspace(0,7,100)
y = np.sin(x)
#plt.plot(x,y)
#plt.show()

plt.title('Title')
plt.xlabel('x')
plt.ylabel('y')
#plt.show()

#A comprehensive list can be found online
#https://matplotlib.org/3.1.1/api/text_api.html#matplotlib.text.Text

#plt.plot(x,y,"g")
#plt.show()

#In the same way, you can also control the plot marker, the most important of which are o = circle, x, +, s = square, *, D = diamond.
#plt.plot(x,y,"o")
#plt.show()

#plt.plot(x,y,linewidth=2,linestyle="--")


#plt.plot(x,y,"go",x,2*y,"r^")
#plt.legend(['sin(x)','cos(x)'])
#plt.show()

#Another way to display multiple graphs in the same figure window is to divide the window into subplots.
#plt.subplot(1,2,1) # 1x2 grid 1. subpicture
#plt.plot(x,y)
#plt.title("first")
#plt.subplot(1,2,2) # 1x2 grid 2. subpicture
#plt.plot(x,2*y)
#plt.title("second")
#plt.suptitle("Common Title")
#plt.show()

#plt.bar(['2018','2019','2020'],[120000,125000,130000],color="blue")
#plt.title("Title")
#plt.xlabel("years")
#plt.ylabel("Sales")
#plt.show()

#In data sciences and machine learning, statistical distributions of data are often studied. These can be quickly illustrated with histograms
#x = np.random.randn(2000)
#plt.hist(x,10)
#plt.ylabel('frequencies')
#plt.show()

#plt.scatter(x,y)
#plt.scatter(x,y,color="r",marker="o",label="Points")
#plt.show()

#points = np.arange(-2,2,0.01)
#x,y = np.meshgrid(points,points)
#z = np.sqrt(x**2 + y**2)
#plt.imshow(z)
#plt.colorbar()
#plt.show()

#As a side note, imshow () can also be used to display the contents of image files as long as the image file is first read in numeric format. E.g.
import matplotlib.image as mpimg
I=mpimg.imread("cameraman.png")
plt.imshow(I)
plt.show()

#Drawing three-dimensional textures also requires the mpl_toolkits module. E.g.
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

x, y = np.meshgrid(np.linspace(-2, 2, 30),np.linspace(-2, 2, 30))
z = np.cos(x ** 2 + y ** 2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z)
ax.set_title('Texture')
plt.show()

#plt.savefig('picture.pdf') #Save figure, type determined by file extension
#You may want to issue the command before the show () command if you want to save the image to a file and display it on the screen in the image window.