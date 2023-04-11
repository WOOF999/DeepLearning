import numpy as np
import matplotlib.pyplot as plt
import urllib.request

class Conv:
    def __init__(self,W):
        self.W=W
    def f_prop(self,X):
        out = np.zeros((X.shape[0]-2,X.shape[1]-2))
        print(X.shape)
        print(out.shape)
        
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                x=X[i:i+3,j:j+3]
                out[i,j]=np.dot(self.W.flatten(),x.flatten())
        return out

local_filename,header=urllib.request.urlretrieve("https://aidemystorageprd.blob.core.windows.net/data/5100_cnn_data/circle.npy")
X=np.load(local_filename)
print(X)
plt.imshow(X)
plt.title("The original image",fontsize=12)
#plt.show()


W1=np.array([[0,0,0],[1,1,1],[0,0,0]])
W2=np.array([[1,0,0],[0,1,0],[0,0,1]])
W3=np.array([[0,0,1],[0,1,0],[1,0,0]])
W4=np.array([[0,1,0],[1,1,1],[0,1,0]])

plt.subplot(1,4,1);plt.imshow(W1)
plt.subplot(1,4,2);plt.imshow(W2)
plt.subplot(1,4,3);plt.imshow(W3)
plt.subplot(1,4,4);plt.imshow(W4)
plt.suptitle("kernel",fontsize=12)

conv1=Conv(W1);c1=conv1.f_prop(X)
conv2=Conv(W2);c2=conv2.f_prop(X)
conv3=Conv(W3);c3=conv3.f_prop(X)
conv4=Conv(W4);c4=conv4.f_prop(X)

plt.subplot(1,4,1);plt.imshow(c1)
plt.subplot(1,4,2);plt.imshow(c2)
plt.subplot(1,4,3);plt.imshow(c3)
plt.subplot(1,4,4);plt.imshow(c4)
plt.suptitle("Convolution result",fontsize=12)
plt.show()


