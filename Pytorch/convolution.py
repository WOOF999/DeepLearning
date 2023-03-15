import torch
import matplotlib.pyplot as plt
import time
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

batch_size = 10000
learning_rate=0.0002
num_epoch =10

mnist_train=dset.MNIST(root="MNIST_data/",train=True,transform=transforms.ToTensor(),target_transform=None,download=True)
mnist_test=dset.MNIST(root="MNIST_data/",train=False,transform=transforms.ToTensor(),target_transform=None,download=True)

train_loader=DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True,num_workers=0)
test_loader=DataLoader(dataset=mnist_test,batch_size=len(mnist_test),shuffle=False,num_workers=0)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.cnv1=torch.nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=0)
        self.cnv2=torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=0)
        self.fc=torch.nn.Linear(512,10)

        self.mp=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

    def forward(self,x):
        x=self.relu(self.mp(self.cnv1(x)))
        x=self.relu(self.mp(self.cnv2(x)))

        x=x.view(-1,512)
        z=self.fc(x)
        return z
        

net= Net()

cel=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.01)

loss_lst=[]
acc_lst=[]

start_time=time.time()
for epoch in range(num_epoch):
    batch_time=time.time()
    net.train()
    l_sum=0


    for batch_idx,[x,y] in enumerate (train_loader):
        z=net(x)
        loss=cel(z,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l_sum+=loss.item()
        print(f'Batch : {batch_idx+1:2d}/{len(train_loader)}',f'Loss : {loss.item():0.6f}')
    loss_lst.append(l_sum/len(train_loader))

    net.eval()
    correct=0
    for batch_idx,[x,y] in enumerate(test_loader):
        z=net(x)
        yhat=torch.argmax(z,dim=1)
        correct+=torch.sum(y==yhat)

    accuracy=correct/len(mnist_test)
    acc_lst.append(accuracy)

    print(f'Accuracy : {accuracy.item()*100:0.2f}%')
    print("elapsed time : ",time.time()-batch_time)
print("elapsed time : ",time.time()-start_time)

plt.plot(range(num_epoch),loss_lst)
plt.plot(range(num_epoch),acc_lst)
plt.title("Epoch / loss and accuracy")
plt.xlabel("Epochs")
plt.ylabel("loss and accuracy")
plt.legend(['loss','accuracy'])
plt.grid(True)
plt.show()


