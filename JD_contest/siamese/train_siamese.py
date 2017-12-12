
# coding: utf-8

# # One Shot Learning with Siamese Networks
# 
# This is the jupyter notebook that accompanies

# ## Imports
# All the imports are defined here

# In[1]:

#get_ipython().magic(u'matplotlib inline')
import torchvision
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
#import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


# ## Helper functions
# Set of helper functions

# In[93]:

# def imshow(img,text,should_save=False):
#     npimg = img.numpy()
#     plt.axis("off")
#     if text:
#         plt.text(75, 8, text, style='italic',fontweight='bold',
#             bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()    

# def show_plot(iteration,loss):
#     plt.plot(iteration,loss)
#     plt.show()


# ## Configuration Class
# A simple class to manage configuration

# In[15]:

class Config():
    training_dir = "/usr/JD/wipe_out/good/train_set_5/train/"
    testing_dir = "/usr/JD/wipe_out/good/train_set_5/val/"
    train_batch_size = 32
    train_number_epochs = 100


# ## Custom Dataset Class
# This dataset generates a pair of images. 0 for geniune pair and 1 for imposter pair

# In[4]:

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        #img0 = img0.convert("L")
        #img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)


# ## Using Image Folder Dataset

# In[5]:

folder_dataset = dset.ImageFolder(root=Config.training_dir)


# In[6]:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Scale((224,224)),
                                            transforms.RandomHorizontalFlip(),
                                                                      transforms.ToTensor(), normalize,
                                                                      ])
                                       ,should_invert=False)


# ## Visualising some of the data
# The top row and the bottom row of any column is one pair. The 0s and 1s correspond to the column of the image.
# 0 indiciates dissimilar, and 1 indicates similar.

# In[7]:

vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=8)
dataiter = iter(vis_dataloader)


example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
#imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())


# ## Neural Net Definition
# We will use a standard convolutional neural network

# In[8]:
model = models.__dict__['resnet50'](pretrained=True)
class MyResNet(nn.Module):

    def __init__(self, model):
        super(MyResNet, self).__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        #self.Net_classifier=Net_classifier


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.Net_classifier(x)


        return x
my_model=MyResNet(model)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = my_model
        # self.cnn1 = nn.Sequential(
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(3, 4, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(4),
        #     nn.Dropout2d(p=.2),
            
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(4, 8, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(8),
        #     nn.Dropout2d(p=.2),

        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(8, 8, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(8),
        #     nn.Dropout2d(p=.2),
        # )

        # self.fc1 = nn.Sequential(
        #     nn.Linear(8*100*100, 500),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(500, 500),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(500, 5))


    def forward_once(self, x):
        output = self.cnn1(x)
        #output = output.view(output.size()[0], -1)
        #output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2



# ## Contrastive Loss

# In[9]:

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        #output1_1=torch.FloatTensor(output1.size(0),output1.size(1))
        #output1_2=torch.FloatTensor(output1.size(0),output1.size(1))
        euclidean_distance = F.pairwise_distance(output1[:,:,0,0], output2[:,:,0,0])
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


# ## Training Time!

# In[10]:

train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=Config.train_batch_size)


# In[11]:

net = SiameseNetwork().cuda()
print(net)
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )


# In[12]:

counter = []
loss_history = [] 
iteration_number= 0


# In[63]:

for epoch in range(0,Config.train_number_epochs):
    for i, data in enumerate(train_dataloader,0):
        img0, img1 , label = data
        img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
        output1,output2 = net(img0,img1)
        optimizer.zero_grad()
        loss_contrastive = criterion(output1,output2,label)
        loss_contrastive.backward()
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.data[0]))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.data[0])
    torch.save(net.state_dict(), './models/siamese_'+str(epoch+1)+'.pth')
#show_plot(counter,loss_history)


# ## Some simple testing
# The last 3 subjects were held out from the training, and will be used to test. The Distance between each image pair denotes the degree of similarity the model found between the two images. Less means it found more similar, while higher values indicate it found them to be dissimilar.

# In[96]:

folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Scale((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

test_dataloader = DataLoader(siamese_dataset,num_workers=6,batch_size=1,shuffle=True)
dataiter = iter(test_dataloader)
x0,_,_ = next(dataiter)

for i in range(10):
    _,x1,label2 = next(dataiter)
    concatenated = torch.cat((x0,x1),0)
    
    output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    #imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]))



# In[ ]:



