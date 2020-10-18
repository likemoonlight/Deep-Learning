# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import transforms, utils
import torch.optim as optim
import matplotlib.pyplot as plt
from thop import profile
import matplotlib.pyplot as plt
import PIL
from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
import torchvision.transforms.functional as TF
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
a = 0
b = 2
c=0.01 
d=0.001
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
data_dir = r'./DL'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ('B卡板豆干', 'C橋板', 'E青砲', 'F光碟機板', 'H硬碟板總貨', 'I雜板', 'M豆干', 'N排骨', 'P P2', 'P P3', 'P P4', 
            'P筆電P3', 'R記憶體厚', 'R記憶體鋁', 'R記憶體薄', 'R筆電記憶體', 'Z 586CPU', 'Z毛粗', 'Z毛細', 'Z南北橋A', 
            'Z南北橋B', 'Z南北橋銅', 'Z筆電光碟機板', 'Z筆電網路卡')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print('==> Preparing dataset..')

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

##The transform function for test data 224
transform_test = transforms.Compose([\
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##Data Augmentation
class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Resize((224,224)),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.25,iaa.OneOf([iaa.Dropout(p=(0, 0.1)),iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True), ])
  
    def __call__(self, img):
        img = np.array(img)  
        return  self.aug.augment_image(img)
    
transform_train_img = transforms.Compose([
    ImgAugTransform(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), ])

trainset_img = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'Train'),transform = transform_train_img)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'Train'),transform = transform_train)
valset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'Val'),transform = transform_test)
testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'Test'),transform = transform_test)

trainloader = torch.utils.data.DataLoader(trainset_img, batch_size=16,
shuffle=True, num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=16,
shuffle=False, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
shuffle=False, num_workers=8)

print(len(trainloader))
print(len(valloader))
print(len(testloader))

# print(next(iter(trainloader)))

def show_batch(imgs):
    grid = utils.make_grid(imgs,nrow=1)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')

for i, (batch_x, batch_y) in enumerate(trainloader):
    if(i<0):
        print(i, batch_x.size(), batch_y.size())
        show_batch(batch_x[0])
        plt.axis('off')
        plt.show()
    else:
        break
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%  
print('==> Building model..')
n18 = models.resnet18(pretrained=True)
n18.fc = nn.Linear(in_features=512, out_features=24, bias=True)
net = n18.to(device)    
net = torch.load('./0880304.h5')

#%%
print('==> Defining loss function and optimize..')
##loss function
criterion = nn.CrossEntropyLoss()
##optimization algorithm
optimizer = optim.SGD(net.parameters(), lr=c, weight_decay=d) # momentum=0.9,

#%%
print('==> Training model..')
net.train()

for epoch in range(a):  # loop over the dataset multiple times
    running_loss = 0.0
    val_running_loss = 0
    correct = 0
    for i, (inputs, labels) in enumerate(trainloader, 0):       
        ##change the type into cuda tensor 
        # print(inputs.shape)
        inputs = inputs.to(device) 
        labels = labels.to(device) 
        
        ##zero the parameter gradients
        optimizer.zero_grad()

        ##forward + backward + optimize
        outputs = net(inputs)
        ##select the class with highest probability
        _, pred = outputs.max(1)
        ##if the model predicts the same results as the true
        ##label, then the correct counter will plus 1
        correct += pred.eq(labels).sum().item()       
        loss = criterion(outputs, labels)     
        loss.backward()
        optimizer.step()
        ##print statistics
        running_loss += loss.item()
        if i % 400 == 20:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0            
    print('%d epoch, training accuracy: %.4f' % (epoch+1, 100.*correct/len(trainset)))            
    ###########################################################################   
    if epoch % b ==0:        
        net.eval()
        val_correct = 0
        for i, (inputs, labels) in enumerate(valloader, 0):
            #print(inputs.shape)
            inputs = inputs.to(device) 
            labels = labels.to(device)          
            outputs = net(inputs)
            _, pred = outputs.max(1)
            val_correct += pred.eq(labels).sum().item()   
        print('val accuracy: %.4f' % (100.*val_correct/len(valset)))
        #######################################################################
        net.train()
print('Finished Training')

#%%
print('==> Testing model..')
##Set the model in evaluation mode
net.eval()
test_correct = 0
top3_correct = 0
classAcc = [0 for _ in range(24)]
classNum = [0 for _ in range(24)]


# idx_to_class = {val:key for key,val in testset.class_to_idx.items()}

idx_to_class = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23')
for inputs, labels in testloader:

    inputs = inputs.to(device)
    labels = labels.to(device)
    
    outputs = net(inputs)
    _, pred = outputs.topk(k=3, dim=1)
    test_correct += pred[:,0].eq(labels).sum().item()
    top3_correct += sum([pred[:,i].eq(labels).sum().item() for i in range(3)])
    for index in range(len(labels)):
        class_ = labels[index]
        classAcc[eval(idx_to_class[class_.item()])] += (pred[index,0]==class_).item()
        classNum[eval(idx_to_class[class_.item()])] += 1

print('Test set: Test Accuacy: %d/%d (%.0f%%)' \
      % (test_correct,len(testset),100.*test_correct/len(testset)))
#######################################################################    
#     torch.save(net, './model_DL4.h5')
#     print('Finished Saving') 
    
for i in range(24):
    print('Class %-2d: Correct:%3d/ Test pic:%3d\t/ Acc: %.2f%%'\
          % (i,classAcc[i],classNum[i],(100.*classAcc[i])/classNum[i]))

# print(len(classAcc))
# print(len(classNum))
# print(len(classAcc))
# print(len(classNum))
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
