# %% -------------------------------------------------------------------------------------------------------------------
# --------------------------------------- Imports -------------------------------------------------------------------
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import cv2
import torchvision
from torchvision import datasets
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data.sampler import SubsetRandomSampler


# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 5e-2
N_EPOCHS = 30
BATCH_SIZE = 512
DROPOUT = 0.5

# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def acc(x, y, return_labels=False):
    with torch.no_grad():
        logits = model(x)
        pred_labels_prob = nn.Sigmoid (logits.cpu().numpy())
        pred_labels = pred_labels_prob[pred_labels_prob>0.5]==1
        pred_labels = pred_labels_prob[pred_labels_prob<=0.5]==0
        print(pred_labels)
        #np.argmax(logits.cpu().numpy(), axis=1)  ##### copy the code to cpu first and then to numpy ###
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)

# % -------------------------------------------------------------------------------------
# Fit a CNN to the cell image dataset: https://storage.googleapis.com/exam-deep-learning/train-Exam2.zip
# % -------------------------------------------------------------------------------------

# %% -------------------------------------- CNN Class ------------------------------------------------------------------
#  LeNet-5 network #:
class CellCNN(nn.Module):
    def __init__(self):
        super(CellCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (3, 3))  # output (n_examples, 16, 48, 48)
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 16, 24, 24)
        self.conv2 = nn.Conv2d(16, 32, (3, 3))  # output (n_examples, 32, 22, 22)
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((2, 2))  # output (n_examples, 32, 11, 11)
        self.linear1 = nn.Linear(32*11*11, 250)  # input will be flattened to (n_examples, 32 * 5 * 5)
        self.linear1_bn = nn.BatchNorm1d(250)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(250, 7)
        self.act = torch.relu

    def forward(self, x):
        #x = x.reshape(1,-1)
        x = self.pool1(self.convnorm1(self.act(self.conv1(x.float()))))  #### first layer ###
        x = self.pool2(self.convnorm2(self.act(self.conv2(x.float()))))  #### second layer ###
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))  # fully connected layer ###
        return self.linear2(x)

# %% -------------------------------------- Data Prep ------------------------------------------------------------------

if "train" not in os.listdir():
    os.system("wget https://storage.googleapis.com/exam-deep-learning/train-Exam2.zip")
    os.system("unzip train-Exam2.zip")

DATA_DIR = os.getcwd () + "/train/"

RESIZE_TO = 50

x, y = [ ], [ ]
for path in [ f for f in os.listdir ( DATA_DIR) if f [ -4: ] == ".png" ]:
    x.append ( cv2.resize ( cv2.imread ( DATA_DIR + path ), (RESIZE_TO, RESIZE_TO) ) )
    with open ( DATA_DIR + path [ :-4 ] + ".txt", "r" ) as s:
       # label = s.read ()
        s1 = s.readlines ()
        label = [ ]
        for sub in s1:
            #print (sub.strip())
            #print(sub)
            label.append(sub.strip().split(","))
        #print(np.array(label).reshape(1,-1)[0])
    y.append(np.array(label).reshape(1,-1)[0])
    
x, y = np.array(x), np.array ( y )
# create multilabel binarizer object #
one_hot= MultiLabelBinarizer(classes=["red blood cell", "difficult", "gametocyte", "trophozoite","ring", "schizont", "leukocyte"])
y = one_hot.fit_transform(y)
dataset=list(zip(x,y))


############ split data into training and test #######
dataset_size=len(dataset)
indices=list(range(dataset_size))
test_size=0.2
split=int(np.floor(test_size*dataset_size))

np.random.seed(402)
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler=SubsetRandomSampler(train_indices)
test_sampler=SubsetRandomSampler(test_indices)

train_loader=torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,sampler=train_sampler)

#x_train, y_train = data_train.data.view(len(data_train), 1, 28, 28).float().to(device), data_train.targets.to(device)
#x_train.requires_grad = True
test_loader=torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,sampler=test_sampler)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = CellCNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
# define a training epoch function #
def trainEpoch(dataloader,epoch):
    print("Starting training loop...")
    print("Training Epoch %i" % (epoch+1))
    model.train()
    train_loss=0
    for i,data in enumerate(train_loader,0):
        inputs,labels=data
        inputs,labels=Variable(inputs),Variable(labels)
        optimizer.zero_grad()
        outputs=model(inputs.permute(0, 3, 1, 2))
        m=nn.Sigmoid()
        labels = labels.float()
        loss=criterion(m(outputs),labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if (i+1) %50 ==0:
            print('[%d,%5d] loss: %.3f' % (epoch+1,i+1, train_loss/50))
            train_loss=0.0


# define a testing function

#def Model_test(dataloader):
 #   model.eval()
  #  for inputs,targets in test_loader:
   #     inputs,targets = Variable(inputs),Variable(targets)
    #    outputs=model(inputs.permute(0, 3, 1, 2))
     #   m = nn.Sigmoid ()
      #  labels = labels.float ()
       # loss = criterion ( m ( outputs ), labels )
        # test_loss = loss.item()


for epoch in range(2):
    trainEpoch(train_loader,epoch)

torch.save(model.state_dict(),'/home/ubuntu/Deep-Learning/Pytorch_/CNN/Exam2/Day3/model_yjiang_test.pt')

model.eval ()  # Deactivates Dropout and makes BatchNorm use mean and std estimates computed during training
correct=0
for input,targets in test_loader:
    input,targets = Variable(input),Variable(targets)
    print(input.shape)
    print(input.permute(0, 3, 1, 2).shape)
with torch.no_grad ():  # The code inside will run without Autograd, which reduces memory usage, speeds up
    outputs = model(input.permute(0, 3, 1, 2)) # computations and makes sure the model can't use the test data to learn
    m = nn.Sigmoid ()
    targets = targets.float ()
    loss = criterion ( m ( outputs ), targets )
    loss_test = loss.item ()

#print ( "Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format (
 #   epoch,  acc ( x_train, y_train ), loss_test, acc ( x_test, y_test ) ) )
print('Test set: Average loss: (:.3f), Accuracy: {}/{} ({:.0f}%)\n'.format(loss_test,len(test_loader.dataset),
100. * correct/len(test_loader.dataset)))
