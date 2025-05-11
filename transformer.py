from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from torch.optim import AdamW

import math
from time import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from uitls_8822_capacity import computation_time
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from numpy import mat
import numbers
import matplotlib.pyplot as plt
from scipy.special import erfc  

from torch import nn, einsum
from torch.nn import Module
from einops import rearrange, pack, unpack
import torch.nn.functional as F




from torch import nn, einsum
from torch.nn import Module
from einops import rearrange, pack, unpack
import torch.nn.functional as F

x = pd.read_csv(r'/home/wwj/chenqiliang/8822_list.csv',header=None)
matrix = x.values



def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i, 4]
a=1
#a=10**0.5
#a=10
#a=10**1.5
#a=100
Nt = 1
P_t =0.01
P_A =0.35
N_s =2
P_ct =0.0344 
N_r =2
P_cr =0.0625
P_sync = 0.050
B_W = 10000000









dataset = pd.read_csv(r'/home/wwj/chenqiliang/SVD/newAll-channel_matrix_p_1.csv').iloc[:, 1:]

dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8, 1)

#label = pd.read_csv(r'/home/wwj/chenqiliang/SVD/newcapacity_labels_p_1.csv').iloc[:, 1]

label = pd.read_csv(r'/home/wwj/chenqiliang/SVD/newgain_labels_p_1.csv').iloc[:, 1]







label = np.asarray(label, np.int32)
label.astype(np.int32)

#one hot
n_class = 784
n_sample = label.shape[0]
label_array = np.zeros((n_sample, n_class))
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1



xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=0.2, random_state=40)
print("xTrain: ", len(xTrain))
print(xTrain.shape)

print("xTest: ", len(xTest))










class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, inputs):
        return self.dropout(inputs + self.pe[:, : inputs.size(1)])

class Model(nn.Module):
    def __init__(self, inputs_size, outputs_size):  
        super(Model, self).__init__()
        self.dim_up = nn.Linear(inputs_size, 512)  
        self.positional_encoding = PositionalEncoding(512, 0.) 
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=2)  
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)  
        self.predict = nn.Linear(512, outputs_size)  
        self.activation = nn.PReLU() 
        self.dropout = nn.Dropout(0.)  

    def transformer_encoder_forward(self, inputs):  
        
        outputs = inputs.permute(1, 0, 2)  
        outputs = self.transformer_encoder(outputs)  
        outputs = outputs.permute(1, 0, 2)  
        outputs = outputs.mean(dim=1)  
        return outputs

    def forward(self, inputs): 
       
        batch_size = inputs.size(0)
        inputs = inputs.view(batch_size, -1) 
        
       
        outputs = self.dim_up(inputs)  
       
        outputs = outputs.unsqueeze(1)  
        
        
        outputs = self.positional_encoding(outputs)  
        
        outputs = self.transformer_encoder_forward(outputs)  
        
        
        outputs = self.activation(outputs)  
        outputs = self.dropout(outputs)  
        outputs = self.predict(outputs) 
        return outputs






















train_dataset = TensorDataset(torch.Tensor(xTrain), torch.Tensor(yTrain))
test_dataset = TensorDataset(torch.Tensor(xTest), torch.Tensor(yTest))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Model(64, 784).to(device)
#model.load_state_dict(torch.load(f"/home/wwj/chenqiliang/model.pth"))




lr = 0.0001 
initial_lr=0.1
weight_decay = 0.0001  
betas = (0.95, 0.999)  
#eps = 1e-8
eps=1e-8
momentum=0.9
alpha=0.99
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)




criterion = nn.CrossEntropyLoss()
current_lr = optimizer.param_groups[0]['lr']
best_loss = float('inf') 


trainStart = time()
num_epochs = 20


 





 
for epoch in range(num_epochs):
    model.train()
    t = tqdm(train_loader, total=len(train_loader))

    for inputs, labels in t:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = torch.argmax(labels, dim=1)
        

        optimizer.zero_grad()
     

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
    if loss < best_loss:
        best_loss = loss.item()
        
        save_file_name = f"/home/wwj/chenqiliang/model.pth"  
        torch.save(model.state_dict(), save_file_name)

    print("epoch is {},loss is {}".format(epoch, loss))
    
    





















  
    


scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()




train = time() - trainStart



model = Model(64, 784).to(device)
model.load_state_dict(torch.load(f"/home/wwj/chenqiliang/model.pth"))
device = torch.device('cpu')
model.to(device)
ResNet_Pre1 = model(torch.from_numpy(xTest[0:40000]).to(device))




########################################################333wo xiede
testStart=time()
pre_array1 = np.zeros((40000, n_class))



index1=torch.argmax(ResNet_Pre1,dim=1)

for i in range(40000):
    pre_array1[i,index1[i]] = 1
  
    
  
test = time() - testStart   

aaa1 = torch.argmax(ResNet_Pre1, axis=1) + 1




ResNet_Pre_np1 = aaa1.numpy()



ResNet_Pre_np1_ = pre_array1


ResNet_Pre_np_=ResNet_Pre_np1_




############################################################3







###############################################################################




xTest_np = np.array(xTest[0:40000])




##################dui yu ce chuli

yTest_true_indices = np.argmax(yTest[:40000], axis=1)  
ResNet_Pre_np_indices = np.argmax(ResNet_Pre_np_, axis=1)  


acc = np.sum(ResNet_Pre_np_indices == yTest_true_indices) / 40000.0 * 100.0








'''

predicted_classes = torch.argmax(ResNet_Pre1, dim=1).numpy() 
true_classes = np.argmax(yTest[:40000], axis=1)  

accuracy = np.sum(predicted_classes == true_classes) / len(true_classes) * 100  
precision = precision_score(true_classes, predicted_classes, average='macro')  
recall = recall_score(true_classes, predicted_classes, average='macro')  
f1 = f1_score(true_classes, predicted_classes, average='macro')  
conf_matrix = confusion_matrix(true_classes, predicted_classes)  



print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
'''








#channel gain
I = np.eye(8)
I2 = np.eye(2)
Loss = []
Gain = []
for i in range(40000):
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)

    i1, i2, j1, j2 = GetIndexFrom(ResNet_Pre_np1[i])
    Pre_sub = mat(np.zeros((2, 2)), dtype=float)
    Pre_sub[0, 0] = ArrayA[i1, j1]
    Pre_sub[0, 1] = ArrayA[i1, j2]
    Pre_sub[1, 0] = ArrayA[i2, j1]
    Pre_sub[1, 1] = ArrayA[i2, j2]
    Pre_fullGian = math.sqrt(1 / 2) * np.linalg.norm(ArrayA, ord='fro')
    Pre_subGian = math.sqrt(1 / 2) * np.linalg.norm(Pre_sub, ord='fro')
    Gain.append(Pre_subGian)
    Loss.append(Pre_fullGian - Pre_subGian)

Gain_Mean = np.mean(Gain)
Loss_Mean = np.mean(Loss)
Loss_Variance = np.var(Loss)

print(Gain_Mean)
print(Loss_Mean)
print(Loss_Variance)

print("160000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("40000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))
print(f"{acc:.2f}%")

#channel capacity

'''
I1 = np.eye(8)
I2 = np.eye(2)

Pre_Loss = []

Pre_Capacity = []
for i in range(40000):
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)

    i1, i2, j1, j2 = GetIndexFrom(ResNet_Pre_np1[i])  
    Pre_sub = ArrayA[[i1, i2]][:, [j1, j2]]
    Pre_fullCapacity = np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 8))
    Pre_subCapacity= np.log2(np.linalg.det(I2 + a *  Pre_sub.T *  Pre_sub / 2))

    Pre_Capacity.append(Pre_subCapacity)
    Pre_Loss.append(Pre_fullCapacity - Pre_subCapacity)


Capacity_Mean = np.mean(Pre_Capacity)
Loss_Mean = np.mean(Pre_Loss)
Loss_Variance = np.var(Pre_Loss)


print(f"{acc:.2f}%")



print(Capacity_Mean)

print(Loss_Mean)
print(Loss_Variance)




print("160000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("40000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))
'''