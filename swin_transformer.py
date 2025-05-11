import torch
import torch.nn as nn
from uitls_8822_capacity import computation_time
from einops.layers.torch import Rearrange

import numpy as np
import pandas as pd
from torch.optim import AdamW

import math
from time import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch_lr_finder import LRFinder
from numpy import mat
import numbers
import matplotlib.pyplot as plt
from scipy.special import erfc  
from torch.nn import Softmax
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d,ReLU6,GELU






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

#a=10
#a=10
#a=10**1.5
a=1
Nt = 1
P_t =0.01
P_A =0.35
N_s =2
P_ct =0.0344 
N_r =2
P_cr =0.0625
P_sync = 0.050
B_W = 10000000


#a=10



dataset = pd.read_csv(r'/home/wwj/chenqiliang/SVD/newAll-channel_matrix_p_1.csv').iloc[:, 1:]

dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8, 1)

label = pd.read_csv(r'/home/wwj/chenqiliang/SVD/newcapacity_labels_p_1.csv').iloc[:, 1]









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























import torch
import torch.nn as nn
import torch.nn.functional as F






from einops import rearrange

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1)**2, num_heads))
        
       
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords), dim=0)
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1))))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size **2, self.window_size**2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.shift_size = shift_size
        self.window_size = window_size

    def forward(self, x):
        H = W = int(x.shape[1]** 0.5)
        B, L, C = x.shape
        
       
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        
        x_windows = shifted_x.view(B, H // self.window_size, self.window_size,
                                  W // self.window_size, self.window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        
      
        attn_windows = self.attn(x_windows)
        
        
        attn_windows = attn_windows.view(B, H // self.window_size, W // self.window_size,
                                        self.window_size, self.window_size, C)
        shifted_x = attn_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * W, C)
        
      
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        
        x = x + self.mlp(self.norm2(x))
        return x

class SwinTransformer(nn.Module):
    def __init__(self, img_size=8, in_chans=1, num_classes=784, embed_dim=512, 
                depths=[1], num_heads=[1], window_size=1):
        super().__init__()
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=8, stride=8)
        patches_resolution = img_size // 8
        num_patches = patches_resolution **2
        
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
       
        self.blocks = nn.ModuleList()
        for i_layer in range(len(depths)):
            for _ in range(depths[i_layer]):
                self.blocks.append(SwinBlock(
                    dim=embed_dim,
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i_layer % 2 == 0) else window_size // 2
                ))
        
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, C]
        x = x + self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)  
        x = self.head(x)
        return x
        
        
        
        
        
        
        








































train_dataset = TensorDataset(torch.Tensor(xTrain), torch.Tensor(yTrain))
test_dataset = TensorDataset(torch.Tensor(xTest), torch.Tensor(yTest))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model =  SwinTransformer().to(device)
#model.load_state_dict(torch.load(f"/home/wwj/chenqiliang/model.pth"))



lr = 0.0001 
#initial_lr=0.1
weight_decay = 0.001  
betas = (0.95, 0.999)  
#eps = 1e-8
eps=1e-8

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


scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()



train = time() - trainStart



model =  SwinTransformer()
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
'''

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

print("acc is %.6f "%(acc))


print("160000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("40000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))
'''


#channel capacity


'''
I = np.eye(8)
I2 = np.eye(2)
Loss = []
Gain = []



for i in range(40000):
  
    three_channel = xTest_np[i]
    
  
    real_part = three_channel[:, :, 0]  
    imag_part = three_channel[:, :, 1] 
    complex_matrix = real_part + 1j * imag_part 
    
  
    ArrayA = np.matrix(complex_matrix)  

    
    i1, i2, j1, j2 = GetIndexFrom(ResNet_Pre_np1[i])
    Pre_sub = mat(np.zeros((2, 2)), dtype=complex)  
    Pre_sub[0, 0] = ArrayA[i1, j1]
    Pre_sub[0, 1] = ArrayA[i1, j2]
    Pre_sub[1, 0] = ArrayA[i2, j1]
    Pre_sub[1, 1] = ArrayA[i2, j2]
    

    Pre_fullGian = np.linalg.norm(ArrayA, ord='fro') * math.sqrt(1/2)
    Pre_subGian = np.linalg.norm(Pre_sub, ord='fro') * math.sqrt(1/2)


    Gain.append(Pre_subGian)
    Loss.append(Pre_fullGian - Pre_subGian)

Gain_Mean = np.mean(Gain)
Loss_Mean = np.mean(Loss)
Loss_Variance = np.var(Loss)

print("160000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("40000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))

print(Gain_Mean)
print(Loss_Mean)
print(Loss_Variance)

print(f"{acc:.2f}%")
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


#print("acc is %.6f "%(acc))
print(f"{acc:.2f}%")
print("160000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("40000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))

print(Capacity_Mean)




print(Loss_Mean)
print(Loss_Variance)
