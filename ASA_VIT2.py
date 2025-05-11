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
#a=1
a=1
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


#a=10





dataset = pd.read_csv(r'/home/wwj/chenqiliang/SVD/newAll-channel_matrix_p_1.csv').iloc[:, 1:]
#dataset = pd.read_csv(r'/home/wwj/chenqiliang/wubiaoqian/All-channel_matrix_p_20.csv').iloc[:, 1:]
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

def l2norm(t):
    return F.normalize(t, dim=-1, p=2)



class SimpleAttention(nn.Module):
    def __init__(self, dim, heads=2, dim_head=784, dropout=0.):
        super(SimpleAttention, self).__init__()

        assert dim % heads == 0, "dim must be divisible by heads"
        inner_dim = dim // heads

        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.relu = nn.Hardswish(dim)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3 * heads, bias=True)

        self.scale = nn.Parameter(torch.ones((heads, 1, 1)) * 0.01)

        self.attend = nn.Softmax(dim=-1)
        
      

        
        self.dropout = nn.Dropout(dropout)



        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim), 
            
            nn.Dropout(dropout)
        )
        
        
        self.ffn = nn.Sequential(
            nn.Linear(784,784),  
            #nn.ReLU(),
            nn.Hardswish(),
            nn.Dropout(dropout)
        )  
        
        
        
        

             
    
    def forward(self, x):
        h = self.heads
      
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # L2 Norm
        q, k = map(l2norm, (q, k))

        # Attention scores
        sim = (q @ k.transpose(-2, -1)) * self.scale.sqrt()
        attn = self.attend(sim)
        
        attn = self.dropout(attn)  

        # Compute the output
        out = (attn @ v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.norm(out)
        out = self.relu(out)
        out_ffn = self.ffn(out)  
       

        # Final output layer
        return self.to_out(out+out_ffn)










class PatchEmbed(nn.Module):
    def __init__(self, img_size=8, patch_size=8, in_chans=1, embed_dim=784):    ##################
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
       
        self.grid_size = img_size // patch_size
     
        self.num_patches = self.grid_size * self.grid_size
        
      
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=8, stride=8)
        
        
       

    def forward(self, x):
        x = self.proj(x)  
        x = x.flatten(2)  
        x = x.transpose(1, 2) 
  
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=784, num_heads=1, mlp_ratio=0.2, depth=1, qkv_bias=True,):    ##############
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    nn.LayerNorm(embed_dim),#0
                    
                  
                    
                 
                    SimpleAttention(embed_dim, num_heads),
                   
                    #nn.LayerNorm(embed_dim),
                    #nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),  # First FC layer of MLP
                    #nn.ReLU(),
                    nn.Hardswish(),
                    #nn.ReLU6(),  # Activation function
                    #nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
             
                   
            
                   
                   
         
                  
                   
                    nn.Dropout(0.),#4
                    
                  
                ])
            )






    def forward(self, x):
        for layer in self.layers:
            x_norm = layer[0](x)  # LayerNorm
            attn_output = layer[1](x_norm)  # Attention
            x = attn_output + x  # Residual connection
            
            #x_norm = layer[2](x)  # Another LayerNorm
            #mlp_output = layer[3](x_norm)  # First Linear layer of MLP
           
            #mlp_output = layer[4](mlp_output)  # ReLU activation
            #mlp_output = layer[5](mlp_output)
            #mlp_output = layer[6](mlp_output)
         # Second Linear layer of MLP
            # Dropout

            #x = mlp_output + x  # Residual connection for MLP
        
        return x









class VisionTransformer(nn.Module):
    def __init__(self, img_size=8, patch_size=8, in_chans=1, num_classes=784, embed_dim=784, depth=1, num_heads=1,
                 mlp_ratio=0.2, qkv_bias=True,  ):   ###############################
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        
        
       #################################################
       

       
       
       ###########################################
        #self.pos_relu = nn.ReLU6()
        #self.pos_relu = nn.ReLU6()
        self.pos_relu = nn.Hardswish()
        self.pos_drop = nn.Dropout(0.)
        
        self.encoder = TransformerEncoder(embed_dim, num_heads, mlp_ratio, depth, qkv_bias)
        self.norm = nn.LayerNorm(embed_dim)
    
        self.head = nn.Linear(embed_dim, num_classes)  
        #self.relu = nn.Sigmoid()  
        self.relu = nn.ReLU()
     
    def forward(self, x):
        x = x.reshape(-1, 1, 8, 8)
        B = x.shape[0]
        x = self.patch_embed(x)
        
        

        
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.pos_drop(x)
        
       
       
       
        
        x = self.encoder(x)
            
        x = self.norm(x)
     
        cls_token_final = x[:, 0]  
        x = self.head(cls_token_final)  
        return x


if __name__ == '__main__':

   
    img_size = 8
    patch_size = 8
    in_chans = 1
    embed_dim = 784    ##################
    depth = 1
    num_heads = 1
    num_classes = 784

    vit = VisionTransformer(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                            num_classes=num_classes)
 
    print(vit)

    
    #x = torch.randn(1, 3, 224, 8) 
    x = torch.randn(16, 1, 8,8)
    
    output = vit(x)

   
    print(f"Input shape: {x.shape}")  
    print(f"Output shape: {output.shape}")  



























































train_dataset = TensorDataset(torch.Tensor(xTrain), torch.Tensor(yTrain))
test_dataset = TensorDataset(torch.Tensor(xTest), torch.Tensor(yTest))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                            num_classes=num_classes).to(device)
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



model = VisionTransformer(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                            num_classes=num_classes)
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

label_array = np.zeros((n_sample, n_class))


for i in range(n_sample):
    label_array[i, label[i] - 1] = 1

###################

yTest_indices = np.array(yTest[:40000])



b=np.all(ResNet_Pre_np_ == yTest_indices, axis=1)

acc = np.sum(b) / 40000.0 * 100.0


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









rint("SNR=0")

print("160000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("40000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))



print(f"{acc:.2f}%")

print("Gain_Mean %s" % (Gain_Mean))
print("Loss_Mean %s" % (Loss_Mean))
print("Loss_Variance %s" %(Loss_Variance))

'''




'''

fullChainCapacity = pd.read_csv(r'/home/wwj/chenqiliang/SVD/newAll_channel_capacity_p_1.csv').iloc[:, 1:]
fullChainCapacity = fullChainCapacity[0:200000]
fullChainCapacity  = np.asarray(fullChainCapacity , np.float32)

subChainCapacity  = pd.read_csv(r'/home/wwj/chenqiliang/SVD/newSub_channel_capacity_p_1.csv').iloc[:, 1:]
subChainCapacity  = subChainCapacity[0:200000]
subChainCapacity  = np.asarray(subChainCapacity, np.float32)

fullChainCapacity_Mean = np.mean(fullChainCapacity)
subChainCapacity_Mean = np.mean(subChainCapacity)
'''











#channel capacity
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











print("SNR=0")

print("160000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("40000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))



print(f"{acc:.2f}%")

print("Capacity_Mean %s" % (Capacity_Mean))
print("Loss_Mean %s" % (Loss_Mean))
print("Loss_Variance %s" %(Loss_Variance))














































