import pandas as pd
import numpy as np

df = pd.read_csv('supp2.csv', na_values='#NUM!')
df = df.loc[:, df.any(axis=0)]  # drop columns with only 0 or #NUM!

df = df.dropna()  # drop rows with #NUM!

df['Drug ID'] = df['Drug ID'] - 1  # 0~95
df['Clone ID'] = df['Clone ID'] - 1  # 0~95
df['Drug ID'] = df['Drug ID'].apply(lambda x: x-48 if x>=48 else x)  # 0~47
df['Clone ID'] = df['Clone ID'].apply(lambda x: x-48 if x>=48 else x)  # 0~47

CAT = list(df.columns)[:2]
DENSE = list(df.columns)[2:]

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # uncomment to get reproducible results
seed = 0

import numpy as np
import torch
import random
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
    torch.backends.cudnn.benchmark = False
set_seed(seed)

import pandas as pd
import os
import datetime
from functools import reduce
from time import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



bs = batch_size = 32

df_train, df_test = train_test_split(df, test_size=0.8, random_state=seed)

scaler = StandardScaler()

X_dense = df_train[DENSE].to_numpy()
X_dense = scaler.fit_transform(X_dense)
df_train[DENSE] = X_dense

X_dense = df_test[DENSE].to_numpy()
X_dense = scaler.transform(X_dense)
df_test[DENSE] = X_dense

df_trn, df_vld = train_test_split(df_train, test_size=0.2, random_state=seed)
df_tst = df_test

def get_data(df, bs=bs, shuffle=True):
    X = df[CAT].to_numpy(dtype='int64')
    y = df[DENSE].to_numpy(dtype='float32')

    X, y = [torch.from_numpy(arr) for arr in [X, y]]
    
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=shuffle)
    return dataset, loader

trnset, trnloader = get_data(df_trn, bs, True)
vldset, vldloader = get_data(df_vld, bs, False)
tstset, tstloader = get_data(df_tst, bs, False)


class MF(nn.Module):
    def __init__(self, top_mlp_units, emb_dim, cat_counts, num_dense):
        super().__init__()
        
        num_cat = len(cat_counts)
        
        self.emb_dim = emb_dim
        self.num_cat = num_cat
        
        embs = [nn.Embedding(cnt, emb_dim) for cnt in cat_counts]
        self.embs = nn.ModuleList(embs)
        
        top_mlp = []
        prev = emb_dim * num_cat
        for units in top_mlp_units:
            top_mlp.append(nn.Linear(prev, units))
            top_mlp.append(nn.ReLU())
            prev = units
        top_mlp.append(nn.Linear(prev, num_dense))
        self.top_mlp = nn.Sequential(*top_mlp)
        
        self._initialize_weights()

    def forward(self, x):
        embs = []
        
        for i in range(self.num_cat):
            emb = self.embs[i](x[:, i])
            embs.append(emb)
        
        out = torch.cat(embs, dim=1)
        
        out = self.top_mlp(out)
            
        return out
    
    def _initialize_weights(self):  # same as keras
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.05, 0.05)



def train():
    model.train()
    trn_loss = 0.0
    trn_total = 0
    for data in trnloader:
        x = data[0].to(device)
        y = data[1].to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        cnt = x.size(0)
        trn_total += cnt
        trn_loss += loss.item() * cnt
    trn_loss /= trn_total
    return trn_loss

def test(dataloader):
    model.eval()
    vld_loss = 0.0
    vld_total = 0
    with torch.no_grad():
        for data in dataloader:
            x = data[0].to(device)
            y = data[1].to(device)
            pred = model(x)
            loss = criterion(pred, y)

            cnt = x.size(0)
            vld_total += cnt
            vld_loss += loss.item() * cnt
    vld_loss /= vld_total
    return vld_loss



emb_dim = 8
top_mlp_units = [32, 32]
cat_counts = [df[c].nunique() for c in CAT]
num_dense = len(DENSE)


print("top_mlp_units", top_mlp_units)
print("emb_dim",emb_dim)
print("cat_counts", cat_counts)
print("num_dense", num_dense)

model = MF(top_mlp_units, emb_dim, cat_counts, num_dense)
print(model)

x = next(iter(trnloader))[0]
print(x)
print(len(x))

summary(model, x)

lr = 0.001

model = model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

saved_trnloss = float('inf')
saved_vldloss = float('inf')
saved_tstloss = float('inf')

for epoch in range(300):
    start_t = time()
    trn_loss = train()
    vld_loss = test(vldloader)
    tst_loss = test(tstloader)
    end_t = time()
    print('Epoch %d trn_loss: %.4f vld_loss: %.4f tst_loss: %.4f Time: %d s' %
          (epoch, trn_loss, vld_loss, tst_loss, end_t-start_t))
    
    if vld_loss < saved_vldloss:
        saved_trnloss = trn_loss
        saved_vldloss = vld_loss
        saved_tstloss = tst_loss
#         torch.save({'net': model.state_dict()}, fn_ncf)

print('best model loss: {:.4f}, {:.4f}, {:.4f}'.format(saved_trnloss, saved_vldloss, saved_tstloss))
