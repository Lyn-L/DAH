import sys
import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import scipy.io as sio
import matplotlib.pyplot as plt
import dannet
import dataiter
import numpy as np
import datetime
import h5py


# Load Dataset
data = sio.loadmat('./myWiki.mat')
data2 = sio.loadmat('./myWiki_lsh.mat')

Tr_64 = torch.FloatTensor(data2['Tr_64'])
Te_64 = torch.FloatTensor(data2['Te_64'])

data['L_te'] = np.array(data['L_te'], dtype='int64')
L_te = Variable(torch.LongTensor(data['L_te']))
data['L_tr'] = np.array(data['L_tr'], dtype='int64')
L_tr = Variable(torch.LongTensor(data['L_tr']))

training_incomplete = torch.FloatTensor(data['training_incomplete'])
training_complete = torch.FloatTensor(data['training_complete'])
data['training_label'] = np.array(data['training_label'], dtype='int64')
training_label = torch.LongTensor(data['training_label'])

test_incomplete = torch.FloatTensor(data['test_incomplete'])
test_complete = torch.FloatTensor(data['test_complete'])
data['test_label'] = np.array(data['test_label'], dtype='int64')
test_label = torch.LongTensor(data['test_label'])

training_I = torch.FloatTensor(data['training_I'])
data['training_T'] = np.array(data['training_T'], dtype = 'double')
training_T = torch.FloatTensor(data['training_T'])
test_I = torch.FloatTensor(data['test_I'])
data['test_T'] = np.array(data['test_T'], dtype = 'double')
test_T = torch.FloatTensor(data['test_T'])

dataset = dataiter.TensorDataset3(training_incomplete, training_complete, Tr_64, training_label)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=True, drop_last=True)
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)

# network
dan = dannet.DAN(dim_feat=138, dim_latent=10000, hash_bit=64, label=10).cuda()

# optimizer
optimizer = optim.Adam(dan.parameters(), lr=0.1, betas=(0.9, 0.999), weight_decay=1e-6)

# training
dan.train()
for epoch in range(250):
    if  epoch % 150 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    if  epoch % 200 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        loss = dan(Variable(batch[0]).cuda(), Variable(batch[1]).cuda(), Variable(batch[2]).cuda(), Variable(batch[3]).cuda())
        loss.backward()
        optimizer.step()

        tr_loss =  loss.cpu().data.numpy()

        s ="\r epoch: %d =========== %s"%(epoch, tr_loss)
        sys.stdout.write(s)
        sys.stdout.flush()

# evaluate
dan = dan.eval().cpu()
training_I = dan.hash(Variable(training_I))
training_T = dan.hash(Variable(training_T))
test_I = dan.hash(Variable(test_I))
test_T = dan.hash(Variable(test_T))
sio.savemat('./mm2018_myWiki_64.mat', {'training_T':training_T.data.numpy(), 'training_I':training_I.data.numpy(), 'test_T':test_T.data.numpy(), 'test_I':test_I.data.numpy()})

