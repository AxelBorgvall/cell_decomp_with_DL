import os
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from my_classes import my_networks,my_datasets
import matplotlib.pyplot as plt

base="../../advneu_proj/data/classified_cells"

finds="anaphase"
ignores=[]
for d in os.listdir(base):
    if d==finds:
        continue
    else:
        ignores.append(os.path.join(base,d))


mu=0.015
dataset = my_datasets.LocClassDataset([os.path.join(base,finds)],ignores,(400,400),mu,mu/2)

# Create DataLoader with random sampling with replacement
dataloader = DataLoader(dataset, batch_size=2, sampler=RandomSampler(dataset, replacement=True))


# Fetch a batch

model= my_networks.Unet(1, 1, layers=[32, 32,32,64],act=nn.ELU,norm=True)
loc= my_networks.LocalizerClassifier(model=model, n_transforms=10)


optim=torch.optim.Adam(loc.parameters(),lr=1e-5)
my_networks.train_localizer_classifier(loc=loc, dataloader=dataloader, optimizer=optim, epochs=6, filename="test_loclass_e25_interrupt_keep.pth")

torch.save(loc.state_dict(), "test_loclass_e25.pth")


