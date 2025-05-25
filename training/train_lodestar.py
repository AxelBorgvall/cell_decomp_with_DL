import torch
from my_classes import my_datasets,my_networks
from torch.utils.data import DataLoader

# Instantiate your dataset
dataset = my_datasets.SingleCellDataset("../../advneu_proj/SingleParticleImages/interphase-control", 120, repeat=5)

# Wrap it in a DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model= my_networks.Unet(1, 1, layers=[64, 128, 256])#this
loc= my_networks.Localizer(model=model, n_transforms=8)
optim=torch.optim.Adam(loc.parameters(),lr=0.0002)
my_networks.train_localizer(loc=loc, dataloader=loader, optimizer=optim, epochs=30, filename="../state_dicts/cell_localizer.pth")

torch.save(loc.state_dict(), "../state_dicts/cell_localizer.pth")
