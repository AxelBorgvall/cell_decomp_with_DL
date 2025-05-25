import torch
from my_classes import my_datasets,my_networks
from torch.utils.data import DataLoader

imgloss= my_networks.DiffImageLoss(scaling=1.0, norm=True, lossfunc=torch.nn.functional.l1_loss)
model= my_networks.VAE(inputshape=(64, 64), latent_dim=245, convchannels=[64, 128, 256], fc_layers=[2048, 1024], beta=1.0)
#model=myNets.VAE(inputshape=(64,64),latent_dim=30,convchannels=[16,32,64],fc_layers=[1024,512,256],beta=1.0,imageloss=imgloss)

dataset= my_datasets.VaeDataset("path/to/dataset")

loader=DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
if __name__=="__main__":
    my_networks.train_vae(model, loader, optimizer, epochs=25)
    torch.save(model.state_dict(), "../state_dicts/VAE_sd.pth")



