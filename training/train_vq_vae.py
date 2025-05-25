import torch
from my_classes import my_datasets,my_networks
from torch.utils.data import DataLoader
if __name__=="__main__":
    imgloss = my_networks.DiffImageLoss(scaling=1.0, norm=True, lossfunc=torch.nn.functional.mse_loss)
    enc = my_networks.ConvDown(1, 64, [32, 64, 64], doubleconv=True, batchnorm=True)
    dec = my_networks.ConvUp(64, 1, [64, 64, 32], doubleconv=True, last_act_sig=False)
    model = my_networks.VQ_VAE(enc, dec, (1, 64, 64), imageloss=imgloss, num_embeddings=700, codebook_refresh_period=-1,codebook_usage_threshold=1)

    dataset= my_datasets.VaeDataset("path/to/dataset")

    loader=DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)

    my_networks.train_vq_vae(model, loader, optimizer, epochs=10)
    torch.save(model.state_dict(), "../state_dicts/VQ_VAE_small_sd.pth")

