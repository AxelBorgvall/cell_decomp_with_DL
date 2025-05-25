import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import translate,rotate,warp_affine
from tqdm import tqdm

class DoubleConv(nn.Module):
    """(Conv => ReLU => Conv => ReLU)"""
    def __init__(self, in_channels, out_channels,act=nn.ReLU,norm=False):
        super().__init__()
        if not norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                act(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                act()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                act(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                act()
            )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels,scaling=2,act=nn.ReLU,norm=False):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(scaling),
            DoubleConv(in_channels, out_channels,act,norm)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True,scaling=2,act=nn.ReLU,norm=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=scaling, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels,act,norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=scaling)
            self.conv = DoubleConv(in_channels, out_channels,act,norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad x1 if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, layers=[64, 128, 256, 512], bilinear=True, scaling=2,act=nn.ReLU,norm=False):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.downs=nn.ModuleList()
        self.ups=nn.ModuleList()

        self.inc=DoubleConv(n_channels, layers[0],norm=norm)
        self.outc=nn.Sequential(nn.Conv2d(layers[0], n_classes, kernel_size=1),nn.ReLU())

        self.nlayers=len(layers)

        self.xlist=[None]*self.nlayers
        for i in range(self.nlayers-1):
            self.downs.append(Down(layers[i], layers[i + 1], scaling=scaling,act=act,norm=norm))
            self.ups.append(Up(layers[self.nlayers-1-i]+layers[self.nlayers-2-i],layers[self.nlayers-2-i],scaling=scaling,bilinear=bilinear,act=act,norm=norm))
        self.to(self.device)


    def forward(self, x):
        self.xlist[0]=self.inc(x)
        for i in range(self.nlayers-1):
            self.xlist[i+1]=self.downs[i](self.xlist[i])
        x=self.xlist[-1]
        for i in range(self.nlayers-1):
            x=self.ups[i](x,self.xlist[-i-2])
        return self.outc(x)


# LodeSTAR definition---------------------------------------------------------------------------------------------------------------------

def mass_centroid(tensor):
    # tensor: (B, C, H, W)
    B, C, H, W = tensor.shape

    device = tensor.device

    y_coords = torch.linspace(-H / 2, H / 2, steps=H, device=device).float()
    x_coords = torch.linspace(-W / 2, W / 2, steps=W, device=device).float()
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H, W)

    x_grid = x_grid.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    y_grid = y_grid.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    mass = tensor.sum(dim=(-2, -1), keepdim=True)  # (B, C, 1, 1)
    mass = mass + 1e-8

    x_centroid = (tensor * x_grid).sum(dim=(-2, -1), keepdim=False) / mass.squeeze(-1).squeeze(-1)
    y_centroid = (tensor * y_grid).sum(dim=(-2, -1), keepdim=False) / mass.squeeze(-1).squeeze(-1)

    centroids = torch.stack((x_centroid, y_centroid), dim=-1)
    return centroids


def image_translation(batch, translation):
    return translate(batch, translation)


def inverse_translation(preds, applied_translation):
    return preds - applied_translation


def image_rotation(batch, angles):
    return rotate(batch, angles)


def inverse_rotation(preds, angles):
    cosines = torch.cos(angles * (torch.pi / 180))
    sines = torch.sin(angles * (torch.pi / 180))

    R = torch.stack([
        torch.stack([cosines, -sines], dim=1),
        torch.stack([sines, cosines], dim=1)
    ], dim=1)

    return torch.bmm(R, preds.unsqueeze(2)).squeeze(2)  # (n,2)


def image_affine_transform(image: torch.Tensor, affine_matrix: torch.Tensor) -> torch.Tensor:
    B, C, H, W = image.shape

    center = torch.tensor([W / 2, H / 2], device=image.device).view(1, 1, 2)
    # creating matrix for transform to centered coords
    T_center = torch.eye(3, device=image.device).unsqueeze(0).repeat(B, 1, 1)
    T_center[:, 0, 2] = -center[:, 0, 0]
    T_center[:, 1, 2] = -center[:, 0, 1]
    # creating matrix for transform to uncentered coords
    T_uncenter = torch.eye(3, device=image.device).unsqueeze(0).repeat(B, 1, 1)
    T_uncenter[:, 0, 2] = center[:, 0, 0]
    T_uncenter[:, 1, 2] = center[:, 0, 1]

    # adding bottom part to affine mat
    A = torch.cat([affine_matrix, torch.tensor([[[0., 0., 1.]]], device=image.device).repeat(B, 1, 1)], dim=1)

    # making compound transform of centering transforming and uncetering
    A_total = T_uncenter @ A @ T_center
    A_total = A_total[:, :2, :]  # warp_affine expects (B, 2, 3)

    return warp_affine(image, A_total, dsize=(H, W), align_corners=False)


def forward_warp(
        pts: torch.Tensor,  # [N,2], in centered coords
        affine_matrix: torch.Tensor) -> torch.Tensor:
    """function for transforming a batch of 2d points through an affine matrix"""
    N = pts.shape[0]
    device = pts.device

    # make 3x3 from 2x3
    A = torch.eye(3, device=device).unsqueeze(0).repeat(N, 1, 1)  # [N,3,3]
    A[:, :2, :] = affine_matrix

    # add ones to bottom for the affine transform
    hom = torch.cat([pts, torch.ones(N, 1, device=device)], dim=1)  # [N,3]

    # do affine transform
    out = (A @ hom.unsqueeze(-1)).squeeze(-1)  # [N,3]
    return out[:, :2]


def inverse_warp(
        pts: torch.Tensor,
        affine_matrix: torch.Tensor) -> torch.Tensor:
    N = pts.shape[0]
    device = pts.device

    # make 3x3 from 2x3
    A = torch.eye(3, device=device).unsqueeze(0).repeat(N, 1, 1)
    A[:, :2, :] = affine_matrix  # [N,3,3]

    # invert
    A_inv = torch.inverse(A)  # [N,3,3]

    # add ones
    hom = torch.cat([pts, torch.ones(N, 1, device=device)], dim=1)  # [N,3]

    # apply inverse transform
    out = (A_inv @ hom.unsqueeze(-1)).squeeze(-1)  # [N,3]

    return out[:, :2]


class Localizer(nn.Module):
    def __init__(self, model, n_transforms=8, **kwargs):
        super(Localizer, self).__init__()
        """
        Setting model by default is bad practice. I learned this after writing this class
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.n_transforms = n_transforms
        return

    def forward(self, x):
        return self.model(x.to(self.device))

    def forward_tranform(self, batch, translation, angles):
        transformed = image_translation(batch, translation)
        return image_rotation(transformed, angles)

    def inverse_tranform(self, pred, translation, angles):
        invpred = inverse_rotation(pred, angles)
        return inverse_translation(invpred, translation)

    def get_loss(self, image):
        b, c, h, w = image.shape
        # expanding for transform
        images = image.unsqueeze(1).expand(-1, self.n_transforms, -1, -1, -1).contiguous()

        # flattening to feed into network
        flat = images.view(b * self.n_transforms, c, h, w)

        # getting random args
        tr = torch.rand(b, self.n_transforms, 2, device=images.device) * h // 3 - h // 6  # translations
        ag = torch.rand(b, self.n_transforms, device=images.device) * 360  # angles
        # flatten random args
        tr_flat = tr.view(b * self.n_transforms, 2)
        ag_flat = ag.view(b * self.n_transforms, )

        # transform images, run model
        transform_im = self.forward_tranform(flat, tr_flat, ag_flat)
        pred_flat = self.model(transform_im)

        centroids_flat = mass_centroid(pred_flat)

        # invert transforms
        invpred = self.inverse_tranform(centroids_flat.squeeze(), tr_flat, ag_flat)
        invpred = invpred.view(b, self.n_transforms, 2)

        # invpred: [B, T, 2]

        diffs = invpred[:, 1:, :] - invpred[:, :-1, :]  # [B, T-1, 2]
        mse_per_sample = torch.mean((diffs ** 2).sum(dim=-1), dim=1)  # [B]
        return mse_per_sample.sum()


class LocalizerClassifier(nn.Module):
    def __init__(self, model: nn.Module, n_transforms: int = 8, baseoffset: int = 120, affine_warp=0.4):
        super(LocalizerClassifier, self).__init__()
        self.model = model
        self.n_transforms = n_transforms
        self.offset = baseoffset
        self.warp = affine_warp
        return

    def forward(self, x):
        return self.model(x)

    def forward_transform(self, batch, translation, angles, affine, ignore):
        # batch: (N, 1, H, W)
        # ignore: (N, 1, h, w)
        # translation: (N, 2)

        _, _, H, W = batch.shape
        N, _, h, w = ignore.shape

        # Apply translation
        transformed = image_translation(batch, translation)

        # Compute offset where to paste the ignore image
        # self.offset should be a scalar or tensor like (2,) for max translation
        noise = torch.rand(N, 2, device=ignore.device) * (self.offset // 1.5)
        ignore_offset = translation - self.offset + noise
        ignore_offset[:, 0] += (H - h) // 2  # y
        ignore_offset[:, 1] += (W - w) // 2  # x

        # Round and convert to int
        ignore_offset = ignore_offset.round().to(dtype=torch.long)

        # Vectorized paste of ignore into transformed
        patch_y = torch.arange(h, device=ignore.device).view(1, h, 1).expand(N, h, w)
        patch_x = torch.arange(w, device=ignore.device).view(1, 1, w).expand(N, h, w)
        offset_y = ignore_offset[:, 0].view(-1, 1, 1)
        offset_x = ignore_offset[:, 1].view(-1, 1, 1)
        target_y = patch_y + offset_y
        target_x = patch_x + offset_x

        # Mask for clipping if necessary (optional)
        in_bounds = (target_y >= 0) & (target_y < H) & (target_x >= 0) & (target_x < W)

        # Flatten indices for scatter
        batch_idx = torch.arange(N, device=ignore.device).view(N, 1, 1).expand(N, h, w)
        channel_idx = torch.zeros_like(batch_idx)

        # Masked paste (ignore pixels out of bounds)
        transformed[batch_idx[in_bounds], channel_idx[in_bounds], target_y[in_bounds], target_x[in_bounds]] = \
        ignore.squeeze(1)[in_bounds]

        transformed = image_affine_transform(transformed, affine)

        # commented code is for plotting the resulting transformed image, helpful for troubleshooting
        '''
        #------------------------------------------------
        example_affine=torch.rand(N,2,3,device=transformed.device)*0.4-0.2+torch.tensor([[1,0,0],[0,1,0]],dtype=torch.float32,device=transformed.device).unsqueeze(0).repeat(N,1,1)
        #first affine then rotation
        output=image_affine_transform(transformed,example_affine)
        output=image_rotation(output, angles).cpu().detach()
        #first affine then rotation
        coords=forward_warp(translation,example_affine)
        coords=inverse_rotation(coords,-angles)

        for i in range(len(output)):
            plt.imshow(output[i].squeeze(),cmap="gray")
            plt.scatter(coords[i,0].cpu().detach().squeeze()+W//2,coords[i,1].cpu().detach().squeeze()+H//2)
            plt.axis("off")
            plt.show()
        assert 1==0
        #------------------------------------------------
        '''

        return image_rotation(transformed, angles)

    def inverse_tranform(self, pred, translation, angles, affine):
        invpred = inverse_rotation(pred, angles)
        invpred = inverse_warp(invpred, affine)
        return inverse_translation(invpred, translation)

    def get_loss(self, detect, ignore):
        b, c, H, W = detect.shape
        _, _, h, w = ignore.shape
        # expanding for transform
        images = detect.unsqueeze(1).expand(-1, self.n_transforms, -1, -1, -1).contiguous()
        ignores = ignore.unsqueeze(1).expand(-1, self.n_transforms, -1, -1, -1).contiguous()
        # flattening to feed into network
        flat = images.view(b * self.n_transforms, c, H, W)
        ignores = ignores.view(b * self.n_transforms, c, h, w)

        # getting random args
        tr = torch.rand(b, self.n_transforms, 2, device=images.device) * h // 2
        ag = torch.rand(b, self.n_transforms, device=images.device) * 360  # angles
        af = torch.rand(b, self.n_transforms, 2, 3, device=images.device) * self.warp - self.warp / 2 + \
             torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32, device=images.device).unsqueeze(0).unsqueeze(
                 0).repeat(b, self.n_transforms, 1, 1)

        # flatten random args
        tr_flat = tr.view(b * self.n_transforms, 2)
        ag_flat = ag.view(b * self.n_transforms, )
        af_flat = af.view(b * self.n_transforms, 2, 3)

        # transform images, run model
        transform_im = self.forward_transform(flat, tr_flat, ag_flat, af_flat, ignores)
        pred_flat = self.model(transform_im)

        # compute mass centroids
        centroids_flat = mass_centroid(pred_flat)

        # invert transforms
        invpred = self.inverse_tranform(centroids_flat.squeeze(), tr_flat, ag_flat, af_flat)
        invpred = invpred.view(b, self.n_transforms, 2)

        # invpred: [B, T, 2]

        diffs = invpred[:, 1:, :] - invpred[:, :-1, :]  # [B, T-1, 2]
        mse_per_sample = torch.mean((diffs ** 2).sum(dim=-1), dim=1)  # [B]

        return mse_per_sample.sum()


def train_localizer(loc, dataloader, optimizer, epochs=300, filename="filename"):
    loc.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loc = loc.to(device)
    try:
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for inputs in dataloader:
                inputs = inputs.to(device)  # [B, C, H, W]

                optimizer.zero_grad()
                loss = loc.get_loss(inputs)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)  # sum up batch loss

            avg_loss = epoch_loss / len(dataloader.dataset)
            print(f"Epoch {epoch:3d}/{epochs}, avg loss: {avg_loss:.4f}")


    except KeyboardInterrupt:
        print("\n Training manually quit")
    finally:
        torch.save(loc.state_dict(), filename)
        print(f"Model saved to {filename}")


def train_localizer_classifier(loc, dataloader, optimizer, epochs=300, filename="filename"):
    loc.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loc = loc.to(device)

    try:
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0

            # Wrap the dataloader in tqdm for batch-level progress
            pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}/{epochs}", leave=False)
            for images, ignores in pbar:
                images = images.to(device)
                ignores = ignores.to(device)

                optimizer.zero_grad()
                loss = loc.get_loss(images, ignores)
                loss.backward()
                optimizer.step()

                batch_loss = loss.item() * images.size(0)
                epoch_loss += batch_loss

                avg_loss = epoch_loss / len(dataloader.dataset)
                pbar.set_postfix(loss=avg_loss)

            print(f"Epoch {epoch:3d}/{epochs}, avg loss: {avg_loss:.4f}")

    except KeyboardInterrupt:
        print("\nTraining manually quit")
    finally:
        torch.save(loc.state_dict(), filename)
        print(f"Model saved to {filename}")

#VAE--------------------------------------------------------------------------------------------

class ReshapeLayer(nn.Module):
    def __init__(self, channels, height, width):
        super(ReshapeLayer, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        #reshape the tensor back to [batch_size, channels, height, width]
        return x.view(-1, self.channels, self.height, self.width)

def normalize_tensor(tens):
    return (tens-torch.mean(tens,dim=(-1,-2)).view(-1,1,1,1))/torch.std(tens,dim=(-1,-2)).view(-1,1,1,1)

class DiffImageLoss(nn.Module):
    def __init__(self,scaling=1,norm=False,lossfunc=F.mse_loss):
        super().__init__()
        self.scale=scaling
        self.normalize=norm
        self.loss=lossfunc
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1],
                                                      [-1, 0, 1],
                                                      [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', torch.tensor([[-1, -1, -1],
                                                      [0, 0, 0],
                                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    def forward(self, input, recon):
        # Gradient loss
        grad_input_x = F.conv2d(input, self.sobel_x, padding=1)
        grad_input_y = F.conv2d(input, self.sobel_y, padding=1)
        grad_recon_x = F.conv2d(recon, self.sobel_x, padding=1)
        grad_recon_y = F.conv2d(recon, self.sobel_y, padding=1)

        if self.normalize:
            input=normalize_tensor(input)
            recon=normalize_tensor(recon)

            grad_recon_x=normalize_tensor(grad_recon_x)
            grad_recon_y=normalize_tensor(grad_recon_y)
            grad_input_x=normalize_tensor(grad_input_x)
            grad_input_y=normalize_tensor(grad_input_y)

        pixel_loss = self.loss(recon, input,reduction="sum")
        grad_loss = self.loss(grad_input_x, grad_recon_x,reduction="sum")/2 + self.loss(grad_input_y, grad_recon_y,reduction="sum")/2

        return pixel_loss + self.scale * grad_loss
# Simple VAE class
class VAE(nn.Module):
    def __init__(self,inputshape,latent_dim,convchannels=[16,32],fc_layers=[512,256],beta=0.1,imageloss=DiffImageLoss(0.5,True)):
        super(VAE, self).__init__()
        self.beta=beta
        self.image_loss=imageloss
        self.conv_dim = (
        convchannels[-1], inputshape[0] // (2 ** len(convchannels)), inputshape[1] // (2 ** len(convchannels)))
        #Loop over convchannels and append conv maxpool/conv upscale to lists
        convchannels.insert(0,1)
        down=[]

        up=[]
        for i in range(len(convchannels)-1):
            down.append(nn.Conv2d(convchannels[i],convchannels[i+1],kernel_size=(3,3),padding=1))
            down.append(nn.ReLU())
            down.append(nn.MaxPool2d(kernel_size=2, stride=2))

            if not i==0:
                up.append(nn.ReLU())
            else:
                up.append(nn.Sigmoid())
            up.append(nn.Conv2d(convchannels[i+1],convchannels[i],kernel_size=(3,3),padding=1))
            up.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))


        down.append(torch.nn.Flatten(start_dim=1))

        up.append(ReshapeLayer(*self.conv_dim))
        up.append(nn.ReLU())
        up.append(nn.Linear(fc_layers[0],self.conv_dim[0]*self.conv_dim[1]*self.conv_dim[2] ))
        up.reverse()

        #Loop over fc_layers and add to list
        down_linear=[]
        up_linear=[]
        for i in range(len(fc_layers)):
            down_linear.append(nn.LazyLinear(fc_layers[i]))
            down_linear.append(nn.ReLU())

            up_linear.append(nn.ReLU())
            up_linear.append(nn.LazyLinear(fc_layers[i]))

        #up_linear.append(nn.Linear(latent_dim,fc_layers[-1]))
        up_linear.reverse()

        self.mu=nn.Linear(fc_layers[-1],latent_dim)
        self.logvar=nn.Linear(fc_layers[-1],latent_dim)

        #Turn into sequentials for decode and encode
        self.down=nn.Sequential(*down,*down_linear)
        self.decode=nn.Sequential(*up_linear,*up)

        dummy=torch.zeros(1,1,*inputshape)
        self.forward(dummy)


    def encode(self, x):
        h=self.down(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_loss(self,recon_x, x,logvar,mu):
        #Perform inverse rotation to judge reconstruction in fixed reference direction
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return  self.image_loss(recon_x,x) +self.beta * KLD


class ConvDown(nn.Module):
    def __init__(self, inputchannels, outputchannels, channels=[16, 32], maxpool=True, batchnorm=False,
                 doubleconv=False):
        super(ConvDown, self).__init__()
        self.conv = []
        if not channels[0] == inputchannels:
            channels.insert(0, inputchannels)
        if not channels[-1] == outputchannels:
            channels.append(outputchannels)
        for i in range(len(channels) - 1):
            self.conv.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=(3, 3), padding=(1, 1)))
            if doubleconv:
                if batchnorm:
                    self.conv.append(nn.BatchNorm2d(channels[i + 1]))
                self.conv.append(nn.GELU())
                self.conv.append(nn.Conv2d(channels[i + 1], channels[i + 1], kernel_size=(3, 3), padding=(1, 1)))
            if batchnorm:
                self.conv.append(nn.BatchNorm2d(channels[i + 1]))
            self.conv.append(nn.GELU())
            if maxpool:
                self.conv.append(nn.MaxPool2d((2, 2), 2))
            else:
                nn.Conv2d(i + 1, i + 1, kernel_size=(4, 4), stride=(2, 2), padding=1)

        self.conv = nn.Sequential(*self.conv)

    def forward(self, inputs):
        return self.conv(inputs)


class ConvUp(nn.Module):
    def __init__(self, inputchannels, outputchannels, channels=[32, 16], doubleconv=False, last_act_sig=False):
        super(ConvUp, self).__init__()
        self.conv = []
        if not channels[0] == inputchannels:
            channels.insert(0, inputchannels)
        if not channels[-1] == outputchannels:
            channels.append(outputchannels)
        for i in range(len(channels) - 1):
            self.conv.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            if doubleconv:
                self.conv.append(nn.Conv2d(channels[i], channels[i], kernel_size=(3, 3), padding=(1, 1)))
                self.conv.append(nn.GELU())
            self.conv.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=(3, 3), padding=(1, 1)))

            if last_act_sig and i == len(channels) - 2:
                self.conv.append(nn.Sigmoid())
            else:
                self.conv.append(nn.GELU())

        self.conv = nn.Sequential(*self.conv)

    def forward(self, inputs):
        return self.conv(inputs)


class VQLookUpTable(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, refresh_every=-1, usage_threshold=1):
        super().__init__()
        # create lookup table
        self.n_emb = num_embeddings
        self.emb_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # initialize with normal embeddings
        nn.init.normal_(self.embedding.weight, std=1.0)

        # forward calls between codebook refreshes
        self.refresh_timer = refresh_every
        self.counter = 0
        self.usage_threshold = usage_threshold
        self.refresh = (refresh_every > 0)

        self.commitment_cost = commitment_cost
        self.register_buffer('usage', torch.zeros(self.n_emb, dtype=torch.long))

    def embedding_indices(self, inputs):
        # inputs: (B, D, H, W)
        # flatten input to shape (B*H*W, D)
        B, D, H, W = inputs.shape

        flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, D)

        # compute L2 distance between encoder outputs and embedding weights
        # dist [B*H*W, num_embeddings]
        distances = (
                flat_input.pow(2).sum(dim=1, keepdim=True)
                - 2 * flat_input @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(dim=1)
        )
        # find nearest embedding index for each input
        encoding_indices = torch.argmin(distances, dim=1)
        # unflattening
        encoding_indices = encoding_indices.view(B, H, W).unsqueeze(1)
        return encoding_indices

    def forward(self, inputs):
        # inputs [B, D, H, W]
        # flatten input to shape (B*H*W, D)
        B, D, H, W = inputs.shape

        flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, D)
        # compute L2 distance between encoder outputs and embedding weights
        # dist [B*H*W, num_embeddings]
        distances = (
                flat_input.pow(2).sum(dim=1, keepdim=True)
                - 2 * flat_input @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(dim=1)
        )
        # find nearest embedding index for each input
        encoding_indices = torch.argmin(distances, dim=1)
        # quantize: lookup embeddings

        self.usage += torch.bincount(encoding_indices, minlength=self.n_emb)

        if self.refresh and self.counter > self.refresh_timer:
            self.refresh_codebook(flat_input)
        else:
            self.counter += 1

        quantized = self.embedding(encoding_indices)  # (B*H*W, D)

        # reshape back to (B, D, H, W)
        quantized = quantized.view(B, H, W, D).permute(0, 3, 1, 2)

        # passing the inputs passes their gradients
        # the rest to make it the looked up value is passed with detach
        quantized_st = inputs + (quantized - inputs).detach()

        # embedding loss for training embedded vectors
        embed_loss = F.mse_loss(quantized, inputs.detach())
        # commitment loss
        commit_loss = self.commitment_cost * F.mse_loss(inputs, quantized.detach())
        vq_loss = embed_loss + commit_loss

        return quantized_st, vq_loss

    @torch.no_grad()
    def refresh_codebook(self, flat_input):
        device = self.embedding.weight.device
        # used_counts = torch.bincount(encoding_indices, minlength=self.n_emb)

        # Identify unused or low-usage codebook indices
        underused = (self.usage <= self.usage_threshold).nonzero(as_tuple=False).squeeze()

        if underused.numel() == 0:
            return  # all entries are fine

        # Select random input vectors to replace them with
        rand_input_indices = torch.randint(0, flat_input.shape[0], (underused.shape[0],), device=device)
        replacement_vectors = flat_input[rand_input_indices]

        # Replace the dead entries
        self.embedding.weight.data[underused] = replacement_vectors
        self.usage *= 0
        self.counter = 0


class VQ_VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 input_shape: tuple[int, int, int],
                 num_embeddings: int = 512,
                 commitment_cost: float = 0.25, imageloss: nn.Module = DiffImageLoss(),
                 codebook_refresh_period: int = -1, codebook_usage_threshold=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.img_loss = imageloss
        dummy = torch.zeros((1, *input_shape))

        with torch.no_grad():
            out = self.encoder(dummy)

        _, D, Hq, Wq = out.shape

        # build a lookuptable that maps D-dim vectors
        # at each of the Hq×Wq positions into a codebook of size num_embeddings
        self.vq = VQLookUpTable(
            num_embeddings=num_embeddings,
            embedding_dim=D,
            commitment_cost=commitment_cost,
            refresh_every=codebook_refresh_period,
            usage_threshold=codebook_usage_threshold
        )

    def get_indices(self, x):
        z_e = self.encoder(x)
        ind = self.vq.embedding_indices(z_e)
        return ind

    def forward(self, x):
        z_e = self.encoder(x)
        # quantize and compute VQ loss
        z_q, vq_loss = self.vq(z_e)
        # reconstruct
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss

    def get_loss(self, x, x_recon, vq_loss):
        return self.img_loss(x, x_recon) + vq_loss


def train_vq_vae(model, dataloader, optimizer, epochs, device='cuda' if torch.cuda.is_available() else 'cpu',
                 save_path='vq_vae_checkpoint'):
    model = model.to(device)

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
            for images in progress_bar:
                images = images.to(device)

                optimizer.zero_grad()

                recon_x, vq_loss = model(images)

                loss = model.get_loss(images, recon_x, vq_loss)
                loss.backward()

                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
            avg_loss = total_loss / len(dataloader.dataset)
            print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


def train_vae(model, dataloader, optimizer, epochs, device='cuda' if torch.cuda.is_available() else 'cpu',
              save_path='small_cell_VAE.pt'):
    model = model.to(device)

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

            for images in progress_bar:
                images = images.to(device)

                optimizer.zero_grad()
                recon_x, mu, logvar = model(images)
                loss = model.get_loss(recon_x, images, logvar, mu)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(dataloader.dataset)
            print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")







