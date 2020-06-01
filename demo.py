import os
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from matplotlib.pyplot import imsave
from PIL import Image
import torchvision.transforms as transforms
from matplotlib.pyplot import imsave
import torch.nn as nn
import torch
import utils.xcorr2 as xcorr2
import time
import utils.network as network

os.makedirs("Reconstructions", exist_ok=True)

cuda = True if torch.cuda.is_available() else False

unet = torch.load('checkpoints/TrainedNetwork.pth', map_location='cuda')#Use this if i used torch.save(model). Reloads both the network and its weights. Uses neglibly more storage


# Configure data loader
dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(
        root="./datasets/SubsampledData/",
        # root="./datasets/ShortExposureData",
        # root="./datasets/SimData",
        transform=transforms.Compose([transforms.Resize((128,256),interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    )
    ,
    batch_size=1,
    shuffle=False,
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

h=128
w=128
for i, (imgs, _) in enumerate(dataloader):

    corr_imgs=Variable(imgs[:,0:1,:,0:128].type(Tensor))
    recon_imgs = unet(corr_imgs)

    plt.subplot(1,2,1)
    plt.imshow(corr_imgs[0,0,:,:].cpu().data.numpy(),vmin=-1,vmax=1)
    plt.subplot(1,2,2)
    plt.imshow(recon_imgs[0,0,(h // 4):((h * 3) // 4), (w // 4):((w * 3) // 4)].cpu().data.numpy(),vmin=-1,vmax=1)
    plt.show()

    imsave("Reconstructions/%d_recon.png" % i, recon_imgs.cpu().data.numpy()[0, 0, (h // 4):((h * 3) // 4), (w // 4):((w * 3) // 4)],vmin=-1,vmax=1)