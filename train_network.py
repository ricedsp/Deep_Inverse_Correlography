import argparse
import os
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
from matplotlib.pyplot import imsave
import torch
import utils.xcorr2 as xcorr2
import utils.network as network

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--n_epochs", type=int, default=400, help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")#Autocorrelation is 2*img_size-1
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

show_results = False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

unet = network.Unet()


if cuda:
    unet.cuda()

# Initialize weights
unet.apply(weights_init_normal)

# Configure data loader
dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(
        root="./datasets/Edges_b80_gammap0.015_res64/",
        transform=transforms.Compose([transforms.Resize((128,256),interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    )
    ,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Loss function
def L1corrloss(recon,GT):
    [h, w] = GT.shape[2:4]
    GT_center = GT[:, :, (h // 4):((h * 3) // 4), (w // 4):((w * 3) // 4)] + 1.
    recon_center = recon[:, :, (h // 4):((h * 3) // 4), (w // 4):((w * 3) // 4)] + 1.
    corr_loss = (xcorr2.xcorr2_torch(recon_center) - xcorr2.xcorr2_torch(  GT_center)).abs().sum()
    return corr_loss

# Optimizer
optimizer = torch.optim.Adam(unet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

decayRate = 0.987
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

load_existing=False
if load_existing:
    epoch_start=1
    unet = torch.load('EdgesNetwork_0.pth', map_location='cuda')
else:
    epoch_start=0

# ----------
#  Training
# ----------
for epoch in range(epoch_start,opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        corr_imgs=imgs[:,0:1,:,0:128]#Use only 1 color channel. 0:1 notation keeps dimensions
        true_imgs=imgs[:,0:1,:,128:]#Use only 1 color channel. 0:1 notation keeps dimensions

        # Configure input
        corr_imgs = Variable(corr_imgs.type(Tensor))
        true_imgs = Variable(true_imgs.type(Tensor))

        optimizer.zero_grad()

        # Generate a batch of images
        recon_imgs = unet(corr_imgs)

        loss = L1corrloss(recon_imgs,true_imgs)

        loss.backward()
        optimizer.step()

        batches_done = epoch * len(dataloader) + i
        if batches_done % 25 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [Corr loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss.item())
            )
        if batches_done % 1000 == 0:
            imsave("Progress/%d_recon.png" % batches_done, recon_imgs.cpu().data.numpy()[0, 0, :, :],vmin=-1,vmax=1)
            imsave("Progress/%d_GT.png" % batches_done,true_imgs.cpu().data.numpy()[0,0,:,:],vmin=-1,vmax=1)
    my_lr_scheduler.step()
    torch.save(unet,'checkpoints/EdgesNetwork_'+str(epoch)+'.pth')