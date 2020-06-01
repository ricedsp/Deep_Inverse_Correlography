#Chris Metzler
#2/13/20

import torch as torch
import numpy as np

def flip(x, dim):
    #From https://discuss.pytorch.org/t/optimizing-diagonal-stripe-code/17777/17
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def xcorr2_torch(a,b=torch.tensor([])):#Torch implementation of correlate2d
    [n_batch,n_c,ha,wa]=a.shape
    mydevice = a.device
    assert n_c == 1, "Only grayscale currently supported"
    if not b.nelement()==0:
        [_,_,hb,wb]=b.shape
        astarb=torch.zeros(n_batch,n_c,ha + hb - 1, wa + wb - 1,dtype=a.dtype,device=mydevice)
        a_full=torch.zeros((n_batch,ha+hb-1,wa+wb-1),dtype=a.dtype,device=mydevice)
        b_full=torch.zeros((n_batch,ha+hb-1,wa+wb-1),dtype=a.dtype,device=mydevice)
        a_full[:,0:ha,0:wa]=a[:,0,:,:]
        b_full[:, 0:hb, 0:wb] = flip(flip(b[:, 0, :, :], 1), 2)
        A=torch.rfft(a_full,signal_ndim=2,onesided=False,normalized=False)
        B=torch.rfft(b_full,signal_ndim=2,onesided=False,normalized=False)
        C=torch.zeros((n_batch,ha + hb - 1, wa + wb - 1,2),dtype=a.dtype,device=mydevice)#Fourth dim used to separate real and imaginary component
        Ar=A[:,:,:,0]
        Ai=A[:,:,:,1]
        Br=B[:,:,:,0]
        Bi=B[:,:,:,1]
        C[:,:,:,0]=Ar*Br-Ai*Bi#Elementwise Product. Need to implement complex multiplication
        C[:,:,:,1]=Ar*Bi+Ai*Br
        astarb[:,0,:,:]=torch.irfft(C,2,onesided=False,normalized=False)
        return astarb
    else: #compute autocorrelation of a
        astara = torch.zeros(n_batch, n_c, 2 * ha - 1, 2 * wa - 1, dtype=a.dtype, device=mydevice)
        a_full = torch.zeros(n_batch, 2 * ha - 1, 2 * wa - 1, dtype=a.dtype, device=mydevice)
        a_full[:, 0:ha, 0:wa] = a[:, 0, :, :]
        A = torch.rfft(a_full, signal_ndim=2, onesided=False, normalized=False)
        Ar = A[:, :, :, 0]
        Ai = A[:, :, :, 1]
        Aabs2 = torch.zeros((n_batch, 2 * ha - 1, 2 * wa - 1, 2), dtype=a.dtype, device=mydevice)  # Fourth dim used to separate real and imaginary component
        Aabs2[:, :, :, 0] = Ar.abs() ** 2 + Ai.abs() ** 2
        astara[:,0,:,:] = torch.irfft(Aabs2, signal_ndim=2, onesided=False, normalized=False)
        #Still need to apply fftshift to astara for it to be consistent with the other definitions
        return astara

#When batch size is one, CPU is faster than GPU
def xcorr2_torch_CPU(a,b=torch.tensor([])):#Torch implementation of correlate2d
    [n_batch,n_c,ha,wa]=a.shape
    mydevice = a.device
    assert n_c == 1, "cpu"
    if not b.nelement()==0:
        [_,_,hb,wb]=b.shape
        astarb=torch.zeros(n_batch,n_c,ha + hb - 1, wa + wb - 1,dtype=a.dtype,device=mydevice)
        a_full=torch.zeros((n_batch,ha+hb-1,wa+wb-1),dtype=a.dtype,device=mydevice)
        b_full=torch.zeros((n_batch,ha+hb-1,wa+wb-1),dtype=a.dtype,device=mydevice)
        a_full[:,0:ha,0:wa]=a[:,0,:,:]
        b_full[:, 0:hb, 0:wb] = flip(flip(b[:, 0, :, :], 1), 2)
        A=torch.rfft(a_full,signal_ndim=2,onesided=False,normalized=False)
        B=torch.rfft(b_full,signal_ndim=2,onesided=False,normalized=False)
        C=torch.zeros((n_batch,ha + hb - 1, wa + wb - 1,2),dtype=a.dtype,device=mydevice)#Fourth dim used to separate real and imaginary component
        Ar=A[:,:,:,0]
        Ai=A[:,:,:,1]
        Br=B[:,:,:,0]
        Bi=B[:,:,:,1]
        C[:,:,:,0]=Ar*Br-Ai*Bi#Elementwise Product. Need to implement complex multiplication
        C[:,:,:,1]=Ar*Bi+Ai*Br
        astarb[:,0,:,:]=torch.irfft(C,2,onesided=False,normalized=False)
        return astarb
    else: #compute autocorrelation of a
        astara = torch.zeros(n_batch, n_c, 2 * ha - 1, 2 * wa - 1, dtype=a.dtype, device=mydevice)
        a_full = torch.zeros(n_batch, 2 * ha - 1, 2 * wa - 1, dtype=a.dtype, device=mydevice)
        a_full[:, 0:ha, 0:wa] = a[:, 0, :, :]
        A = torch.rfft(a_full, signal_ndim=2, onesided=False, normalized=False)
        Ar = A[:, :, :, 0]
        Ai = A[:, :, :, 1]
        Aabs2 = torch.zeros((n_batch, 2 * ha - 1, 2 * wa - 1, 2), dtype=a.dtype, device=mydevice)  # Fourth dim used to separate real and imaginary component
        Aabs2[:, :, :, 0] = Ar.abs() ** 2 + Ai.abs() ** 2
        astara[:,0,:,:] = torch.irfft(Aabs2, signal_ndim=2, onesided=False, normalized=False)
        #Still need to apply fftshift to astara for it to be consistent with the other definitions
        return astara

def FourierMod2_nopad(a):
    [n_batch,n_c,ha,wa]=a.shape
    mydevice=a.device
    assert n_c==1, "Only grayscale currently supported"
    a=a.view(n_batch,ha,wa)
    A=torch.rfft(a,signal_ndim=2,onesided=False,normalized=False)
    Ar = A[:, :, :, 0]
    Ai = A[:, :, :, 1]
    Aabs2=Ar.abs()**2+Ai.abs()**2#Unlike the definition used in xcorr2, Aabs2 is not complex here.
    return Aabs2.reshape([n_batch,n_c,ha,wa]), Ar.reshape([n_batch,n_c,ha,wa]), Ai.reshape([n_batch,n_c,ha,wa])

def FourierMod2_nopad_complex(a):
    [n_batch,n_c,ha,wa,_]=a.shape
    mydevice=a.device
    assert n_c==1, "Only grayscale currently supported"
    a=a.view(n_batch,ha,wa,2)
    A=torch.fft(a,signal_ndim=2,normalized=False)
    Ar = A[:, :, :, 0]
    Ai = A[:, :, :, 1]
    Aabs2=Ar.abs()**2+Ai.abs()**2#Unlike the definition used in xcorr2, Aabs2 is not complex here.
    return Aabs2.reshape([n_batch,n_c,ha,wa]), Ar.reshape([n_batch,n_c,ha,wa]), Ai.reshape([n_batch,n_c,ha,wa])

def FourierMod2(a):
    [n_batch,n_c,ha,wa]=a.shape
    mydevice=a.device
    assert n_c==1, "Only grayscale currently supported"
    a_pad = torch.zeros(n_batch, 2*ha - 1, 2*wa - 1, dtype=a.dtype, device=mydevice)
    a_pad[:, 0:ha, 0:wa]=a[:, 0, :, :]
    A=torch.rfft(a_pad,signal_ndim=2,onesided=False,normalized=False)
    Ar = A[:, :, :, 0]
    Ai = A[:, :, :, 1]
    Aabs2=Ar.abs()**2+Ai.abs()**2#Unlike the definition used in xcorr2, Aabs2 is not complex here.
    return Aabs2.reshape([n_batch,n_c,2*ha-1,2*wa-1]), Ar.reshape([n_batch,n_c,2*ha-1,2*wa-1]), Ai.reshape([n_batch,n_c,2*ha-1,2*wa-1])

def FourierMod2_CPU(a):
    [n_batch,n_c,ha,wa]=a.shape
    mydevice="cpu"
    assert n_c==1, "Only grayscale currently supported"
    a_pad = torch.zeros(n_batch, 2*ha - 1, 2*wa - 1, dtype=a.dtype, device=mydevice)
    a_pad[:, 0:ha, 0:wa]=a[:, 0, :, :]
    A=torch.rfft(a_pad,signal_ndim=2,onesided=False,normalized=False)
    Ar = A[:, :, :, 0]
    Ai = A[:, :, :, 1]
    Aabs2=Ar.abs()**2+Ai.abs()**2#Unlike the definition used in xcorr2, Aabs2 is not complex here.
    return Aabs2.reshape([n_batch,n_c,2*ha-1,2*wa-1])


def test():
    #a=np.random.randn(128,1,64,64)
    a=np.zeros((2,1,5,5))
    a[0,0,:,:]=np.array([[1.,2.,3.,4.,5.],[1.,2.,3.,4.,5.],[1.,2.,3.,4.,5.],[1.,2.,3.,4.,5.],[1.,2.,3.,4.,5.]])

    a_torch = torch.tensor(a, dtype=torch.float32,device='cuda')

    astara_1 = xcorr2_torch(a_torch,a_torch).to(device='cuda')
    astara_2=xcorr2_torch_CPU(a_torch,a_torch).to(device='cuda')
    astara_3 = xcorr2_torch(a_torch).to(device='cuda')

    #Apply an fftshift to astara_3 to make it consistent with the other definitions
    astara_3=np.array(astara_3.cpu())
    astara_3=np.fft.fftshift(astara_3,(2,3))
    astara_3=torch.tensor(astara_3, dtype=torch.float32,device='cuda')

    [n_batch, n_c, ha, wa]=a.shape
    absFa2=torch.zeros((n_batch, n_c, 2*ha-1, 2*wa-1,2))
    absFa2[:,:,:,:,0]=FourierMod2(a_torch)
    astara_4= torch.irfft(absFa2, signal_ndim=2, onesided=False, normalized=False)#0-lag is at 0

    # Apply an fftshift to astara_3 to make it consistent with the other definitions
    astara_4=np.array(astara_4)
    astara_4=np.fft.fftshift(astara_4,(2,3))
    astara_4=torch.tensor(astara_4, dtype=torch.float32,device='cuda')

    diff_12=torch.norm(astara_1-astara_2,2)
    diff_13 = torch.norm(astara_1 - astara_3, 2)
    diff_14= torch.norm(astara_1 - astara_4, 2)

    print(diff_12)
    print(diff_13)
    print(diff_14)


if __name__ == "__main__":
    test()
