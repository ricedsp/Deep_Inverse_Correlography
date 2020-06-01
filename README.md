# Deep Inverse Correlography
Code associated with the paper "Deep-inverse correlography: towards real-time high-resolution non-line-of-sight imaging." Optica, 2020.


## Abstract
Low signal-to-noise ratio (SNR) measurements, primarily due to the quartic attenuation of intensity with distance, are
arguably the fundamental barrier to real-time, high-resolution, non-line-of-sight (NLoS) imaging at long standoffs.
To better model, characterize, and exploit these low SNR measurements, we use spectral estimation theory to derive a
noise model for NLoS correlography. We use this model to develop a speckle correlation-based technique for recovering
occluded objects from indirect reflections. Then, using only synthetic data sampled from the proposed noise model,
and without knowledge of the experimental scenes nor their geometry, we train a deep convolutional neural network
to solve the noisy phase retrieval problem associated with correlography. We validate that the resulting deep-inverse
correlography approach is exceptionally robust to noise, far exceeding the capabilities of existing NLoS systems both in
terms of spatial resolution achieved and in terms of total capture time. We use the proposed technique to demonstrate
NLoS imaging with 300 µm resolution at a 1 m standoff, using just two 1/8th s exposure-length images from a standard
complementary metal oxide semiconductor detector.

## Dependencies
The code to train and test the network uses Pytorch and assumes access to an Nvidia GPU. All dependencies can be downloaded by running "conda install pytorch torchvision matplotlib"

The "create_training_data.m" script requires a recent version of matlab.

## How to Run
To test a pretrained network, download network weights from https://stanford.box.com/s/zyzdhzyk68xw6z8whpcup5cman55esl7, place them in the "checkpoints" directory, and run "demo.py". Select which dataset to reconstruct by changing the root directory. 

To train your own network:
1. Download the BSD-500 dataset and place it in datasets/BSD500 https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html.
1. Run "create_training_data.m" to create a training dataset.
1. Run "train_network.py" to train a network.

Please direct questions to cmetzler@stanford.edu.
## Acknowledgements
We used the BSD-500 dataset for training [A]. Our U-net implementation is based of off that provided in Cycle-GAN [B].

[A] Martin, David, et al. "A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics." Proceedings Eighth IEEE International Conference on Computer Vision. ICCV 2001. Vol. 2. IEEE, 2001.

[B] Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." Proceedings of the IEEE international conference on computer vision. 2017. 
https://junyanz.github.io/CycleGAN/
