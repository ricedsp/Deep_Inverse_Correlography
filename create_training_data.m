%% Generate training data
% Chris Metzler

addpath(genpath('.'))

%% Noise parameters
b=80;
gamma_prime=.015;



%% Load all images into memory as a 64x64xn array
im_res=64;
patch_size=im_res/2;
n_train_images=450;
n_val_images=50;
n_patch_per_image=1296;%Must be the square of an integer
TrainSize=n_train_images*n_patch_per_image;
ValSize=n_val_images*n_patch_per_image;
n_total=TrainSize+ValSize;
filenames= dir(fullfile('./datasets/BSD500/', '*.jpg'));
ims=zeros(im_res/2,im_res/2,TrainSize+ValSize);
ind=0;
for i=1:n_total/n_patch_per_image
    A_i_full=double(rgb2gray(imread(['./datasets/BSD500/',filenames(i).name])))/255;
    [h,w]=size(A_i_full);
    i_sep=floor((h-patch_size)/sqrt(n_patch_per_image));
    j_sep=floor((w-patch_size)/sqrt(n_patch_per_image));
    for ii=1:sqrt(n_patch_per_image)
        for jj=1:sqrt(n_patch_per_image)
            im_i=A_i_full((ii-1)*i_sep+1:(ii-1)*i_sep+patch_size,(jj-1)*j_sep+1:(jj-1)*j_sep+patch_size);

            %Thresholds determine how sparse images are
            thresh=[.1,.6];
            im_i=double(edge(im_i,'canny',thresh));
            
            im_i=imrotate(im_i,360*rand(),'crop','bilinear');

            im_i(im_i>0)=1;
            im_i(im_i<0)=0;
            im_i=im_i/(max(im_i(:))+eps);
            if sum(im_i(:))>100 && sum(im_i(:))<250
                ind=ind+1;
                ims(:,:,ind)=imresize(im_i,[im_res/2, im_res/2]);
            end
        end
    end
end
ims=ims(:,:,1:ind);
n_total=ind;
TrainSize=round(.9*n_total);
ValSize=n_total-TrainSize;

%% Check if directories exist yet
directory_name=['Edges'];
directory_name=[directory_name,'_b',num2str(b)];
directory_name=[directory_name,'_gammap',num2str(gamma_prime)];
directory_name=[directory_name,'_res',num2str(im_res)];

if 7~=exist(['./datasets/',directory_name],'dir')%Check if the directory exists yet and if not create it
    mkdir(['./datasets/',directory_name]);
    mkdir(['./datasets/',directory_name,'/train']);
    mkdir(['./datasets/',directory_name,'/val']);
end

rand_inds=randperm(TrainSize+ValSize);

%% Form the training data
f = waitbar(0,'Processing Training Data ...');

rand_inds_train=rand_inds(1:TrainSize);
for i=1:TrainSize
    if mod(i,100)==0
        waitbar(i/(TrainSize),f,'Processing Training Data ...');
    end
    
    ii=rand_inds_train(i);
    
    im_i=ims(:,:,ii);
   
    im_i=padarray(im_i,[floor((im_res-size(im_i,1))/2),floor((im_res-size(im_i,2))/2)],0,'both');
   
    %Form noisy measurement
    corr_i=Myxcorr2(im_i);%FFT-base xcorr2 is much faster
    corr_i(corr_i<0)=0;
    
    %Set central lag to 0 s.t. they are ignored.
    corr_i(im_res,im_res)=0;

    corr_i = corr_i / max(corr_i(:))*255;
    corr_i = corr_i + b;
    corr_i = corr_i + randn(size(corr_i)).*sqrt(gamma_prime).*corr_i;


    im_i= padarray(im_i,[im_res/2-1 im_res/2-1],0,'both');
    im_i= padarray(im_i,[1 1],0,'post');
    
    im_i=im_i/max(im_i(:));
    corr_i=corr_i/max(corr_i(:));

    
    AB=[corr_i,im_i];

    filename=['./datasets/',directory_name,'/train/',num2str(i),'_AB.png'];
    imwrite(AB,filename,'png');
end
close(f)

%% Form the validation data
f = waitbar(0,'Processing Validation Data ...');
rand_inds_val=rand_inds(TrainSize+(1:ValSize));
for i=1:ValSize
    if mod(i,100)==0
        waitbar(i/(ValSize),f,'Processing Validation Data ...');
    end
    
    ii=rand_inds_val(i);
    
    im_i=ims(:,:,ii);
     
    im_i=padarray(im_i,[floor((im_res-size(im_i,1))/2),floor((im_res-size(im_i,2))/2)],0,'both');
   
    %Form noisy measurements
    corr_i=Myxcorr2(im_i);%FFT-base xcorr2 is much faster
    corr_i(corr_i<0)=0;
    
    %Set central lag to 0 s.t. they are ignored.
    corr_i(im_res,im_res)=0;
    
    corr_i = corr_i / max(corr_i(:))*255;
    corr_i = corr_i + b;
    corr_i = corr_i + randn(size(corr_i)).*sqrt(gamma_prime).*corr_i;
    
    im_i= padarray(im_i,[im_res/2-1 im_res/2-1],0,'both');
    im_i= padarray(im_i,[1 1],0,'post');
    
    im_i=im_i/max(im_i(:));
    corr_i=corr_i/max(corr_i(:));

    AB=[corr_i,im_i];

    filename=['./datasets/',directory_name,'/val/',num2str(i),'_AB.png'];
    imwrite(AB,filename,'png');
end

close(f)

