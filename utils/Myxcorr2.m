function [ c ] = Myxcorr2( a,b )
%Derived from https://www.mathworks.com/matlabcentral/fileexchange/53570-xcorr2_fft-a-b
%FFT based xcorr2 implementation

if nargin==1
    b=a;
end

% Matrix dimensions
adim = size(a);
bdim = size(b);
% Cross-correlation dimension
cdim = adim+bdim-1;

if isequal(class(a),'gpuArray');
    bpad = gpuArray(zeros(cdim));
    apad = gpuArray(zeros(cdim));
else
    bpad = zeros(cdim);
    apad = zeros(cdim);
end
apad(1:adim(1),1:adim(2)) = a;
bpad(1:bdim(1),1:bdim(2)) = b(end:-1:1,end:-1:1);
ffta = fft2(apad);
fftb = fft2(bpad);
c = real(ifft2(ffta.*fftb));

end

