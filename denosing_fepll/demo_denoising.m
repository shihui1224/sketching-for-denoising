% % Denoising demo script as explained in:
%
% ________________________________________

clear all
close all

addpathrec('.')
deterministic('on');

% Noise level
sig = 20;

% Load and generate images
x = double(imread('cameraman.tif'))/255;
x = x(:,:,1);
[M, N] = size(x);
sig    = sig / 255;
y      = x + sig * randn(M, N);

% Load prior computed offline
prior_model{1} = get_prior_model('EM');
prior_model{2} = get_prior_model('Sketching');


figure;
subplot(2, 2, 1)
imshow(x);
title('Ref Image');

subplot(2, 2, 2)
imshow(y);
title(sprintf('Noisy image PSNR %.2f SSIM %.3f', psnr(y, x), ssim(y, x)));

for k = 1:length(prior_model)
    xhat{k} = fepll(y, sig, prior_model{k});
    subplot(2, 2, 2+k)
    imshow(xhat{k});
    title(sprintf('%s (PSNR %.2f, SSIM %.3f)', ...
                    upper(prior_model{k}.name), ...
                    psnr(xhat{k}, x), ssim(xhat{k}, x))); 
end
    

