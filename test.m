close all;
I = rgb2gray(imread('./inputs/drlse_knee.jpg'));
I = double(I)/255;
I = imnoise(I,"gaussian",0,0.01);
imshow(I);
h = iso_frac_filter(0.1,31)
I_frac = rescale(abs(imfilter(I,h)));
% I_frac = imfilter(I,h);
figure();
imshow(I_frac);