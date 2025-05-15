

sz = [256,256];

%% oval and rounded rectangle
I = imread("inputs/oval_and_rectangle.png");
size(I)
I = rgb2gray(I);
imshow(I);
imwrite(I,"inputs/oval_and_rectangle.png");


%% a