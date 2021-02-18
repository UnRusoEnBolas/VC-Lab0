%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% VC i PSIV                                                      %%%
%%% Lab 0 (basat en les pr�ctiques de Gemma Rotger)                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 
% Hello! Welcome to the computer vision LAB. This is a section, and 
% you can execute it using the run section button on the top panel. If 
% you prefer, you can run all the code using the run button. Run this 
% section when you need to clear your data, figures and console 
% messages.
clearvars,
close all,
clc,

% With addpath you are adding the image path to your main path
% addpath('img')


%% PROBLEM 1 (+0.5) --------------------------------------------------
% DONE. READ THE CAMERAMAN IMAGE.

cameraman_img = imread('img/cameraman.jpg');

%% PROBLEM 2 (+0.5) --------------------------------------------------
% DONE: SHOW THE CAMERAMAN IMAGE.

imshow(cameraman_img);

%% PROBELM 3 (+2.0) --------------------------------------------------
% DONE. Negative effect using a double for instruction

tic,
source_sizes = size(cameraman_img);
neg_cameraman_img = cameraman_img;
for row = 1:source_sizes(1)
    for col = 1:source_sizes(2)
        neg_cameraman_img(row, col) = 255 - neg_cameraman_img(row, col);
    end
end
toc
figure('Name', 'Naive (double for) calculation');
imshow(neg_cameraman_img);

% DONE. Negative efect using a vectorial instruction
tic,
neg_cameraman_img = 255 - cameraman_img;
toc,
figure('Name', 'Vectorial calculation');
imshow(neg_cameraman_img);

% You should see that results in figures 1 and 2 are the same but times
% are much different.

%% PROBLEM 4 (+2.0) --------------------------------------------------

% DONE. Give some color (red, green or blue)
r = cameraman_img;
g = neg_cameraman_img;
b = cameraman_img;

colored_cameraman_img = zeros(source_sizes(1), source_sizes(2), 3, 'uint8');
% Importante ese 'uint8'....
colored_cameraman_img(:,:,1) = r;
colored_cameraman_img(:,:,2) = g;
colored_cameraman_img(:,:,3) = b;
figure('Name', 'Colored image (normal, neg, normal)');
imshow(colored_cameraman_img);

colored_cameraman_img2 = cat(3, r, g, b);
figure('Name', 'Colored image (normal, neg, normal)- Using cat() function');
imshow(colored_cameraman_img2);

%% PROBLEM 5 (+1.0) --------------------------------------------------

imwrite(colored_cameraman_img, "./img/generated_colored_cameraman_img.png");
imwrite(colored_cameraman_img, "./img/generated_colored_cameraman_img.tif");
imwrite(colored_cameraman_img, "./img/generated_colored_cameraman_img.jpg");
imwrite(colored_cameraman_img, "./img/generated_colored_cameraman_img.bmp");

sizes = size(colored_cameraman_img);
raw_size_kbytes = (sizes(1)*sizes(2)*sizes(3))/1024;
% Es una matriz de uint8 (ocupa un byte cada elemento de la matriz), por lo
% tanto multiplicamos las dimensiones para obtener el total de elementos y
% finalmente dividimos entre cu�ntos bytes es un kilobyte.
disp("'RAW' size: " + raw_size_kbytes + " kb");

%% PROBLEM 6 (+1.0) --------------------------------------------------

lin128 = cameraman_img(128,:);
figure('Name', 'Row 128 plot - original image');
lin128_mean = mean(lin128);
plot(lin128, 'k');
hold on;
plot(lin128_mean*ones(1, length(lin128)), '--m');
hold off;
grid on;

lin128rgb = colored_cameraman_img(128,:,:);
figure('Name', 'Row 128 plot - colored image');
lin128rbg_mean = mean(lin128rgb,3);
hold on;
plot(lin128rgb(:,:,1), 'r');
plot(lin128rgb(:,:,2), 'g');
plot(lin128rgb(:,:,3), 'b');
plot(lin128rbg_mean, '--m');
hold off;
grid on;

%% PROBLEM 7 (+2) ----------------------------------------------------

pict0004_img = imread('img/pict0004.png');
t22_img = imread('img/t22.jpg');

tic;
figure('Name', 'cameraman.jpg histogram');
imhist(cameraman_img);
figure('Name', 'pict0004.png histogram');
imhist(pict0004_img);
figure('Name', 't22.jpg histogram');
imhist(t22_img);
toc;

% TODO. Compute the histogram.
tic;
flat_cameraman_img = reshape(cameraman_img,1,[]);
histogram_vector = zeros(1, 256);
for pixel = 1:length(flat_cameraman_img)
    histogram_vector(flat_cameraman_img(pixel)+1) = histogram_vector(flat_cameraman_img(pixel)+1)+1;
end
figure('Name', 'Self-calculated histogram of cameraman.jpg');
bar(histogram_vector);

flat_pict0004_img = reshape(pict0004_img,1,[]);
histogram_vector = zeros(1, 256);
for pixel = 1:length(flat_pict0004_img)
    histogram_vector(flat_pict0004_img(pixel)+1) = histogram_vector(flat_pict0004_img(pixel)+1)+1;
end
figure('Name', 'Self-calculated histogram of pict0004.png');
bar(histogram_vector);

flat_t22_img = reshape(t22_img,1,[]);
histogram_vector = zeros(1, 256);
for pixel = 1:length(flat_t22_img)
    histogram_vector(flat_t22_img(pixel)+1) = histogram_vector(flat_t22_img(pixel)+1)+1;
end
figure('Name', 'Self-calculated histogram of t22.jpg');
bar(histogram_vector);
toc;

% Parece que voy un poqito m�s r�pido jejeje

%% PROBLEM 8 Binarize the image text.png (+1) ------------------------

% DONE. Read the image
alice_img = rgb2gray(imread('img/alice.jpg'));
figure('Name', 'Alice.jpg image');
imshow(alice_img);
figure('Name', 'Alice.jpg histogram');
imhist(alice_img);

% DONE. Define 3 different thresholds
th1 = 85;
th2 = 195;
th3 = 244;

% TODO. Apply the 3 thresholds 5 to the image
threshimtext1 = alice_img > th1;
figure('Name', 'Alice image as binary - threshold 1');
imshow(threshimtext1)
threshimtext2 = alice_img > th2;
figure('Name', 'Alice image as binary - threshold 2');
imshow(threshimtext2)
threshimtext3 = alice_img > th3;
figure('Name', 'Alice image as binary - threshold 3');
imshow(threshimtext3)

% DONE. Show the original image and the segmentations in a subplot
figure('Name', 'Alice original image and the binarized images')
subplot(2,1,1);
imshow(alice_img);
title('Original image');
subplot(2,3,4);
imshow(threshimtext1);
subplot(2,3,5);
imshow(threshimtext2);
subplot(2,3,6);
imshow(threshimtext3);

%% THE END -----------------------------------------------------------
% Well done, you finished this lab! Now, remember to deliver it 
% properly on Caronte.

% File name:
% lab0_NIU.zip 
% (put matlab file lab0.m and python file lab0.py in the same zip file)
% Example lab0_1234567.zip

















