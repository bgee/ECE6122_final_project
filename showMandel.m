function [ ] = showMandel(  )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
r = csvread('r.csv'); r = r(1:512, 1:512);
g = csvread('g.csv'); g = g(1:512, 1:512);
b = csvread('b.csv'); b = b(1:512, 1:512);
rgb = zeros(512, 512, 3);
rgb(:, :, 1) = r;
rgb(:, :, 2) = g;
rgb(:, :, 3) = b;
figure;
imshow(rgb);

end

