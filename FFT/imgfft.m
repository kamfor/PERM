clear all; 
data  =imread('img.png');
imggrey = rgb2gray(data); 

r = 50; 

fftimg = fft2(imggrey); 

imgshift = fftshift(fftimg); 

%mesh(abs(imgshift));
%mesh(angle(imgshift));

for x=1:512
    for y=1:512
        xx = abs(x-256); 
        yy = abs(y-256); 
        if xx*xx+yy*yy<r*r
         imgshift(x,y) = 0.0;
        end
    end
end

mesh(abs(imgshift));



fftimg = fftshift(imgshift); 

new_image = uint8(ifft2(fftimg)); 

imshow(new_image); 
