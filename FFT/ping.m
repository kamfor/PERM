clear all; 
load('output.txt','Y'); 
Y = output;
plot(Y); 

% 1 porbka = 10hz  

pingfft = fft(Y); 
spingfft = fftshift(pingfft); 

spingfft(10000:10000)

plot(abs(spingfft)); 