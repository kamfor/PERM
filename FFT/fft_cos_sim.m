clear all; 

f_cos = @(A,w,t,f,c) A*cos(2*pi*w*t+f)+c;

symulacja = zeros(3,3000);

f1 = 0.02;  
% jedna probka to czestotliwosc proobkowania / ilosc probek czyli 1000 / 3000
for time=0:3000
    symulacja(1,time+1) = f_cos(1,15.7,time/1000,0.4,0.5);
end

for time=0:3000
    symulacja(2,time+1) = f_cos(0.7,20,time/1000,0.1,0.1);
end

for time=0:3000
    symulacja(2,time+1) = f_cos(0.4,10,time/1000,0.8,-0.6);
end

suma = zeros(1,3000); 

suma(1:3000) = symulacja(1,1:3000)+symulacja(2,1:3000)+symulacja(3,1:3000);

plot(suma)

my_fft = fft(suma); 
shifted = fftshift(my_fft); 


amp = abs(shifted); 
phase = angle(shifted); 

