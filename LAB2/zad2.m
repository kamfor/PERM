clear all;

n = 20000; 

data=dlmread('signal.txt')';
data=data(1:n);

data = fft(data);
data_shifted=fftshift(data);

r=3500;
data_shifted(10000-r:10000+r)=0; %wyciêcie czêstotliwoœci
signal=abs(ifft(fftshift(data_shifted))); %rekonstrukcja sygna³u
s_time=0:0.005:100; %podstawa czasu 40kHz
s_time=s_time(1:n);

figure;
hold on;
subplot(3,1,1);
plot(s_time,signal);
title('Sygna³');
subplot(3,1,2);
plot(abs(data_shifted));
title('Amplituda');
subplot(3,1,3);
plot(angle(data_shifted));
title('Faza');
hold off;
t_ping = 50.5; 
t_echo = 60.5; 
time=(t_echo-t_ping); 
velocity=300; %m/s
distance=velocity*time/2 %tam i spowrotem w mm