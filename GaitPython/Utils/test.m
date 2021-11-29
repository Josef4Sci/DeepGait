close all;
x=zeros(2,512);
L=length(x);
y= fft(x);
P2 = abs(y/L);
P1 = P2(2:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = 100*(0:(L/2))/L;
plot(f,P1) 
figure
plot(angle(y))