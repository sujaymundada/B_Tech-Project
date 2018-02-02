pkg load communications ; 

dataPoints = 2000 ; 

QAM16 = randi([0 15],dataPoints,1) ; % 16 QAM 
QAM4 = randi([0 3],dataPoints,1) ; % 4 QAM 
QAM32 = randi([0 31],dataPoints,1) ; %32 QAM

s = [1+1i 1+3i 3+3i 3+i 3-3i 1-i 1-3i 3-i -1+i -1+3i -3-3i -3-i -1-i -1-3i -3+i -3+3i]/sqrt(10) ;
f=[1+i 1-i -1-i -1+i]/sqrt(2) ; 
t=[1+i 3+3i 1+3i 1+5i 3+i 3+5i 5+i 5+3i -1-i -3-3i -1-3i -1-5i -3-i -3-5i -5-i -5-3i -1+i -3+3i -1+3i -1+5i -3+i -3+5i -5+i -5+3i 1-i 3-3i 1-3i 1-5i 3-i 3-5i 5-i 5-3i]/sqrt(20) ;

mod16=genqammod(QAM16,s) ;
mod4 = genqammod(QAM4,f) ;
mod32 = genqammod(QAM32,t) ;
value = mod4(1,1)*10 ; 
plot(value,"*");
hold on ; 
myData = ones(dataPoints,6*10);

for j = 1:10
    noise = (sqrt(1/2)*[(randn(dataPoints,1)+1i*randn(dataPoints,1))])';
    rx16 = sqrt(100)*mod16 + noise ; 
    %figure(1); 
    %plot(rx16);
    x16 = real(rx16) ;
    y16 = imag(rx16) ;

    noise = (sqrt(1/2)*[(randn(dataPoints,1)+1i*randn(dataPoints,1))])';
    rx4 = sqrt(100)*mod4 + noise ;  
    %figure(2);
    %plot(rx4);
    x4 = real (rx4) ;
    y4 = imag (rx4) ; 

    noise = (sqrt(1/2)*[(randn(dataPoints,1)+1i*randn(dataPoints,1))])';
    rx32 = sqrt(100)*mod32 + noise ;  
    %figure(3);
    %plot(rx32);
    x32 = real (rx32) ;
    y32 = imag (rx32) ; 

    myData(:,[(j-1)*6+1]) = x16';
    myData(:,[(j-1)*6+2]) = y16';
    myData(:,[(j-1)*6+3]) = x4';
    myData(:,[(j-1)*6+4]) = y4';
    myData(:,[(j-1)*6+5]) = x32';
    myData(:,[(j-1)*6+6]) = x32';
    plot(rx4(1,1));
    hold on ; 
    %plot(10*mod16(1,2));
    %hold on; 
endfor
%csvwrite('myData_SNR10.csv',myData);
