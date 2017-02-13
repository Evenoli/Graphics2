
%Indeces of refraction
N1 = 1.0;
N2 = 1.4;

%From air (1.0) into material (1.4)
theta1 = 1:90;
Rp = [];
Rt = [];

for theta=1:90
    i = (theta / 180) * pi;
    t = sqrt(1 - ((N1/N2)*sin(i))^2);
    
    Rp(theta) = ( (N1 * cos(i) - N2 *t) / ...
        (N1 * cos(i) + N2 *t)) ^ 2;
    Rt(theta) = ((N1 * t - N2 *cos(i)) / ...
        (N1 * t + N2 *cos(i))) ^ 2;
    if(Rt(theta) < 0.00002)
        disp(['Brewsters angle: ', num2str(theta)])
    end
end

figure;
title ('N1 = 1.0, N2 = 1.4');
plot(theta1, Rt, theta1, Rp);
legend('Rt','Rp');
axis([0,90,0,1]);

%Exiting material
N1 = 1.4;
N2 = 1.0;

theta1 = 0:0.03125:90;
Rp = [];
Rt = [];

for theta=1:2881
    i = (theta1(theta) / 180) * pi;
    t = sqrt(1 - ((N1/N2)*sin(i))^2);
    
    Rp(theta) = ( (N1 * cos(i) - N2 *t) / ...
        (N1 * cos(i) + N2 *t)) ^ 2;
    Rt(theta) = ((N1 * t - N2 *cos(i)) / ...
        (N1 * t + N2 *cos(i))) ^ 2;
    if(abs(1 - Rp(theta)) < 0.1 && (theta/32 < 70) )
        disp(['Critical angles: ', num2str(theta/32)]);
    end
    
end
figure;
title ('N1 = 1.0, N2 = 1.4');
plot(theta1, Rt, theta1, Rp);
legend('Rt','Rp');
axis([0,90,0,1]);