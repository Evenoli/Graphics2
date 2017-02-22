
%Indeces of refraction
N1 = 1.0;
N2 = 1.4;

%From air (1.0) into material (1.4)
theta1 = 1:90;
Rp = [];
Rt = [];
schlicks = [];
avg = [];
brews = 0;

for theta=1:90
    i = (theta / 180) * pi;
    t = sqrt(1 - ((N1/N2)*sin(i))^2);
    
    Rp(theta) = ( (N1 * cos(i) - N2 *t) / ...
        (N1 * cos(i) + N2 *t)) ^ 2;
    Rt(theta) = ((N1 * t - N2 *cos(i)) / ...
        (N1 * t + N2 *cos(i))) ^ 2;
    if(Rt(theta) < 0.00002)
        disp(['Brewsters angle: ', num2str(theta)])
        brews = theta;
    end
    avg(theta) = (Rp(theta) + Rt(theta)) /2;
end

NormInc = Rp(1);
for theta=1:90
    i = (theta / 180) * pi;
    
    schlicks(theta) = NormInc + (1 - NormInc) * (1 - cos(i))^5;
end

length(schlicks)

figure;
vax = axes;
plot(theta1, Rt, theta1, Rp, theta1, schlicks, theta1, avg);
title ('N1 = 1.0, N2 = 1.4');
v=line([brews, brews], get(vax, 'YLim'), 'Color', 'g');
set(v, 'LineStyle', '--');
legend('Rt','Rp', 'shlicks', 'average', 'Brewster Angle');
xlabel('Angle of Incidence (Degrees)') % x-axis label
ylabel('Reflectane value') % y-axis label
axis([0,90,0,1]);

%Exiting material
N1 = 1.4;
N2 = 1.0;

theta1 = 0:0.03125:90;
Rp = [];
Rt = [];
crit = 100000;

for theta=1:2881
    i = (theta1(theta) / 180) * pi;
    t = sqrt(1 - ((N1/N2)*sin(i))^2);
    
    Rp(theta) = ( (N1 * cos(i) - N2 *t) / ...
        (N1 * cos(i) + N2 *t)) ^ 2;
    Rt(theta) = ((N1 * t - N2 *cos(i)) / ...
        (N1 * t + N2 *cos(i))) ^ 2;
    if(abs(1 - Rp(theta)) < 0.1 && (theta/32 < 70) )
        disp(['Critical angles: ', num2str(theta/32)]);
        crit = theta/32;
    end
    if (theta/32 > crit)
        Rp(theta) = 1.1;
        Rt(theta) = 1.1;
    end
end


figure;
vax = axes;
plot(theta1, Rt, theta1, Rp);
title('N1 = 1.4, N2 = 1.0');
v =line([crit, crit], get(vax, 'YLim'), 'Color', 'g');
set(v, 'LineStyle','--'); 
legend('Rt','Rp', 'Critical angle');
xlabel('Angle of Incidence (Degrees)') % x-axis label
ylabel('Reflectane value') % y-axis label
axis([0,90,0,1]);