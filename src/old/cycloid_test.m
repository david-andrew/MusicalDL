clc
close all
t = linspace(0,2*pi, 1000);
xp = 2 * (t - sin(t));
yp = 2 * (1 - cos(t));
plot(xp,yp);
hold on
plot(linspace(0,2*pi*10,500), interp1(xp, yp, mod(linspace(0,2*pi*10,500),4*pi)), 'o');

