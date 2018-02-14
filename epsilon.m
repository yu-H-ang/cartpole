np = 10000-1;
eps0 = 0.5;
epsn = 0.01;
A = eps0';
B = log(eps0/epsn)/np;

x = linspace(0,np,100);
y1 = A*exp(-B*x);
plot(x,y1);hold on;

y2 = eps0*(epsn/eps0).^(x/np);
plot(x,y2,'.');