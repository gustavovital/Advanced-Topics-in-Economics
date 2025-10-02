using SparseArrays
## parameters 

γ = 2; # risk aversion parameter (γ) in the CRRA utility function.
σ_2 = (0.8)^2; # the variance of the innovation in the Ornstein-Uhlenbeck (O-U) process for log income.
Corr = exp(-0.9); # the persistence of the O-U process (with the parameter θ = -log(Corr)).
ρ = 0.05; # the discount rate (ρ).
r = 0.035; # the interest rate.
w = 1; # the wage rate.


zmean = exp(σ_2/2);

J=15; # number of z points 
zmin = 0.75; # Range z
zmax = 2.5;
amin = 0; # borrowing constraint
amax = 100; # range a
I=300;  # number of a points 

T=75; # maximum age
N=300; # number of age steps
# N=75; # number of age steps
# N=10; # number of age steps
∂t=T/N;

# Simulation parameters
#maxit  = 100;     # maximum number of iterations in the HJB loop
crit = 10^(-10); # criterion HJB loop

# ORNSTEIN-UHLENBECK IN LOGS
the = -log(Corr);

# VARIABLES
a = LinRange(amin,amax,I);  #wealth vector
∂a = (amax-amin)/(I-1);      
z = LinRange(zmin,zmax,J)';   # productivity vector
∂z = (zmax-zmin)/(J-1);
∂z2 = dz^2;

aa = a*ones(1,J);
zz = ones(I,1)*z;

μ = -the.*z.*log.(z)+ σ_2/2*z; 
s2 = σ_2.*z.^2; 

Vaf = zeros(I,J);             
Vab = zeros(I,J);
Vzf = zeros(I,J);
Vzb = zeros(I,J);
Vzz = zeros(I,J);
c = zeros(I,J);

# construction of Aswitch matrix

yy = - s2/dz2 - μ/∂z;
χ =  s2/(2*∂z2);
ζ = μ/∂z + s2/(2*∂z2);

#This will be the upperdiagonal of the matrix Aswitch
updiag=zeros(I,1); #This is necessary because of the peculiar way spdiags is defined.
for j=1:J
    updiag= [updiag; repeat([ζ[j]], I, 1)];
end
#This will be the center diagonal of the matrix Aswitch
centdiag=repeat([χ[1]+yy[1]],I);
for j=2:J-1
    centdiag=[centdiag;repeat([yy[j]],I)];
end

centdiag=[centdiag;repeat([yy[J]+ζ[J]],I)];

#This will be the lower diagonal of the matrix Aswitch
lowdiag=repeat([chi[2]],I);
for j=3:J
    lowdiag=[lowdiag;repeat([χ[j]],I)];
end

#Add up the upper, center, and lower diagonal into a sparse matrix
Aswitch=spdiags(centdiag,0,I*J,I*J)+spdiags(lowdiag,-I,I*J,I*J)+spdiags(updiag,I,I*J,I*J);