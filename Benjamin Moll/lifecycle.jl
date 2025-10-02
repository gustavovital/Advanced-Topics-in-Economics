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
∂z2 = ∂z^2;

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

∂Vf = zeros(I, J)
∂Vb = zeros(I, J)

# construction of Aswitch matrix

yy = - s2/ ∂z2 - μ/∂z;
χ =  s2/(2*∂z2);
ζ = μ/∂z + s2/(2*∂z2);

#This will be the upperdiagonal of the matrix Aswitch
# updiag=zeros(I,1); #This is necessary because of the peculiar way spdiags is defined.
# for j=1:J
#     updiag= [updiag; repeat([ζ[j]], I, 1)];
# end
# #This will be the center diagonal of the matrix Aswitch
# centdiag=repeat([χ[1]+yy[1]],I);
# for j=2:J-1
#     centdiag=[centdiag;repeat([yy[j]],I)];
# end

# centdiag=[centdiag;repeat([yy[J]+ζ[J]],I)];

# #This will be the lower diagonal of the matrix Aswitch
# lowdiag=repeat([χ[2]],I);
# for j=3:J
#     lowdiag=[lowdiag;repeat([χ[j]],I)];
# end

updiag   = repeat(ζ[1:J-1], inner=I)    # length I*(J-1)
centdiag = vcat(repeat([χ[1] + yy[1]], I),
                [repeat([yy[j]], I) for j in 2:J-1]...,
                repeat([yy[J] + ζ[J]], I))   # length I*J
lowdiag  = repeat(χ[2:J], inner=I)      # length I*(J-1)


#Add up the upper, center, and lower diagonal into a sparse matrix
# Aswitch=spdiags(centdiag,0,I*J,I*J)+spdiags(lowdiag,-I,I*J,I*J)+spdiags(updiag,I,I*J,I*J);

# spdiagm(-1 => lowdiag', 1 => updiag')
# 0 => centdiag, 

# Aswitch = spdiagm(-1 => lowdiag, 1 => updiag, 0 => centdiag);
Aswitch = spdiagm(-I => lowdiag, I => updiag, 0 => centdiag)
v = zeros(I,J,N);
gg = Vector{Any}(undef, N+1);
maxit = 1000;
convergence_criterion = 10^(-5);

# terminal condition on value function: value of death \approx 0
small_number1 = 10^(-8); small_number2 = 10^(-8);
v_terminal = small_number1.*(small_number2 .+ aa).^(1-γ)./(1-γ);

V = v_terminal;

# solving the problem ====

for n = N:-1:1
    println("age = ", n * ∂t)
    v[:, :, n] .= V

    # forward difference
    ∂Vf[1:I-1, :] .= (V[2:I, :] .- V[1:I-1, :]) ./ ∂a
    ∂Vf[I, :] .= (w .* z' .+ r .* amax).^(-γ)
    # backward difference
    ∂Vb[2:I, :] .= (V[2:I, :] .- V[1:I-1, :]) ./ ∂a
    ∂Vb[1, :] .= (w .* z' .+ r .* amin).^(-γ)
    
    # consumption and savings with forward difference
    cf = ∂Vf.^(-1/γ);
    ssf = w*zz + r.*aa - cf;
    # consumption and savings with backward difference
    cb = ∂Vb.^(-1/γ);
    ssb = w*zz + r.*aa - cb;
    # consumption and derivative of value function at steady state
    c0 = w*zz + r.*aa;

    # upwind method
    If = ssf .> 0; # positive drift --> forward difference
    Ib = ssb .< 0; # negative drift --> backward difference
    I0 = .!(If .| Ib); # at steady state

    c = cf.*If + cb.*Ib + c0.*I0;
    u = c.^(1-γ)/(1-γ);

    # CONSTRUCT MATRIX
    # X = .- min.(ssb,0)./∂a;
    # Y = .- max.(ssf,0)./∂a + min.(ssb,0)./∂a;
    # Z = max.(ssf,0)./∂a;
    # CONSTRUCT MATRIX  (replace your X,Y,Z block with this)
    X = max.(0, -ssb) ./ ∂a          # outflow to i-1 (subdiagonal), nonnegative
    Z = max.(0,  ssf) ./ ∂a          # outflow to i+1 (superdiagonal), nonnegative

    # Enforce state-constraint boundaries in a-direction
    X[1,  :] .= 0.0                  # no flow to i=0
    Z[end,:] .= 0.0                  # no flow to i=I+1
    # println(I)
    # Center diagonal to make each row sum to zero
    Y = .-(X .+ Z)

    
    # superdiagonal (k = +1): for blocks j = 1..J-1 pad a trailing 0; last block has only I-1 entries
    # updiag = vcat([vcat(Z[1:I-1, j], 0) for j in 1:J-1]..., Z[1:I-1, J])  # length 4499
    # superdiagonal (k = +1)
    # updiag = vcat([vcat(Z[1:I-1, j], 0) for j in 1:J]...)   # length = I*J
    # updiag = updiag[1:end-1]    
    # main diagonal (k = 0)
    # centdiag = reshape(Y, I*J)  # length 4500

    # subdiagonal (k = -1): first block has I-1 entries; for blocks j = 2..J pad a leading 0
    # lowdiag = vcat(X[2:I, 1], [vcat(0, X[2:I, j]) for j in 2:J]...)        # length 4499
    # lowdiag = vcat([vcat(0, X[2:I, j]) for j in 1:J]...)   # length = I*J
    # lowdiag = lowdiag[2:end]    

    # superdiagonal (k = +1): pad one 0 per block, then trim the very last
    # updiag = vcat([vcat(Z[1:I-1, j], 0.0) for j in 1:J]...)   # length I*J
    # updiag = updiag[1:end-1]                                  # length I*J-1

    # # subdiagonal (k = -1): pad one 0 at top of each block, then drop the very first
    # lowdiag = vcat([vcat(0.0, X[2:I, j]) for j in 1:J]...)    # length I*J
    # lowdiag = lowdiag[2:end]                                  # length I*J-1
    
    # # main diagonal
    # centdiag = reshape(Y, I*J)

    updiag_z   = repeat(ζ[1:J-1], inner=I)              # length I*(J-1)
    centdiag_z = vcat(repeat([χ[1] + yy[1]], I),
                    [repeat([yy[j]], I) for j in 2:J-1]...,
                    repeat([yy[J] + ζ[J]], I))        # length I*J
    lowdiag_z  = repeat(χ[2:J], inner=I)  
    # A = Aswitch.+ spdiagm(-1 => lowdiag, 1 => updiag, 0 => centdiag);
    A = Aswitch .+ spdiagm(-I => lowdiag, I => updiag, 0 => centdiag);
    # Aswitch = spdiagm(-I => lowdiag, I => updiag, 0 => centdiag)
    # println((A))

    rowsums = vec(sum(A, dims = 2))
    max_abs_rowsum = maximum(abs.(rowsums))
    if max_abs_rowsum > 1e-9
        println("Improper Transition Matrix: row sums must be ~0. Max |row sum| = $(max_abs_rowsum)")
        bad_i = argmax(abs.(rowsums))
        println("Worst row = $(bad_i), sum = $(rowsums[bad_i])")
        break
    end
# end
    # rowsums = vec(sum(A, dims = 2))             # somas por linha como Vector
    # max_abs_rowsum = maximum(abs.(rowsums))     # escalar
    # if max_abs_rowsum > 1e-9
    #     println("Improper Transition Matrix: row sums must be ~0. Max |row sum| = $(max_abs_rowsum)")
    #     bad_i = argmax(abs.(rowsums))
    #     println("Worst row = $(bad_i), sum = $(rowsums[bad_i])")
    #     break
    # end

end
