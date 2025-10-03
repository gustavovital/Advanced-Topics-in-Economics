using SparseArrays
using Plots
# gr(show = :window)
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
R_age = 65;
∂t=T/N;
∂age = round(R_age/∂t);

pension_θ = 0.5; # pension rate
pension = pension_θ * w * zmean;

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

updiag_z   = repeat(ζ[1:J-1], inner=I)    # length I*(J-1)
centdiag_z = vcat(repeat([χ[1] + yy[1]], I),
                [repeat([yy[j]], I) for j in 2:J-1]...,
                repeat([yy[J] + ζ[J]], I))   # length I*J
lowdiag_z  = repeat(χ[2:J], inner=I)      # length I*(J-1)


#Add up the upper, center, and lower diagonal into a sparse matrix
# Aswitch=spdiags(centdiag,0,I*J,I*J)+spdiags(lowdiag,-I,I*J,I*J)+spdiags(updiag,I,I*J,I*J);

# spdiagm(-1 => lowdiag', 1 => updiag')
# 0 => centdiag, 

# Aswitch = spdiagm(-1 => lowdiag, 1 => updiag, 0 => centdiag);
Aswitch = spdiagm(-I => lowdiag_z, I => updiag_z, 0 => centdiag_z)
v = zeros(I,J,N);
gg = Vector{Any}(undef, N+1);
A_t = Vector{SparseMatrixCSC{Float64, Int}}(undef, N)
c_t = Vector{Array{Float64,2}}(undef, N)
ss_t = Vector{Array{Float64,2}}(undef, N)
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
    eps = 1e-12
    cf = clamp.(∂Vf, eps, Inf).^(-1/γ)
    ssf = w*zz + r.*aa - cf
    # consumption and savings with backward difference
    cb = clamp.(∂Vb, eps, Inf).^(-1/γ)
    ssb = w*zz + r.*aa - cb
    # consumption and derivative of value function at steady state
    c0 = max.(w*zz .+ r.*aa, 1e-12)

    # upwind method
    If = float.(ssf .> 0)
    Ib = float.(ssb .< 0)
    I0 = 1 .- If .- Ib

    c = cf.*If .+ cb.*Ib .+ c0.*I0
    c_safe = max.(c, 1e-12)
    u = c_safe.^(1-γ) ./ (1-γ)

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

    
    # superdiagonal (k = +1): pad one 0 per block, then trim the very last
    updiag_a = vcat([vcat(Z[1:I-1, j], 0.0) for j in 1:J]...)
    updiag_a = updiag_a[1:end-1]
    # subdiagonal (k = -1): pad one 0 at top of each block, then drop the very first
    lowdiag_a = vcat([vcat(0.0, X[2:I, j]) for j in 1:J]...)
    lowdiag_a = lowdiag_a[2:end]
    # main diagonal
    centdiag_a = reshape(Y, I*J)
    A = Aswitch .+ spdiagm(-1 => lowdiag_a, 1 => updiag_a, 0 => centdiag_a)
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

    # Note the syntax for the cell array
    A_t[n] = A;
    # B = (1/dt + rho)*speye(I*J) - A;

    B = (1/∂t + ρ) * spdiagm(0 => ones(I*J)) - A   
    u_stacked = reshape(u,I*J,1);
    V_stacked = reshape(V,I*J,1);
    
    b = u_stacked + V_stacked/∂t;
    V_stacked = B\b; #SOLVE SYSTEM OF EQUATIONS
    
    V = reshape(V_stacked,I,J);
    c_t[n] = copy(c)
    # ss_t{n} = w*zz + r.*aa - c;
    # ss_t[n] = copy(w .* zz .+ r .* aa .- c)
    if n > ∂age
        ss_t[n] = copy( r .* aa .- c)
    elseif n == ∂age
        ss_t[n] = copy(pension .+ r .* aa .- c)
    else
        ss_t[n] = copy(w .* zz .+ r .* aa .- c)
    end

end

# =========
# --- First plots ---
# plot(a, c_t[1][:,1], label="c_t[1]")   # first column of c_t[1]
# plot!(a, ss_t[1][:,1], label="ss_t[1]")
# plot!(a, zeros(I), label="0", linestyle=:dash, color=:black)

# Another plot for N-1
# plot(a, c_t[N-1][:,1], label="c_t[N-1]")
# plot!(a, ss_t[N-1][:,1], label="ss_t[N-1]")
# plot!(a, zeros(I), label="0", linestyle=:dash, color=:black)

# --- Subplots like MATLAB's subplot(1,2,1) and subplot(1,2,2) ---
p1 = plot(a, c_t[Int(1/∂t)][:,1], label="Age 1, Lowest Income", lw=2)
plot!(p1, a, c_t[Int(1/∂t)][:,J], label="Age 1, Highest Income", lw=2)
plot!(p1, a, c_t[Int(40/∂t)][:,1], label="Age 40, Lowest Income", lw=2)
plot!(p1, a, c_t[Int(40/∂t)][:,J], label="Age 40, Highest Income", lw=2)
plot!(p1, a, c_t[Int(70/∂t)][:,1], label="Age 70, Lowest Income", lw=2)
plot!(p1, a, c_t[Int(70/∂t)][:,J], label="Age 70, Highest Income", lw=2)
xlabel!(p1, "Wealth")
ylabel!(p1, "Consumption, c(a,z,t)")
ylims!(p1, (0,5))
plot!(p1, legend=:topright, fontsize=12)

p2 = plot(a, ss_t[Int(1/∂t)][:,1], label="Age 1, Lowest Income", lw=2)
plot!(p2, a, ss_t[Int(1/∂t)][:,J], label="Age 1, Highest Income", lw=2)
plot!(p2, a, ss_t[Int(40/∂t)][:,1], label="Age 40, Lowest Income", lw=2)
plot!(p2, a, ss_t[Int(40/∂t)][:,J], label="Age 40, Highest Income", lw=2)
plot!(p2, a, ss_t[Int(70/∂t)][:,1], label="Age 70, Lowest Income", lw=2)
plot!(p2, a, ss_t[Int(70/∂t)][:,J], label="Age 70, Highest Income", lw=2)
plot!(p2, a, zeros(I), label="0", linestyle=:dash, color=:black, lw=2)
xlabel!(p2, "Wealth")
ylabel!(p2, "Saving, s(a,z,t)")
ylims!(p2, (-5,2))
plot!(p2, legend=:topright, fontsize=12)

plot(p1, p2, layout=(1,2), size=(1000,400))
