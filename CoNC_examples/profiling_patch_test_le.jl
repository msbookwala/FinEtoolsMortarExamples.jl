# CoNC patch test

# plain cg does better than amg

using FinEtools
using FinEtools.AlgoBaseModule: matrix_blocked, vector_blocked
using FinEtoolsDeforLinear
using FinEtoolsDeforLinear.AlgoDeforLinearModule
using LinearAlgebra
using SparseArrays
using Statistics
using CoNCMOR: CoNCData, transfmatrix, LegendreBasis
using Krylov
using LinearOperators
using AlgebraicMultigrid
using Plots
using Profile


function patch_coordinates(panelX, list)
    krondelta(i, k) = i == k ? 1.0 : 0.0
    lX = panelX[list, :]
    center = mean(lX, dims=1)

    for j in 1:size(lX, 1)
        lX[j, :] -= center[:]
    end

    It = fill(0.0, 3, 3)

    for j in 1:size(lX, 1)
        r2 = dot(lX[j, :], lX[j, :])

        for i in 1:3
            for k in i:3
                It[i, k] += krondelta(i, k) * r2 - lX[j, i] * lX[j, k]
            end
        end
    end

    for i in 1:3
        for k in 1:i-1
            It[i, k] = It[k, i]
        end
    end

    @assert It == It'

    epsol = eigen(It)
    e1 = epsol.vectors[:, 1]
    e2 = epsol.vectors[:, 2]

    lxy = fill(0.0, length(list), 2)

    for j in 1:size(lX, 1)
        lxy[j, 1] = dot(lX[j, :], e1)
        lxy[j, 2] = dot(lX[j, :], e2)
    end

    return lxy
end


function run_patch_test(nelem1, nelem2, n_partitions; profile = false, profile_path = nothing)
    N_elem1 = nelem1
    N_elem2 = nelem2
    lam_order = 1

    E = 1.0
    nu = 0.0
    traction = 1.0

    MR = DeforModelRed3D
    material = MatDeforElastIso(MR, 0.0, E, nu, 0.0)


    width1 = 0.5
    height1 = 1.0
    depth1 = 1.0

    fens1, fes1 = T4block(width1, height1, depth1, N_elem1, N_elem1, N_elem1)
    Rule1 = TetRule(4)

    boundaryfes1 = meshboundary(fes1)
    edge_fes1 = subset(boundaryfes1, selectelem(fens1, boundaryfes1; box=[0.5, 0.5, 0.0, 1.0, 0.0, 1.0], inflate=1e-8))

    geom1 = NodalField(fens1.xyz)
    u1 = NodalField(zeros(size(fens1.xyz, 1), 3))

    dbc_nodes1 = selectnode(fens1; box=[0.0, 0.0, 0.0, 1.0, 0.0, 1.0], inflate=1e-8)

    for c in 1:3
        setebc!(u1, dbc_nodes1, c, 0.0)
    end

    applyebc!(u1)
    numberdofs!(u1)

    femm1 = FEMMDeforLinear(MR, IntegDomain(fes1, Rule1), material)
    K1 = stiffness(femm1, geom1, u1)
    K1_ff = matrix_blocked(K1, nfreedofs(u1), nfreedofs(u1))[:ff]

    F1_ff = zeros(Float64, nfreedofs(u1))

    width2 = 0.5
    height2 = 1.0
    depth2 = 1.0

    fens2, fes2 = T4block(width2, height2, depth2, N_elem2, N_elem2, N_elem2)
    fens2.xyz[:, 1] .+= 0.5
    Rule2 = TetRule(4)

    boundaryfes2 = meshboundary(fes2)
    edge_fes2 = subset(boundaryfes2, selectelem(fens2, boundaryfes2; box=[0.5, 0.5, 0.0, 1.0, 0.0, 1.0], inflate=1e-8))

    geom2 = NodalField(fens2.xyz)
    u2 = NodalField(zeros(size(fens2.xyz, 1), 3))

    applyebc!(u2)
    numberdofs!(u2)

    femm2 = FEMMDeforLinear(MR, IntegDomain(fes2, Rule2), material)
    K2 = stiffness(femm2, geom2, u2)
    K2_ff = matrix_blocked(K2, nfreedofs(u2), nfreedofs(u2))[:ff]

    l2 = selectelem(fens2, boundaryfes2; box=[1.0, 1.0, 0.0, 1.0, 0.0, 1.0], inflate=1e-8)
    el2femm = FEMMBase(IntegDomain(subset(boundaryfes2, l2), TriRule(3)))
    fi2 = ForceIntensity(Float64[traction, 0.0, 0.0])
    F2 = distribloads(el2femm, geom2, u2, fi2, 2)
    F2_ff = vector_blocked(F2, nfreedofs(u2))[:f]


    _, meta = common_refinement(fens1, edge_fes1, fens2, edge_fes2; lam_order=lam_order, h=0.03)

    fens_u = meta["fens_u"]
    fes_u = meta["fes_u"]


    ppartitioning = pointpartitioning(fens_u.xyz, n_partitions)
    partitionnumbers = unique(ppartitioning)

    println("Number of CoNC clusters = ", length(partitionnumbers))

    mor = CoNCData(list -> patch_coordinates(fens_u.xyz, list), ppartitioning)

    lam = NodalField(fens_u.xyz[:, 1])
    numberdofs!(lam)

    Phi = transfmatrix(mor, LegendreBasis, 2, lam)

    println("Fine scalar LM dimension    = ", size(Phi, 1))
    println("Reduced scalar LM dimension = ", size(Phi, 2))
    println("Vector LM dimension         = ", 3 * size(Phi, 2))

    D1n = sparse(Phi' * meta["M"] * meta["PiA"])
    D2n = sparse(Phi' * meta["M"] * meta["PiB"])

    I3 = sparse(Matrix{Float64}(I, 3, 3))

    D1all = sparse(kron(D1n, I3))
    D2all = sparse(kron(D2n, I3))

    dofnums1 = vec(permutedims(u1.dofnums))
    dofnums2 = vec(permutedims(u2.dofnums))

    free_cols1 = sortperm(dofnums1)[1:nfreedofs(u1)]
    free_cols2 = sortperm(dofnums2)[1:nfreedofs(u2)]

    D1 = sparse(D1all[:, free_cols1])
    D2 = sparse(D2all[:, free_cols2])


    n1 = size(K1_ff, 1)
    n2 = size(K2_ff, 1)
    n = n1 + n2

    K = blockdiag(sparse(K1_ff), sparse(K2_ff))
    F = vcat(F1_ff, F2_ff)
    D = sparse(hcat(D1, -D2))
    Dt = transpose(D)

    println("Primal DOFs             = ", n)
    println("Reduced LM constraints = ", size(D, 1))


    G = sparse(D * Dt)
    Gfac = cholesky(Symmetric(G))

    n_lam = size(D, 1)

    qproj = zeros(Float64, n_lam)
    alphaproj = zeros(Float64, n_lam)

    function project_constraint!(y, x)
        mul!(qproj, D, x)
        alphaproj .= Gfac \ qproj
        copyto!(y, x)
        mul!(y, Dt, alphaproj, -1.0, 1.0)
        return y
    end

    px = zeros(Float64, n)
    Kpx = zeros(Float64, n)

    function projected_K_mul!(y, x)
        project_constraint!(px, x)
        mul!(Kpx, K, px)
        project_constraint!(y, Kpx)
        @. y += x - px
        return y
    end

    Aproj = LinearOperator(Float64, n, n, true, true, projected_K_mul!)

    Fp = similar(F)
    project_constraint!(Fp, F)

    K1_amg = sparse(copy(K1_ff))
    K2_amg = sparse(copy(K2_ff))

    nK2 = size(K2_amg, 1)
    id = sparse(I, nK2,nK2)
    K2_amg += id


    amg_setup1 = @elapsed ml1 = smoothed_aggregation(K1_amg)
    AMG1 = aspreconditioner(ml1)


    amg_setup2 = @elapsed ml2 = smoothed_aggregation(K2_amg)
    AMG2 = aspreconditioner(ml2)

    amg_setup_time = amg_setup1 + amg_setup2


    rproj = zeros(Float64, n)
    zlocal = zeros(Float64, n)

    function projected_amg_mul!(z, r)
        project_constraint!(rproj, r)

        r1 = view(rproj, 1:n1)
        r2 = view(rproj, n1+1:n)

        z1 = view(zlocal, 1:n1)
        z2 = view(zlocal, n1+1:n)

        ldiv!(z1, AMG1, r1)
        ldiv!(z2, AMG2, r2)

        project_constraint!(z, zlocal)

        @. z += r - rproj

        return z
    end

    Mproj = LinearOperator(Float64, n, n, true, true, projected_amg_mul!)


    u = zeros(Float64, n)
    stats = nothing

    time_pcg = @elapsed begin
        # u, stats = Krylov.cg(Aproj, Fp; M=Mproj, rtol=1e-12, atol=1e-14, itmax=5000, verbose=1)
        u, stats = Krylov.cg(Aproj, Fp; rtol=1e-12, atol=1e-14, itmax=5000, verbose=1)

    end

    println()
    println("Iterations = ", stats.niter)
    println("AMG-PCG solve time = ", time_pcg, " s")


    utmp = copy(u)

    project_constraint!(u, utmp)


    Ku = K * u
    rhs_lambda = D * (F - Ku)
    lambda_reduced = Gfac \ rhs_lambda

    u1vec = u[1:n1]
    u2vec = u[n1+1:n1+n2]

    scattersysvec!(u1, u1vec)
    scattersysvec!(u2, u2vec)


        return (
        cg_time = time_pcg,
        iterations = stats.niter,
        nprimal = n,
        nlambda = size(D, 1),
    )
end


warmup = run_patch_test(5, 7, 4)
results_dir = joinpath("results", splitext(basename(@__FILE__))[1])
mkpath(results_dir)

for file in readdir(results_dir; join = true)
    rm(file; recursive = true)
end


cases = [(5, 7), (10, 14), (15, 21), (20, 28)]
results = Tuple{Int, Int, NamedTuple}[]
for (nelem1, nelem2) in cases
    result = run_patch_test(nelem1, nelem2, 4)
    println(
        "nelem1 = $nelem1, nelem2 = $nelem2, nprimal = $(result.nprimal), ",
        "iterations = $(result.iterations), cg_time = $(result.cg_time)",
    )
    push!(results, (nelem1, nelem2, result))
end

open(joinpath(results_dir, "profile_cg.csv"), "w") do io
    println(io, "nelem1,nelem2,npartitions,nprimal,nlambda,iterations,cg_time")
    for (nelem1, nelem2, result) in results
        println(
            io,
            "$nelem1,$nelem2,4,$(result.nprimal),$(result.nlambda),",
            "$(result.iterations),$(result.cg_time)",
        )
    end
end

ndofs = [result.nprimal for (_, _, result) in results]
cg_times = [result.cg_time for (_, _, result) in results]
linear_reference = cg_times[1] .* ndofs ./ ndofs[1]
quadratic_reference = cg_times[1] .* (ndofs ./ ndofs[1]) .^ 2

cg_plot = plot(
    ndofs,
    cg_times;
    marker = :circle,
    linewidth = 2,
    label = "CG time",
    xlabel = "Primal DOFs",
    ylabel = "CG time (s)",
    xscale = :log10,
    yscale = :log10,
    title = "CG time scaling for the CoNC patch test",
    legend = :topleft,
    dpi = 300,
)
plot!(cg_plot, ndofs, linear_reference; linestyle = :dash, label = "O(N) reference")
plot!(cg_plot, ndofs, quadratic_reference; linestyle = :dot, label = "O(N^2) reference")
savefig(cg_plot, joinpath(results_dir, "cg_time_scaling.png"))
