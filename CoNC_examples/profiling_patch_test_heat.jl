# CoNC patch test
# heat equation; solved with null space projection
# additional steps to make the whole system SPD : PtKP + (I-P)

using FinEtools
using FinEtools.AlgoBaseModule: solve_blocked!, matrix_blocked, vector_blocked
using FinEtoolsHeatDiff
using FinEtoolsHeatDiff.AlgoHeatDiffModule
using FinEtools.MeshExportModule.VTK: vtkexportmesh, T3, vtkexportvectors
using LinearAlgebra
using SparseArrays
using Statistics
using ShellStructureTopo: create_partitions
using CoNCMOR: CoNCData, nclusters, nfuncspercluster, nbasisfunctions, transfmatrix, LegendreBasis
using Krylov
using LinearOperators
using AlgebraicMultigrid
using Plots

function patch_coordinates(panelX, list)
    krondelta(i, k) = i == k ? 1.0 : 0.0
    lX = panelX[list, :]
    center = mean(lX, dims = 1)
    for j = 1:size(lX, 1)
        lX[j, :] -= center[:]
    end
    It = fill(0.0, 3, 3)
    for j = 1:size(lX, 1)
        r2 = dot(lX[j, :], lX[j, :])
        for i = 1:3
            for k = i:3
                It[i, k] += krondelta(i, k) * r2 - lX[j, i] * lX[j, k]
            end
        end
    end
    for i = 1:3
        for k = 1:(i-1)
            It[i, k] = It[k, i]
        end
    end
    @assert It == It'
    epsol = eigen(It)
    normal = epsol.vectors[:, 3]
    e1 = epsol.vectors[:, 1]
    e2 = epsol.vectors[:, 2]
    lxy = fill(0.0, length(list), 2)
    for j = 1:size(lX, 1)
        lxy[j, 1] = dot(lX[j, :], e1)
        lxy[j, 2] = dot(lX[j, :], e2)
    end
    return lxy
end

function run_patch_test(N_elem1, N_elem2, n_clusters)
    # N_elem1 = 30
    # N_elem2 = 50
    left_m = "t"
    right_m = "t"
    skew = 0.0
    lam_order = 1
    bend = 0.3

    kappa = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    material = MatHeatDiff(kappa)

    ################################################################################
    # SUBDOMAIN 1
    ################################################################################

    width1 = 0.5
    height1 = 1.0
    depth1 = 1.0

    if left_m == "h"
        fens1, fes1 = H8block(width1, height1, depth1, floor(Int, N_elem1), N_elem1, N_elem1)
        Rule1 = GaussRule(3, 2)
    else
        fens1, fes1 = T4block(width1, height1, depth1, floor(Int, N_elem1), N_elem1, N_elem1)
        Rule1 = TetRule(4)
    end

    boundaryfes1 = meshboundary(fes1)
    edge_fes1 = subset(
        boundaryfes1,
        selectelem(fens1, boundaryfes1; box = [width1, width1, 0.0, height1, 0.0, depth1], inflate = 1e-8),
    )

    fens1.xyz[:, 1] .+= bend .* fens1.xyz[:, 1] .* (fens1.xyz[:, 2] .- 0.5) .^ 2

    geom1 = NodalField(fens1.xyz)
    T1 = NodalField(zeros(size(fens1.xyz, 1), 1))

    applyebc!(T1)
    numberdofs!(T1)

    femm1 = FEMMHeatDiff(IntegDomain(fes1, Rule1), material)
    K1 = conductivity(femm1, geom1, T1)
    K1_ff = matrix_blocked(K1, nfreedofs(T1), nfreedofs(T1))[:ff]

    l1 = selectelem(fens1, meshboundary(fes1); box = [0.0, 0.0, 0.0, height1, 0.0, depth1], inflate = 1e-8)
    el1femm = FEMMBase(IntegDomain(subset(meshboundary(fes1), l1), TriRule(3)))
    fi1 = ForceIntensity(Float64[-1.0])
    F1 = distribloads(el1femm, geom1, T1, fi1, 2)
    F1_ff = vector_blocked(F1, nfreedofs(T1))[:f]

    ################################################################################
    # SUBDOMAIN 2
    ################################################################################

    width2 = 0.5
    height2 = 1.0
    depth2 = 1.0

    if right_m == "h"
        fens2, fes2 = H8block(width2, height2, depth2, floor(Int, N_elem2), N_elem2, N_elem2)
        Rule2 = GaussRule(3, 2)
    else
        fens2, fes2 = T4block(width2, height2, depth2, floor(Int, N_elem2), N_elem2, N_elem2)
        Rule2 = TetRule(4)
        fens2.xyz[:, 1] .+= 0.5
    end

    boundaryfes2 = meshboundary(fes2)
    edge_fes2 = subset(
        boundaryfes2,
        selectelem(fens2, boundaryfes2; box = [0.5, 0.5, 0.0, height2, 0.0, depth2], inflate = 1e-8),
    )

    fens2.xyz[:, 1] .+= bend .* (1.0 .- fens2.xyz[:, 1]) .* (fens2.xyz[:, 2] .- 0.5) .^ 2

    geom2 = NodalField(fens2.xyz)
    T2 = NodalField(zeros(size(fens2.xyz, 1), 1))

    box2 = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    dbc_nodes2 = selectnode(fens2; box = box2, inflate = 1e-8)

    for i in dbc_nodes2
        setebc!(T2, [i], 1, 0.0)
    end

    applyebc!(T2)
    numberdofs!(T2)

    femm2 = FEMMHeatDiff(IntegDomain(fes2, Rule2), material)
    K2 = conductivity(femm2, geom2, T2)
    K2_ff = matrix_blocked(K2, nfreedofs(T2), nfreedofs(T2))[:ff]

    l2 = selectelem(fens2, meshboundary(fes2); box = [1.0, 1.0, 0.0, height2, 0.0, depth2], inflate = 1e-8)
    el2femm = FEMMBase(IntegDomain(subset(meshboundary(fes2), l2), TriRule(3)))
    fi2 = ForceIntensity(Float64[1.0])
    F2 = distribloads(el2femm, geom2, T2, fi2, 2)
    F2_ff = vector_blocked(F2, nfreedofs(T2))[:f]

    ################################################################################
    # COMMON REFINEMENT
    ################################################################################

    _, meta = common_refinement(fens1, edge_fes1, fens2, edge_fes2; lam_order = lam_order, h = 0.03)

    fens_u = meta["fens_u"]
    fes_u = meta["fes_u"]

    ################################################################################
    # CoNC REDUCED LM SPACE
    ################################################################################

    n_partitions = n_clusters
    ppartitioning = pointpartitioning(fens_u.xyz, n_partitions)
    partitionnumbers = unique(ppartitioning)

    println("Number of CoNC clusters = ", length(partitionnumbers))

    mor = CoNCData(list -> patch_coordinates(fens_u.xyz, list), ppartitioning)

    lam = NodalField(fens_u.xyz[:, 1])
    numberdofs!(lam)

    Phi = transfmatrix(mor, LegendreBasis, 2, lam)

    println("Fine LM dimension    = ", size(Phi, 1))
    println("Reduced LM dimension = ", size(Phi, 2))
    println("LM reduction factor  = ", size(Phi, 1) / size(Phi, 2))

    D1 = sparse(Phi' * meta["M"] * meta["PiA"])
    D2 = sparse(Phi' * meta["M"] * meta["PiB"])

    free_nodes2 = setdiff(1:count(fens2), dbc_nodes2)
    D2 = D2[:, free_nodes2]

    ################################################################################
    # GLOBAL PRIMAL AND CONSTRAINT MATRICES
    ################################################################################

    n1 = size(K1_ff, 1)
    n2 = size(K2_ff, 1)
    n = n1 + n2

    K = blockdiag(sparse(K1_ff), sparse(K2_ff))
    F = vcat(F1_ff, F2_ff)
    D = sparse(hcat(D1, -D2))
    Dt = transpose(D)

    println("Primal DOFs            = ", n)
    println("Reduced LM constraints = ", size(D, 1))

    ################################################################################
    # NULL-SPACE PROJECTOR
    #
    # P = I - D'*(D*D')^-1*D
    ################################################################################

    G = sparse(D * Dt)
    Gfac = cholesky(Symmetric(G))

    nλ = size(D, 1)
    qproj = zeros(Float64, nλ)
    alphaproj = zeros(Float64, nλ)

    function project_constraint!(y, x)
        mul!(qproj, D, x)
        alphaproj .= Gfac \ qproj
        copyto!(y, x)
        mul!(y, Dt, alphaproj, -1.0, 1.0)
        return y
    end

    ################################################################################
    # PROJECTED SPD OPERATOR
    # A = P*K*P + (I-P)
    ################################################################################

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

    ################################################################################
    # AMG PRECONDITIONER
    #
    # K1 is floating, so it has the constant null mode.
    # K2 is already nonsingular because it contains the actual DBC.
    ################################################################################

    K1_amg = sparse(copy(K1_ff))
    K2_amg = sparse(copy(K2_ff))

    diagK1 = abs.(diag(K1_amg))
    anchor_dof = argmax(diagK1)
    anchor_scale = diagK1[anchor_dof]

    # println()
    # println("AMG artificial anchor DOF   = ", anchor_dof)
    # println("AMG artificial anchor value = ", anchor_scale)

    K1_amg[anchor_dof, anchor_dof] += anchor_scale

    # println()
    # println("Building AMG hierarchy for subdomain 1...")
    amg_setup1 = @elapsed ml1 = smoothed_aggregation(K1_amg)
    AMG1 = aspreconditioner(ml1)

    # println("Building AMG hierarchy for subdomain 2...")
    amg_setup2 = @elapsed ml2 = smoothed_aggregation(K2_amg)
    AMG2 = aspreconditioner(ml2)

    amg_setup_time = amg_setup1 + amg_setup2

    # println("AMG setup 1 = ", amg_setup1, " s")
    # println("AMG setup 2 = ", amg_setup2, " s")
    # println("Total AMG setup = ", amg_setup_time, " s")

    ################################################################################
    # PROJECTED BLOCK-AMG PRECONDITIONER
    # M^-1 = P * blockdiag(AMG1^-1, AMG2^-1) * P + (I-P)
    ################################################################################

    rproj = zeros(Float64, n)
    zlocal = zeros(Float64, n)

    function projected_amg_mul!(z, r)
        project_constraint!(rproj, r)
        r1 = view(rproj, 1:n1)
        r2 = view(rproj, (n1+1):n)
        z1 = view(zlocal, 1:n1)
        z2 = view(zlocal, (n1+1):n)
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
        u, stats = Krylov.cg(Aproj, Fp; M = Mproj, rtol = 1e-10, atol = 1e-12, itmax = 5000, verbose = 0)
    end

    println()
    println("Iterations = ", stats.niter)
    println("Solve time = ", time_pcg, " s")

    # FINAL PROJECTION

    utmp = copy(u)
    project_constraint!(u, utmp)

    # RECOVER REDUCED LAGRANGE MULTIPLIER

    Ku = K * u
    rhs_lambda = D * (F - Ku)
    lambda_reduced = Gfac \ rhs_lambda


    constraint_error = norm(D * u)
    equilibrium_error = norm(K * u + Dt * lambda_reduced - F) / max(norm(F), eps(Float64))

    # println()
    # println("Constraint error             = ", constraint_error)
    # println("Equilibrium error            = ", equilibrium_error)

    # # TIMING SUMMARY

    # println()
    # println("================================================")
    # println("SOLVER COMPARISON")
    # println("================================================")
    # println("Fine LM DOFs                 = ", size(Phi, 1))
    # println("Reduced LM DOFs              = ", size(Phi, 2))
    # println("Primal DOFs                  = ", n)
    # println("AMG setup time               = ", amg_setup_time, " s")
    # println("AMG-PCG solve time           = ", time_pcg, " s")
    # println("Constraint error             = ", constraint_error)
    # println("Equilibrium error            = ", equilibrium_error)
    # println("================================================")

    # SCATTER SOLUTION

    u1 = u[1:n1]
    u2 = u[(n1+1):(n1+n2)]

    scattersysvec!(T1, u1)
    scattersysvec!(T2, u2)

    # ERROR
    sol(x) = x[1] - 1.0
    err1 = L2error(femm1, geom1, T1, sol)
    err2 = L2error(femm2, geom2, T2, sol)

    # OUTPUT



    # File1 = "$filename/mesh_left.vtk"
    # vtkexportmesh(File1, fens1, fes1; scalars=[("Temperature", T1.values), ("Err", err1.values)])

    # File2 = "$filename/mesh_right.vtk"
    # vtkexportmesh(File2, fens2, fes2; scalars=[("Temperature", T2.values), ("Err", err2.values)])

    # vtkexportmesh("$filename/CONC.vtk", fens_u, fes_u; scalars=[("point_partitioning", ppartitioning)])

    # println()
    # println("Finished.")

    return (
        solve_time = time_pcg,
        iterations = stats.niter,
        nprimal = n,
        nlambda = size(D, 1),
        constraint_error = constraint_error,
        equilibrium_error = equilibrium_error,
    )
end

# warmup
run_patch_test(15, 25, 2)

results_dir = joinpath("results", splitext(basename(@__FILE__))[1])
mkpath(results_dir)

for file in readdir(results_dir; join = true)
    rm(file; recursive = true)
end

NDOF_CASES = [(15, 25), (21, 35), (30, 50), (60, 100)]
FIXED_NPATCHES = 8

FIXED_N1 = 30
FIXED_N2 = 50
NPATCH_CASES = [2, 4, 8, 16, 32, 64, 128]

function write_ndof_results(path, results)
    open(path, "w") do io
        println(
            io,
            "N_elem1,N_elem2,nclusters,nprimal,nlambda,iterations,solve_time,constraint_error,equilibrium_error",
        )
        for (N1, N2, result) in results
            println(
                io,
                "$N1,$N2,$FIXED_NPATCHES,$(result.nprimal),$(result.nlambda),$(result.iterations),$(result.solve_time),$(result.constraint_error),$(result.equilibrium_error)",
            )
        end
    end
end

function write_ncluster_results(path, results)
    open(path, "w") do io
        println(
            io,
            "N_elem1,N_elem2,nclusters,nprimal,nlambda,iterations,solve_time,constraint_error,equilibrium_error",
        )
        for (n_clusters, result) in results
            println(
                io,
                "$FIXED_N1,$FIXED_N2,$n_clusters,$(result.nprimal),$(result.nlambda),$(result.iterations),$(result.solve_time),$(result.constraint_error),$(result.equilibrium_error)",
            )
        end
    end
end

ndof_results = Tuple{Int,Int,NamedTuple}[]
for (N1, N2) in NDOF_CASES
    result = run_patch_test(N1, N2, FIXED_NPATCHES)
    println("N_elem1 = $N1, N_elem2 = $N2, nprimal = $(result.nprimal), nlambda = $(result.nlambda), iterations = $(result.iterations), solve_time = $(result.solve_time), constraint_error = $(result.constraint_error), equilibrium_error = $(result.equilibrium_error)")
    push!(ndof_results, (N1, N2, result))
end

write_ndof_results(joinpath(results_dir, "profile_ndofs.csv"), ndof_results)

ndofs = [result.nprimal for (_, _, result) in ndof_results]
ndof_solve_times = [result.solve_time for (_, _, result) in ndof_results]
ndof_iterations = [result.iterations for (_, _, result) in ndof_results]

ndof_plot = plot(
    ndofs,
    ndof_solve_times;
    marker = :circle,
    linewidth = 2,
    label = "CG solve",
    xlabel = "Primal DOFs",
    ylabel = "Solve time (s)",
    xscale = :log10,
    yscale = :log10,
    title = "Patch-test solve time vs. primal DOFs",
    legend = :topleft,
    dpi = 300,
)
savefig(ndof_plot, joinpath(results_dir, "solve_time_vs_ndofs.png"))

ndof_iterations_plot = plot(
    ndofs,
    ndof_iterations;
    marker = :circle,
    linewidth = 2,
    label = "CG iterations",
    xlabel = "Primal DOFs",
    ylabel = "Iterations",
    xscale = :log10,
    title = "Patch-test iterations vs. primal DOFs",
    dpi = 300,
)
savefig(ndof_iterations_plot, joinpath(results_dir, "iterations_vs_ndofs.png"))

ncluster_results = Tuple{Int,NamedTuple}[]
for n_clusters in NPATCH_CASES
    result = run_patch_test(FIXED_N1, FIXED_N2, n_clusters)
    push!(ncluster_results, (n_clusters, result))
end

write_ncluster_results(joinpath(results_dir, "profile_nclusters.csv"), ncluster_results)

numclusters = [n_clusters for (n_clusters, _) in ncluster_results]
ncluster_solve_times = [result.solve_time for (_, result) in ncluster_results]
ncluster_iterations = [result.iterations for (_, result) in ncluster_results]

ncluster_plot = plot(
    numclusters,
    ncluster_solve_times;
    marker = :circle,
    linewidth = 2,
    label = "CG solve",
    xlabel = "CoNC clusters",
    ylabel = "Solve time (s)",
    xscale = :log10,
    yscale = :log10,
    title = "Patch-test solve time vs. CoNC clusters",
    legend = :topleft,
    dpi = 300,
)
savefig(ncluster_plot, joinpath(results_dir, "solve_time_vs_nclusters.png"))

ncluster_iterations_plot = plot(
    numclusters,
    ncluster_iterations;
    marker = :circle,
    linewidth = 2,
    label = "CG iterations",
    xlabel = "CoNC clusters",
    ylabel = "Iterations",
    xscale = :log10,
    title = "Patch-test iterations vs. CoNC clusters",
    dpi = 300,
)
savefig(ncluster_iterations_plot, joinpath(results_dir, "iterations_vs_nclusters.png"))
