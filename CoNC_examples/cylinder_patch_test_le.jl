# Patch test for hydrostatic pressure in a cylinder wiht 5 sectors with CoNC interface constraints.

using FinEtools
using FinEtools.AlgoBaseModule: matrix_blocked, vector_blocked
using FinEtoolsDeforLinear
using FinEtoolsDeforLinear.AlgoDeforLinearModule
using FinEtools.MeshExportModule.VTK: vtkexportmesh
using LinearAlgebra
using SparseArrays
using Statistics
using CoNCMOR: CoNCData, transfmatrix, LegendreBasis
using Krylov
using LinearOperators
using AlgebraicMultigrid

p = 1.0
E = 1.0
nu = 1 / 3
MR = DeforModelRed3D
material = MatDeforElastIso(MR, 1.0, E, nu, 0.0)

sector_angle = 2pi / 5
inner_radius = 0.5
depth = 0.5
lam_order = 1

alpha = (1.0 - 2.0 * nu) * p / E
exact_displacement(x) = alpha .* [x[1] - 0.5, x[2] - 0.5, x[3]]

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
    normal = epsol.vectors[:, 3]
    e1 = epsol.vectors[:, 1]
    e2 = epsol.vectors[:, 2]
    lxy = fill(0.0, length(list), 2)
    for j in 1:size(lX, 1)
        lxy[j, 1] = dot(lX[j, :], e1)
        lxy[j, 2] = dot(lX[j, :], e2)
    end
    return lxy
end

function radial_interface_faces(fens, fes, theta)
    bfes = meshboundary(fes)
    ids_start = selectelem(fens,bfes;facing=true,direction=x -> [sin(theta), -cos(theta), 0.0],tolerance=0.1,)
    theta_end = theta + sector_angle
    ids_end = selectelem(fens,bfes;facing=true,direction=x -> [-sin(theta_end), cos(theta_end), 0.0],tolerance=0.1)
    return subset(bfes, ids_start), subset(bfes, ids_end)
end

function radial_out(x)
    dx = x[1] - 0.5
    dy = x[2] - 0.5
    r = hypot(dx, dy)
    return [dx / r, dy / r, 0.0]
end

function hydrostatic_boundary_load(fens, bfes, geom, u, sigma0)
    outer = subset(bfes, selectelem(fens, bfes; facing=true, direction=radial_out, tolerance=0.1))
    bottom = subset(bfes, selectelem(fens, bfes; box=[0.0, 1.0, 0.0, 1.0, 0.0, 0.0], inflate=1e-8))
    top = subset(bfes, selectelem(fens, bfes; box=[0.0, 1.0, 0.0, 1.0, depth, depth], inflate=1e-8))
    center = [0.5, 0.5, depth / 2]

    function traction(forceout::Vector{T}, XYZ, tangents, feid, qpid) where {T}
        n = cross(vec(tangents[:, 1]), vec(tangents[:, 2]))
        n ./= norm(n)
        if dot(n, vec(XYZ) - center) < 0.0
            n .*= -1.0
        end
        forceout .= sigma0 .* n
        return forceout
    end

    fi = ForceIntensity(Float64, 3, traction)
    return distribloads(FEMMBase(IntegDomain(outer, TriRule(3))), geom, u, fi, 2) +
           distribloads(FEMMBase(IntegDomain(bottom, TriRule(3))), geom, u, fi, 2) +
           distribloads(FEMMBase(IntegDomain(top, TriRule(3))), geom, u, fi, 2)
end

function make_sector(i, n; fix_rbm=false)
    theta = (i - 1) * sector_angle

    fens_c, fes_c = T3circleseg(sector_angle, inner_radius, n, n)

    x = copy(fens_c.xyz[:, 1])
    y = copy(fens_c.xyz[:, 2])

    fens_c.xyz[:, 1] .= cos(theta) .* x .- sin(theta) .* y .+ 0.5
    fens_c.xyz[:, 2] .= sin(theta) .* x .+ cos(theta) .* y .+ 0.5

    fens, fes = T4extrudeT3(fens_c, fes_c, n, (x, k) -> [x[1], x[2], k * depth / n])

    bfes = meshboundary(fes)
    geom = NodalField(fens.xyz)
    u = NodalField(zeros(size(fens.xyz)))

    fixedcols = Int[]

    if fix_rbm
        theta_end = theta + sector_angle

        pA = [0.5, 0.5, 0.0]
        pB = [0.5 + inner_radius * cos(theta), 0.5 + inner_radius * sin(theta), 0.0]
        pC = [0.5 + inner_radius * cos(theta_end), 0.5 + inner_radius * sin(theta_end), 0.0]

        nA = selectnode(fens; box=[pA[1], pA[1], pA[2], pA[2], 0.0, 0.0], inflate=1e-8)
        nB = selectnode(fens; box=[pB[1], pB[1], pB[2], pB[2], 0.0, 0.0], inflate=1e-8)
        nC = selectnode(fens; box=[pC[1], pC[1], pC[2], pC[2], 0.0, 0.0], inflate=1e-8)

        setebc!(u, nA, 1, 0.0)
        setebc!(u, nA, 2, 0.0)
        setebc!(u, nA, 3, 0.0)
        setebc!(u, nB, 2, 0.0)
        setebc!(u, nB, 3, 0.0)
        setebc!(u, nC, 3, 0.0)

        fixedcols = sort([3 * (nA[1] - 1) + 1, 3 * (nA[1] - 1) + 2, 3 * (nA[1] - 1) + 3, 3 * (nB[1] - 1) + 2, 3 * (nB[1] - 1) + 3, 3 * (nC[1] - 1) + 3])
    end

    applyebc!(u)
    numberdofs!(u)

    femm = FEMMDeforLinear(MR, IntegDomain(fes, TetRule(4)), material)

    K = stiffness(femm, geom, u)
    K = matrix_blocked(K, nfreedofs(u), nfreedofs(u))[:ff]

    F = hydrostatic_boundary_load(fens, bfes, geom, u, p)
    F = vector_blocked(F, nfreedofs(u))[:f]

    freecols = isempty(fixedcols) ? collect(1:(3 * count(fens))) : setdiff(1:(3 * count(fens)), fixedcols)

    ifs, ife = radial_interface_faces(fens, fes, theta)

    return (fens=fens, fes=fes, geom=geom, u=u, femm=femm, K=sparse(K), F=F, ifs=ifs, ife=ife, freecols=freecols, theta=theta)
end

function conc_interface(A, B, nclusters)
    _, meta = common_refinement(A.fens, A.ife, B.fens, B.ifs; lam_order=lam_order, tri_order=1, h=0.1, dim_u=3)

    fens_u = meta["fens_u"]
    fes_u = meta["fes_u"]

    partitioning = pointpartitioning(fens_u.xyz, nclusters)
    mor = CoNCData(list -> patch_coordinates(fens_u.xyz, list), partitioning)

    lam = NodalField(zeros(size(fens_u.xyz)))
    numberdofs!(lam)

    Phi = sparse(transfmatrix(mor, LegendreBasis, 2, lam))

    DA = sparse(Phi' * meta["M"] * meta["PiA"])
    DB = sparse(Phi' * meta["M"] * meta["PiB"])

    DA = DA[:, A.freecols]
    DB = DB[:, B.freecols]

    return (DA=DA, DB=DB, Phi=Phi, lam=lam, fens=fens_u, fes=fes_u, partitioning=partitioning)
end

# make a whole row of global D matrix with proper offsets (eg: [0 0 D34 D43 0])
function embed_interface(C, i, j, offsets, n)
    ia, ja, va = findnz(C.DA)
    ib, jb, vb = findnz(C.DB)

    return sparse(vcat(ia, ib), vcat(ja .+ offsets[i], jb .+ offsets[j]), vcat(va, -vb), size(C.DA, 1), n)
end


function run_hs_frag_cyl_conc(
    ref=0,
    nclusters=2^ref,
    savevtk=false,
)
    nelems = [5, 4, 5, 4, 3] .* 2^ref

    sd = [
        make_sector(1, nelems[1]; fix_rbm=true),
        make_sector(2, nelems[2]),
        make_sector(3, nelems[3]),
        make_sector(4, nelems[4]),
        make_sector(5, nelems[5]),
    ]

    pairs = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 1),
    ]

    CR = [conc_interface(sd[i], sd[j], nclusters) for (i, j) in pairs]

    ns = [size(s.K, 1) for s in sd]
    offsets = cumsum(vcat(0, ns))[1:end-1]
    ranges = [offsets[i] + 1:offsets[i] + ns[i] for i in 1:5]

    n = sum(ns)
    K = blockdiag([s.K for s in sd]...)
    F = vcat([s.F for s in sd]...)

    Dblocks = [embed_interface(CR[k], pairs[k][1], pairs[k][2], offsets, n) for k in 1:5]

    D = sparse(vcat(Dblocks...))
    Dt = transpose(D)

    println("Primal DOFs = ", n)

    for k in 1:5
        i, j = pairs[k]
        println("Interface $i-$j: fine LM = ", size(CR[k].Phi, 1), ", CoNC LM = ", size(CR[k].Phi, 2), ", clusters = ", length(unique(CR[k].partitioning)))
    end

    r = rank(D)
    # --------------------------------------------------------------
    # Constraint projector
    # P = I - D' * (D*D')^+ * D
    # --------------------------------------------------------------

    G = Symmetric(Matrix(D * Dt))
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
    # --------------------------------------------------------------
    # SPD null-space operator
    # A = P*K*P + I-P
    # --------------------------------------------------------------

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

    # AMG


    C = sparse(Dt * D)

    dk = abs.(diag(K))
    dc = abs.(diag(C))
    ids = findall(dc .> 100eps(Float64) * maximum(dc))

    gamma = median(dk[ids] ./ dc[ids])
    println("Augmentation gamma = ", gamma)

    Kaug = sparse(K + gamma * C)

    amg_setup_time = @elapsed begin
        ml = smoothed_aggregation(Kaug)
        AMG = aspreconditioner(ml)
    end

    println("AMG setup = ", amg_setup_time, " s")

    rproj = zeros(Float64, n)
    zlocal = zeros(Float64, n)

    function projected_amg_mul!(z, r)
        project_constraint!(rproj, r)
        ldiv!(zlocal, AMG, rproj)
        project_constraint!(z, zlocal)
        @. z += r - rproj
        return z
    end

    Mproj = LinearOperator(Float64, n, n, true, true, projected_amg_mul!)

    # --------------------------------------------------------------
    # CG
    # --------------------------------------------------------------

    u = zeros(Float64, n)
    stats = nothing

    solve_time = @elapsed begin
        u, stats = Krylov.cg(Aproj, Fp; rtol=1e-10, atol=1e-12, itmax=5000, verbose=1)
    end

    project_constraint!(u, u)

    # --------------------------------------------------------------
    # Minimum-norm reduced LM recovery
    # --------------------------------------------------------------

    rhs_lambda = D * (F - K * u)
    lambda = Gfac \ rhs_lambda

    constraint_error = norm(D * u)
    equilibrium_error = norm(K * u + Dt * lambda - F) / max(norm(F), eps(Float64))

    println()
    println("CG iterations = ", stats.niter)
    println("CG solve time = ", solve_time, " s")
    println("Constraint error = ", constraint_error)
    println("Equilibrium error = ", equilibrium_error)

    # --------------------------------------------------------------
    # Scatter primal solution
    # --------------------------------------------------------------

    for i in 1:5
        scattersysvec!(sd[i].u, u[ranges[i]])
    end

    errs = [L2error(sd[i].femm, sd[i].geom, sd[i].u, exact_displacement) for i in 1:5]

    println("Maximum errors")
    for i in 1:5
        println("SD$i: ", maximum(errs[i].values))
    end

    global_error = maximum(vcat([e.values for e in errs]...))

    total_error = sqrt(sum(sum(e.values .^ 2) for e in errs))

    println("Global: ", global_error)

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------

    if savevtk
        filename = basename(@__FILE__)
        filename = "results/" * filename[1:end-3]


        if !isdir(filename)
            mkdir(filename)
        else
            for file in readdir(filename)
                rm(joinpath(filename, file); recursive=true)
            end
        end

        for i in 1:5
            vtkexportmesh("$filename/$i.vtk", sd[i].fens, sd[i].fes; vectors=[("u", sd[i].u.values)], scalars=[("u_error", errs[i].values)])
        end

        ms = [size(C.DA, 1) for C in CR]
        loffsets = cumsum(vcat(0, ms))[1:end-1]

        for k in 1:5
            i, j = pairs[k]
            lr = loffsets[k] + 1:loffsets[k] + ms[k]

            lambda_fine = CR[k].Phi * lambda[lr]

            scattersysvec!(CR[k].lam, lambda_fine)

            vtkexportmesh("$filename/cr_$i$j.vtk", CR[k].fens, CR[k].fes; vectors=[("lambda", CR[k].lam.values)], scalars=[("cluster", CR[k].partitioning)])
        end
    end

    return (
        error=total_error,
        iterations=stats.niter,
        constraint_error=constraint_error,
        equilibrium_error=equilibrium_error,
        nprimal=n,
        nlambda=n_lam,
        rankD=r,
        amg_setup=amg_setup_time,
        solve_time=solve_time,
    )
end

result = run_hs_frag_cyl_conc(2, 2, true)

println(result)