using FinEtools
using FinEtoolsDeforLinear
using FinEtoolsDeforLinear.AlgoDeforLinearModule
using FinEtools.MeshExportModule
using LinearAlgebra
using FinEtools.AlgoBaseModule: matrix_blocked, vector_blocked
using SparseArrays
using Krylov

function elasticity_rbm_3d(xyz::Matrix{Float64})
    nnode = size(xyz, 1)
    ndof = 3 * nnode

    xc = vec(sum(xyz, dims=1)) ./ nnode

    R = zeros(ndof, 6)

    for a in 1:nnode
        x = xyz[a, 1] - xc[1]
        y = xyz[a, 2] - xc[2]
        z = xyz[a, 3] - xc[3]

        ix = 3a - 2
        iy = 3a - 1
        iz = 3a

        # translations
        R[ix, 1] = 1.0
        R[iy, 2] = 1.0
        R[iz, 3] = 1.0

        # rotation about x-axis: omega = (1,0,0), u = omega x r
        R[iy, 4] = -z
        R[iz, 4] =  y

        # rotation about y-axis: omega = (0,1,0)
        R[ix, 5] =  z
        R[iz, 5] = -x

        # rotation about z-axis: omega = (0,0,1)
        R[ix, 6] = -y
        R[iy, 6] =  x
    end

    return R
end

function independent_columns(R; tol=1e-10)
    F = qr(R, ColumnNorm())
    r = sum(abs.(diag(F.R)) .> tol * maximum(abs.(diag(F.R))))
    return Matrix(R[:, F.p[1:r]])
end

function make_floating_solver(K::SparseMatrixCSC, R::Matrix{Float64})
    n = size(K, 1)
    m = size(R, 2)

    if m == 0
        F = cholesky(Symmetric(K))
        return rhs -> F \ rhs
    end

    Aaug = [
        K           sparse(R)
        sparse(R')  spzeros(m, m)
    ]

    Faug = lu(Aaug)

    function solve(rhs)
        # sol,_ = krylov_solve(Val(:minres), Aaug, vec(vcat(rhs, zeros(m))), rtol=1e-8)
        sol = Faug \ vcat(rhs, zeros(m))
        return sol[1:n]
    end

    return solve
end

# function run_stress_concentration(r, m=2)
    r = 2
    m=2

    println("Running with r = $r")
    mult = floor(Int,m^r)
    N_elem1 = 2 * mult
    N_elem2 = 2 * mult
    N_elem3 = 2 * mult
    N_elem_i = 2 * mult
    lam_order = 0
    depth = 0.3

    E = 1.0
    nu = 1/3
    MR = DeforModelRed3D
    material = MatDeforElastIso(MR, 0.0, E, nu, 0.0)
    trule3d = TetRule(4)
    trule2d = TriRule(3)
    grule3d = GaussRule(3, 4)
    grule2d = GaussRule(2, 3)

    # interface left ####################################################################################
    fensi1_b, fesi1_b = T3block(0.15,0.3, N_elem_i, N_elem_i)
    fensi1_b.xyz = hcat(fensi1_b.xyz[:, 1].+0.15, 0.3*ones(size(fensi1_b.xyz, 1)), fensi1_b.xyz[:, 2])

    fensi1_t, fesi1_t = T3block(0.15,0.3, N_elem_i, N_elem_i)
    fensi1_t.xyz = hcat(fensi1_t.xyz[:, 1].+0.15, 0.7*ones(size(fensi1_t.xyz, 1)), fensi1_t.xyz[:, 2])

    fensi1_c, fesi1_c = T3block(0.3,0.4, N_elem_i, N_elem_i)
    fensi1_c.xyz = hcat(0.15*ones(size(fensi1_c.xyz, 1)), fensi1_c.xyz[:, 2].+0.3, fensi1_c.xyz[:, 1])

    fensi1, fesi1out = mergenmeshes(Tuple{FENodeSet, AbstractFESet}[
                                                                    (fensi1_b, fesi1_b),
                                                                    (fensi1_t, fesi1_t),
                                                                    (fensi1_c, fesi1_c),
                                                                    ], 1e-8)

    fesi1 = FESetT3(vcat(connasarray.(fesi1out)...))
    # interface right ####################################################################################
    fensi2, fesi2 = T3block(0.3,0.2, N_elem_i, N_elem_i)
    fensi2.xyz = hcat(0.4*ones(size(fensi2.xyz, 1)), fensi2.xyz[:, 2].+0.4, fensi2.xyz[:, 1])

    # left ####################################################################################

    fens1b, fes1b = T3block(0.3,0.3,2*N_elem1, 2*N_elem1)
    fens1t, fes1t = T3block(0.3,0.3,2*N_elem1, 2*N_elem1)
    fens1t.xyz[:,2] .+= 0.7

    fens1c, fes1c = T3block(0.15,0.4, N_elem1, ceil(Int, 2.66*N_elem1))
    fens1c.xyz[:,2] .+= 0.3

    fens1_2d, output_fes1 = mergenmeshes(Tuple{FENodeSet, AbstractFESet}[
                                                                    (fens1b, fes1b),
                                                                    (fens1c, fes1c),
                                                                    (fens1t, fes1t),
                                                                    ], 1e-8)

    fes1_2d = FESetT3(vcat(connasarray.(output_fes1)...))
    fens1, fes1 = T4extrudeT3(fens1_2d, fes1_2d, 2*N_elem1, (x, k) -> [x[1], x[2], k * depth / N_elem1/2])
    geom1 = NodalField(fens1.xyz)
    u1 = NodalField(zeros(size(fens1.xyz, 1), 3))

    dbc_node1 = selectnode(fens1, box=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], inflate=1e-8)
    setebc!(u1, dbc_node1, 1, 0.0)
    setebc!(u1, dbc_node1, 2, 0.0)
    setebc!(u1, dbc_node1, 3, 0.0)
    dbc_node2 = selectnode(fens1, box=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0], inflate=1e-8)
    setebc!(u1, dbc_node2, 1, 0.0)
    # setebc!(u1, dbc_node2, 3, 0.0)
    dbc_node3 = selectnode(fens1, box=[0.0, 0.0, 0.0, 0.0, 0.3, 0.3], inflate=1e-8)
    setebc!(u1, dbc_node3, 1, 0.0)
    setebc!(u1, dbc_node3, 2, 0.0)

    dbc_nodes = sort([dbc_node1; dbc_node2; dbc_node3])
    applyebc!(u1)
    numberdofs!(u1)
    femm1 = FEMMDeforLinear(MR, IntegDomain(fes1, trule3d), material)
    K1 = stiffness(femm1, geom1, u1)
    K1_ff = matrix_blocked(K1, nfreedofs(u1), nfreedofs(u1))[:ff]
    K1_fd = matrix_blocked(K1, nfreedofs(u1), nfreedofs(u1))[:fd]

    mb1 = meshboundary(fes1)
    left_fes1 = subset(mb1, selectelem(fens1, mb1, box=[0.0, 0.0, 0.0, 1.0, 0.0, 1.0], inflate=1e-8))
    left_femm1 = FEMMBase(IntegDomain(left_fes1, trule2d))
    function fi_fun_l(forceout::Vector{T}, XYZ, tangents, feid, qpid) where {T}
        return [-1.0e-2, 0.0, 0.0]
    end
    fi_l = ForceIntensity(Float64, 3, fi_fun_l)
    F1 = distribloads(left_femm1, geom1, u1, fi_l,2)
    F1_ff = vector_blocked(F1, nfreedofs(u1))[:f] - K1_fd*gathersysvec(u1, :d)


    ifes1b = subset(mb1, selectelem(fens1, mb1, box=[0.15, 0.3, 0.3, 0.3, 0, 1], inflate=1e-8))
    ifes1t = subset(mb1, selectelem(fens1, mb1, box=[0.15, 0.3, 0.7, 0.7, 0, 1], inflate=1e-8))
    ifes1c = subset(mb1, selectelem(fens1, mb1, box=[0.15, 0.15, 0.3, 0.7, 0, 1], inflate=1e-8))
    ifes1 = FESetT3(vcat(connasarray(ifes1b), connasarray(ifes1c), connasarray(ifes1t)))

    D11,_ = common_refinement(fens1, ifes1, fensi1, fesi1; lam_order=lam_order, h=0.05, dim_u=3)
    dbc_dofs = sort([3*dbc_node1 .- 2; 3*dbc_node1 .- 1; 3*dbc_node1;
                        3*dbc_node2 .- 2;
                        3*dbc_node3; 3*dbc_node3.-1 ])
    D11 = D11[:,setdiff(1:3*count(fens1), dbc_dofs)]


    # central meshing ################################################################################

    N2_inc = ceil(Int, N_elem2*2)
    # renumb = (c) -> c[[1, 4, 3, 2]]

    fens2tt, fes2tt = Q4block(0.15,0.05, N2_inc, ceil(Int, N2_inc*0.5))
    fens2bb, fes2bb = Q4block(0.15,0.05, N2_inc, ceil(Int, N2_inc*0.5))

    fens_wh, fes_wh = Q4elliphole(0.05, 0.05, 0.2, 0.15, N_elem2, N_elem2, N2_inc)
    fens2b,fes2b = mirrormesh(fens_wh, fes_wh, [1.0,0.0], [0.1,0.1], renumb = (c) -> c[[1, 4, 3, 2]])

    fens2t,fes2t = mirrormesh(fens2b, fes2b, [0.0,1.0], [0.15,0.15], renumb = (c) -> c[[1, 4, 3, 2]])
    fens2bb.xyz[:, 1] .+= 0.15
    fens2bb.xyz[:, 2] .+= 0.3
    fens2tt.xyz[:, 1] .+= 0.15
    fens2tt.xyz[:, 2] .+= 0.65

    fens2b.xyz[:, 1] .+= 0.15
    fens2b.xyz[:, 2] .+= 0.35
    fens2t.xyz[:, 1] .+= 0.15
    fens2t.xyz[:, 2] .+= 0.35

    fens2r, fes2r = Q4block(0.05, 0.2, ceil(Int, N_elem2*0.75*0.8), 2*N2_inc)
    fens2r.xyz[:, 1] .+= 0.35
    fens2r.xyz[:, 2] .+= 0.4
    fens2_2d, output_fes2 = mergenmeshes(Tuple{FENodeSet, AbstractFESet}[
                                                                    (fens2b, fes2b),
                                                                    (fens2t, fes2t),
                                                                    (fens2r, fes2r),
                                                                    (fens2bb, fes2bb),
                                                                    (fens2tt, fes2tt),
                                                                    ], 1e-8)
    fes2_2d = FESetQ4(vcat(connasarray.(output_fes2)...))

    N_elem2_depth = ceil(Int, 2*N2_inc*1.5)
    fens2, fes2 = H8extrudeQ4(fens2_2d, fes2_2d, N_elem2_depth, (x, k) -> [x[1], x[2], k * depth / N_elem2_depth])
    geom2 = NodalField(fens2.xyz)
    u2 = NodalField(zeros(size(fens2.xyz, 1), 3))
    femm2 = FEMMDeforLinear(MR, IntegDomain(fes2, grule3d), material)
    numberdofs!(u2)
    K2 = stiffness(femm2, geom2, u2)
    K2_ff = matrix_blocked(K2, nfreedofs(u2), nfreedofs(u2))[:ff]
    F2 = zeros((nfreedofs(u2),1))


    mb2  = meshboundary(fes2)
    ifes21c = subset(mb2, selectelem(fens2, mb2, box=[0.15, 0.15, 0.3, 0.7, 0., 1.], inflate=1e-8))
    ifes21b  = subset(mb2, selectelem(fens2, mb2, box=[0.15, 0.3, 0.3, 0.3, 0., 1.], inflate=1e-8))
    ifes21t  = subset(mb2, selectelem(fens2, mb2, box=[0.15, 0.3, 0.7, 0.7, 0., 1.], inflate=1e-8))
    ifes21 = FESetQ4(vcat(connasarray(ifes21b), connasarray(ifes21c), connasarray(ifes21t)))
    D21,_ = common_refinement(fens2, ifes21, fensi1, fesi1; lam_order=lam_order, h=0.05, dim_u=3, tri_order=2)

    ifes22 = subset(mb2, selectelem(fens2, mb2, box=[0.4, 0.4, 0.0, 1.0, 0., 1.], inflate=1e-8))
    D22,_ = common_refinement(fens2, ifes22, fensi2, fesi2; lam_order=lam_order, h=0.05, dim_u=3, tri_order=2)



    ## right meshing ##############################################################################
    N_elem3_depth = N_elem3
    N_elem3_height = ceil(Int, N_elem3_depth*0.666)
    fens3, fes3 = T4block(0.3, 0.2, depth, N_elem3, N_elem3_height, N_elem3_depth)
    fens3.xyz[:, 1] .+= 0.4
    fens3.xyz[:, 2] .+= 0.4

    geom3 = NodalField(fens3.xyz)
    u3 = NodalField(zeros(size(fens3.xyz, 1), 3))
    femm3 = FEMMDeforLinear(MR, IntegDomain(fes3, trule3d), material)
    numberdofs!(u3)
    K3 = stiffness(femm3, geom3, u3)
    K3_ff = matrix_blocked(K3, nfreedofs(u3), nfreedofs(u3))[:ff]

    mb3 = meshboundary(fes3)

    right_fes3 = subset(mb3, selectelem(fens3, mb3, box=[0.7, 0.7, 0.0, 1.0, 0.0, 1.0], inflate=1e-8))
    right_femm3 = FEMMBase(IntegDomain(right_fes3, trule2d))
    function fi_fun_r(forceout::Vector{T}, XYZ, tangents, feid, qpid) where {T}
        return [5.0e-2, 0.0, 0.0]
    end
    fi_r = ForceIntensity(Float64, 3, fi_fun_r)
    F3 = distribloads(right_femm3, geom3, u3, fi_r,2)
    F3_ff = vector_blocked(F3, nfreedofs(u3))[:f]

    ifes32 = subset(mb3, selectelem(fens3, mb3, box=[0.4, 0.4, 0.0, 1., 0., 1.], inflate=1e-8))
    D32,_ = common_refinement(fens3, ifes32, fensi2, fesi2; lam_order=lam_order, h=0.05, dim_u=3)
    # assembly and results ######################################################################################

    D = [D11 -D21 spzeros(size(D11,1),size(D32,2));
        spzeros(size(D22,1),size(D11,2)) -D22 D32;]
    # K = [K1_ff spzeros(size(K1_ff, 1), size(K2_ff, 2)) spzeros(size(K1_ff, 1), size(K3_ff, 2));
    #     spzeros(size(K2_ff, 1), size(K1_ff, 2)) K2_ff spzeros(size(K2_ff, 1), size(K3_ff, 2));
    #     spzeros(size(K3_ff, 1), size(K1_ff, 2)) spzeros(size(K3_ff, 1), size(K2_ff, 2)) K3_ff]
    B1 = [D11; spzeros(size(D22, 1), size(D11, 2))]
    B2 = [-D21; -D22]
    B3 = [spzeros(size(D11, 1), size(D32, 2)); D32]
    
    K = blockdiag(K1_ff, K2_ff, K3_ff) 
    b = vec([F1_ff; F2; F3_ff; zeros(size(D, 1), 1)])
    A = [K D'; 
            D 0*sparse(I, size(D, 1), size(D, 1))]

    R1 = elasticity_rbm_3d(fens1.xyz)
    R2 = elasticity_rbm_3d(fens2.xyz)
    R3 = elasticity_rbm_3d(fens3.xyz)
    R1 = zeros(size(R1, 1)-6,0)
    print("Made rigid body modes\n")
    K1_solve = make_floating_solver(K1_ff, R1)
    K2_solve = make_floating_solver(K2_ff, R2)
    K3_solve = make_floating_solver(K3_ff, R3)
    print("Made floating solvers\n")

    g = B1 * K1_solve(F1_ff) + B2 * K2_solve(F2) + B3 * K3_solve(F3_ff)
    G = hcat(B1* R1, B2*R2, B3*R3)
    e = vcat(R1'* F1_ff, R2'*F2, R3'*F3_ff)
    rhs_cond = vcat(g, e)
    print("Assembled rhs_cond\n")
    function S_mul(x)
    return B1 * K1_solve(B1' * x) +
           B2 * K2_solve(B2' * x) +
           B3 * K3_solve(B3' * x)
    end

    nl = size(B1, 1)

    S = zeros(nl, nl)

    for j in 1:nl
        ej = zeros(nl)
        ej[j] = 1.0
        S[:, j] .= S_mul(ej)
    end

    print("Assembled S\n")
    nc = size(G, 2)

    Acond = [
        sparse(S)    sparse(G)
        sparse(G')   spzeros(nc, nc)
    ]

    sol = Acond \ rhs_cond
    print("Solved for λ and β\n")
    λ = sol[1:nl]
    β = sol[nl+1:end]
    m1 = size(R1, 2)
m2 = size(R2, 2)
m3 = size(R3, 2)

β1 = β[1:m1]
β2 = β[m1+1:m1+m2]
β3 = β[m1+m2+1:m1+m2+m3]
u1f = K1_solve(F1_ff - B1' * λ) - R1 * β1
u2f = K2_solve(F2 - B2' * λ) - R2 * β2
u3f = K3_solve(F3_ff - B3' * λ) - R3 * β3
print("Solved for u1, u2, u3\n")

scattersysvec!(u1, u1f)
scattersysvec!(u2, u2f)
scattersysvec!(u3, u3f)    
# st1 = fieldfromintegpoints(femm1, geom1, u1,:Cauchy, 1)
    # st2 = fieldfromintegpoints(femm2, geom2, u2,:Cauchy, 1)
    # st3 = fieldfromintegpoints(femm3, geom3, u3,:Cauchy, 1)
    st1 = elemfieldfromintegpoints(femm1, geom1, u1,:vm, 1)
    st2 = elemfieldfromintegpoints(femm2, geom2, u2,:vm, 1)
    st3 = elemfieldfromintegpoints(femm3, geom3, u3,:vm, 1)

    println("max stress = ", maximum(st2.values))

    # # output #########################################################################################
    # filename = basename(@__FILE__)

    # # if !isdir(filename)
    # #     mkdir(filename)
    # # else
    # #     for file in readdir(filename)
    # #         rmr(joinpath(filename, file))
    # #     end
    # # end
    # if isdir("$filename/r$r")
    #     for file in readdir("$filename/r$r")
    #         rm(joinpath("$filename/r$r", file))
    #     end
    # else
    #     mkdir("$filename/r$r")
    # end

    # file_left = "$filename/r$r/left.vtk"
    # vtkexportmesh(
    #     file_left,
    #     fens1, fes1,
    #     scalars = [ ("St", st1.values)],
    #     vectors = [("Displacement", u1.values)]
    # )
    # file_right = "$filename/r$r/right.vtk"
    # vtkexportmesh(
    #     file_right,
    #     fens3, fes3,
    #     scalars = [ ("St", st3.values)],
    #     vectors = [("Displacement", u3.values)]
    # )
    # file_center = "$filename/r$r/center.vtk"
    # vtkexportmesh(
    #     file_center,
    #     fens2, fes2,
    #     scalars = [ ("St", st2.values)],
    #     vectors = [("Displacement", u2.values)]
    # )
    # file_li = "$filename/r$r/left_interface.vtk"
    # vtkexportmesh(
    #     file_li,
    #     fensi1, fesi1
    # )
    # file_ri = "$filename/r$r/right_interface.vtk"
    # vtkexportmesh(
    #     file_ri,
    #     fensi2, fesi2
    # )
    # return maximum(st2.values)

    # # file_11 = "$filename/union_11.vtk"
    # # vtkexportmesh(
    # #     file_11,
    # #     meta11["fens_u"], meta11["fes_u"]
    # # )
    # # file_21 = "$filename/union_21.vtk"
    # # vtkexportmesh(
    # #     file_21,
    # #     meta21["fens_u"], meta21["fes_u"]
    # # )
    # # file_22 = "$filename/union_22.vtk"
    # # vtkexportmesh(
    # #     file_22,
    # #     meta22["fens_u"], meta22["fes_u"]
    # # )
    # # file_32 = "$filename/union_32.vtk"
    # # vtkexportmesh(
    # #     file_32,
    # #     meta32["fens_u"], meta32["fes_u"]
    # # )

# end

# for r in 1:1
#     max_stress = run_stress_concentration(r,2)
#     push!(s, max_stress)
# end
# s = s[end-2:end]
# extrapolated_s = (s[2]^2 - s[3]*s[1])/(2*s[2] - s[1] - s[3])
# println("extrapolated s = ", extrapolated_s)