using FinEtools
using FinEtools.AlgoBaseModule: solve_blocked!, matrix_blocked, vector_blocked
using FinEtoolsHeatDiff
using FinEtoolsHeatDiff.AlgoHeatDiffModule
using FinEtools.MeshExportModule.VTK: vtkexportmesh, T3, vtkexportvectors
using LinearAlgebra
using FinEtoolsDeforLinear
using FinEtoolsDeforLinear.AlgoDeforLinearModule
using Infiltrator
include("AL_solver.jl")

function run(r=0, savevtk = false)
    
    N_elem1 = 6*2^r
    N_elem2 = 8*2^r
    N_elem3 = 4*2^r
    N_elem_i = 6*2^r
    N_elem_i2 =3*2^r
    left_m = "t"
    right_m = "t"
    skew = 0.
    lam_order = 1
    bend=1.

    function bend_map(xyz; hinge_x = 0.35, angle_deg = 120.0)
        X0 = copy(xyz)

        theta = deg2rad(angle_deg)
        delta = 0.5 / tan(theta / 2.0)

        for a in axes(X0, 1)
            x = X0[a, 1]
            y = X0[a, 2]

            # Desired position of the original vertical skeleton x = 0.5.
            #
            # y = 0.5  -> x_interface = hinge_x
            # y = 0, 1 -> x_interface = hinge_x + delta
            x_interface = hinge_x + 2.0 * delta * abs(y - 0.5)

            # Map each horizontal line piecewise linearly.
            # This keeps x=0 and x=1 fixed.
            if x <= 0.5
                xyz[a, 1] = 2.0 * x * x_interface
            else
                xyz[a, 1] =
                    x_interface +
                    2.0 * (x - 0.5) * (1.0 - x_interface)
            end

            # Keep horizontal exterior boundaries flat.
            xyz[a, 2] = y
            xyz[a, 3] = X0[a, 3]
        end

        return xyz
    end
    E = 1.0
    nu = 1/3
    MR = DeforModelRed3D
    material =  MatDeforElastIso(MR, 1.0, E, nu, 0.0)

    t0 = 1.0e-3
    function u_exact(X)
        return [
            -(t0 / E) * (X[1] - 1.0),
            (nu * t0 / E) * X[2],
            (nu * t0 / E) * X[3],
        ]
    end

    #########################################################################################
    width1 = 0.5
    height1 = 0.5
    depth1 = 1.0
    if left_m == "h"
        fens1, fes1 = H8block(width1, height1, depth1, floor(Int, N_elem1), N_elem1, N_elem1)
        Rule1 = GaussRule(3,2)
    else
        fens1, fes1 = T4block(width1, height1, depth1, floor(Int, N_elem1/2), N_elem1, N_elem1)
        Rule1 = TetRule(4)
    end
    # fens1.xyz[:,2] .+= 0.25
    # fens1.xyz[:,3] .+= 0.25


    boundaryfes1 = meshboundary(fes1)
    edge_fes1 = subset(boundaryfes1, selectelem(fens1, boundaryfes1,  box=[width1,width1, 0.0,1.0, 0.0, 1.0], inflate=1e-8))

    edge_fes1h = subset(boundaryfes1, selectelem(fens1, boundaryfes1,  box=[0.0,0.5, 0.5,0.5, 0.0, 1.0], inflate=1e-8))
    fens1.xyz = bend_map(fens1.xyz)

    geom1 = NodalField(fens1.xyz)
    u1 = NodalField(zeros(size(fens1.xyz, 1), 3)) # displacement field

    applyebc!(u1)
    numberdofs!(u1)
    femm1 = FEMMDeforLinear(MR, IntegDomain(fes1, Rule1), material)
    K1 = stiffness(femm1, geom1, u1)
    K1_ff = matrix_blocked(K1, nfreedofs(u1), nfreedofs(u1))[:ff]

    left_elem = subset(boundaryfes1, selectelem(fens1, boundaryfes1, box=[0.0, 0.0, 0.0, 1.0, 0.0, 1.0], inflate=1e-8))
    fi_1 = ForceIntensity(Float64[t0, 0.0, 0.0])

    el1femm = FEMMBase(IntegDomain(left_elem, TriRule(3)))
    F1 = distribloads(el1femm, geom1, u1, fi_1, 2)
    F1_ff = vector_blocked(F1, nfreedofs(u1))[:f]

    ###########################################################################################


    if left_m == "h"
        fens1, fes1 = H8block(width1, height1, depth1, floor(Int, N_elem1), N_elem1, N_elem1)
        Rule1 = GaussRule(3,2)
    else
        fens3, fes3 = T4block(width1, height1, depth1, floor(Int, N_elem3/2), N_elem3, N_elem3)
        Rule3 = TetRule(4)
    end
    fens3.xyz[:,2] .+= 0.5


    boundaryfes3 = meshboundary(fes3)
    edge_fes3 = subset(boundaryfes3, selectelem(fens3, boundaryfes3,  box=[width1,width1, 0.0,1.0, 0.0, 1.0], inflate=1e-8))

    edge_fes3h = subset(boundaryfes3, selectelem(fens3, boundaryfes3,  box=[0.0,0.5, 0.5,0.5, 0.0, 1.0], inflate=1e-8))
    fens3.xyz = bend_map(fens3.xyz)


    geom3 = NodalField(fens3.xyz)
    u3 = NodalField(zeros(size(fens3.xyz, 1), 3)) # displacement field

    applyebc!(u3)
    numberdofs!(u3)
    femm3 = FEMMDeforLinear(MR, IntegDomain(fes3, Rule3), material)
    K3 = stiffness(femm3, geom3, u3)
    K3_ff = matrix_blocked(K3, nfreedofs(u3), nfreedofs(u3))[:ff]

    left_elem = subset(boundaryfes3, selectelem(fens3, boundaryfes3, box=[0.0, 0.0, 0.0, 1.0, 0.0, 1.0], inflate=1e-8))
    fi_3 = ForceIntensity(Float64[t0, 0.0, 0.0])

    el1femm = FEMMBase(IntegDomain(left_elem, TriRule(3)))
    F3 = distribloads(el1femm, geom3, u3, fi_3, 2)
    F3_ff = vector_blocked(F3, nfreedofs(u3))[:f]

    ##########################################################################################
    width2 = 0.5
    height2 = 1.0
    depth2 = 1.0
    if right_m == "h"
        fens2, fes2 = H8block(width2, height2, depth2, floor(Int, N_elem2), N_elem2, N_elem2)
        Rule2 = GaussRule(3,2)
    else
        fens2, fes2 = T4block(width2, height2, depth2, floor(Int, N_elem2), N_elem2, N_elem2)
        Rule2 = TetRule(4)
        fens2.xyz[:,1] .+= 0.5
    end 


    boundaryfes2 = meshboundary(fes2)
    edge_fes2 = subset(boundaryfes2, selectelem(fens2, boundaryfes2,  box=[0.5,0.5, 0.0,height2, 0.0, depth2], inflate=1e-8))
    fens2.xyz = bend_map(fens2.xyz)
    # fens2.xyz[:, 1] .+= bend * (1.0 .-fens2.xyz[:, 1]).*(fens2.xyz[:, 2] .- 0.5).^2

    geom2 = NodalField(fens2.xyz)
    u2 = NodalField(zeros(size(fens2.xyz, 1), 3)) # displacement field

    box2_1 = [1.0,1.0,0.0,0.0,0.0,0.0]
    dbc_nodes2_1 = selectnode(fens2; box=box2_1, inflate=1e-8)
    setebc!(u2, dbc_nodes2_1, 1, 0.0)
    setebc!(u2, dbc_nodes2_1, 2, 0.0)
    setebc!(u2, dbc_nodes2_1, 3, 0.0)

    box2_2 = [1.0,1.0,1.0,1.0,0.0,0.0]
    dbc_nodes2_2 = selectnode(fens2; box=box2_2, inflate=1e-8)
    setebc!(u2, dbc_nodes2_2, 1, 0.0)
    # setebc!(u2, [dbc_nodes2_2], 2, 0.0)
    setebc!(u2, dbc_nodes2_2, 3, 0.0)

    box2_3 = [1.0,1.0,0.0,0.0,1.0,1.0]
    dbc_nodes2_3 = selectnode(fens2; box=box2_3, inflate=1e-8)
    setebc!(u2, dbc_nodes2_3, 1, 0.0)
    applyebc!(u2)
    numberdofs!(u2)

    femm2 = FEMMDeforLinear(MR, IntegDomain(fes2, Rule2), material)
    K2 = stiffness(femm2, geom2, u2)
    K2_ff = matrix_blocked(K2, nfreedofs(u2), nfreedofs(u2))[:ff]

    l2_right = selectelem(fens2, meshboundary(fes2), box = [1.0,1.0, 0.0,1.0, 0.0, 1.0], inflate=1e-8)
    el2femm_right = FEMMBase(IntegDomain(subset(meshboundary(fes2), l2_right), TriRule(3)))
    fi2_right = ForceIntensity(Float64[-t0, 0.0, 0.0])
    F2 = distribloads(el2femm_right, geom2, u2, fi2_right, 2) 
    F2_ff = vector_blocked(F2, nfreedofs(u2))[:f]

    ##########################################################################################

    xs_i1 = 0.5
    ys_i1 = collect(linearspace(0.0, 1.0, N_elem_i+1))
    zs_i1 = collect(linearspace(0.0, 1.0, N_elem_i+1))
    fens_i1, fes_i1 = T3blockx(ys_i1, zs_i1, :a)
    fens_i1.xyz = hcat(xs_i1*ones(size(fens_i1.xyz, 1), 1), fens_i1.xyz)
    fens_i1.xyz = bend_map(fens_i1.xyz)
    # fens_i1.xyz[:, 1] .+= bend * fens_i1.xyz[:, 1].*(fens_i1.xyz[:, 2] .- 0.5).^2

    if lam_order==1
        u_i1  = NodalField(zeros(size(fens_i1.xyz, 1), 3))
    else
        u_i1  = ElementalField(zeros(size(fes_i1.conn, 1), 3))
    end

    femm_i1 = FEMMDeforLinear(MR, IntegDomain(fes_i1, TriRule(3)), material)
    geom_i1 = NodalField(fens_i1.xyz)
    applyebc!(u_i1)
    numberdofs!(u_i1)

    m_i1 = mass(femm_i1, geom_i1, u_i1)




    ys_i2 = 0.5
    xs_i2 = collect(linearspace(0.0, 0.5, N_elem_i2+1))
    zs_i2 = collect(linearspace(0.0, 1.0, N_elem_i2+1))
    fens_i2, fes_i2 = T3blockx(xs_i2, zs_i2, :a)
    fens_i2.xyz = hcat(ys_i2*ones(size(fens_i2.xyz, 1), 1), fens_i2.xyz)

    fens_i2.xyz[:,[1,2,3]] = fens_i2.xyz[:,[2,1,3]]
    # fens_i2.xyz[:, 1] .+= bend * fens_i2.xyz[:, 1].*(fens_i2.xyz[:, 2] .- 0.5).^2

    fens_i2.xyz = bend_map(fens_i2.xyz)
    if lam_order==1
        u_i2  = NodalField(zeros(size(fens_i2.xyz, 1), 3))
    else
        u_i2  = ElementalField(zeros(size(fes_i2.conn, 1), 3))
    end

    femm_i2 = FEMMDeforLinear(MR, IntegDomain(fes_i2, TriRule(3)), material)
    geom_i2 = NodalField(fens_i2.xyz)
    applyebc!(u_i2)

    numberdofs!(u_i2)
    m_i2 = mass(femm_i2, geom_i2, u_i2)


    D11, _ = common_refinement(fens1, edge_fes1, fens_i1, fes_i1; lam_order=lam_order, h=0.03, dim_u=3, tri_order=1)
    D21, _ = common_refinement(fens2, edge_fes2, fens_i1, fes_i1; lam_order=lam_order, h=0.033, dim_u=3, tri_order=1)
    D31, _ = common_refinement(fens3, edge_fes3, fens_i1, fes_i1; lam_order=lam_order, h=0.033, dim_u=3, tri_order=1)

    D12, _ = common_refinement(fens1, edge_fes1h, fens_i2, fes_i2; lam_order=lam_order, h=0.03, dim_u=3, tri_order=1)
    D32, _ = common_refinement(fens3, edge_fes3h, fens_i2, fes_i2; lam_order=lam_order, h=0.033, dim_u=3, tri_order=1)


    # free_dofs2 = setdiff(1:(3 * count(fens2)), dbc_dofs2)
    # D2_ff = D2[:, free_dofs2]

    # mu_plus = zeros(size(D2_ff, 1))
    # mu_plus[1:3:end] .= t0

    # F2_ff .-= transpose(D2_ff) * mu_plus
    # @infiltrate

    dbc_dofs2 = sort([
            3*dbc_nodes2_1 .- 2;  # ux
            3*dbc_nodes2_1 .- 1;  # uy
            3*dbc_nodes2_1;       # uz

            3*dbc_nodes2_2 .- 2;  # ux
            3*dbc_nodes2_2;       # uz

            3*dbc_nodes2_3 .- 2        # ux
        ])


    D21 = D21[:, setdiff(1:3*count(fens2), dbc_dofs2)]



    us, lambdas, X, stats = AL_solve(
        [K1_ff, K2_ff, K3_ff],
        [F1_ff, F2_ff, F3_ff],
            [
            [(1, +1.0, D11), (2, -1.0, D21),(3, 1.0, D31)],
            [(1, +1.0, D12), (3, -1.0, D32)]
        ],
        [m_i1, m_i2];
        gamma = 1.0,
        tau = 1e-3,
        rtol = 1e-8,
        atol = 1e-10,
        itmax = 5000
    )

    scattersysvec!(u1, us[1])
    scattersysvec!(u2, us[2])
    scattersysvec!(u3, us[3])

    # sol(x) = x[1]-1
    err1 = L2error(femm1, geom1, u1, u_exact)
    err2 = L2error(femm2, geom2, u2, u_exact)
    err3 = L2error(femm3, geom3, u3, u_exact)

    filename = basename(@__FILE__)
    if !isdir(filename)
        mkdir(filename)
    else
        for file in readdir(filename)
            rm(joinpath(filename, file))
        end
    end
    s1 = elemfieldfromintegpoints(femm1, geom1, u1, :Cauchy, 1)
    s2 = elemfieldfromintegpoints(femm2, geom2, u2, :Cauchy, 1)
    s3 = elemfieldfromintegpoints(femm3, geom3, u3, :Cauchy, 1)
    File1 = "$filename/mesh_left_down.vtk"
    vtkexportmesh(
        File1,
        fens1, fes1,vectors = [("Displacement", u1.values)],
        scalars = [("Stress", s1.values), ("Error", err1.values)]

    )
    File2 = "$filename/mesh_right.vtk"
    vtkexportmesh(
        File2,
        fens2, fes2,vectors = [("Displacement", u2.values)],
        scalars = [("Stress", s2.values), ("Error", err2.values)]
    )

    File3 = "$filename/mesh_left_up.vtk"
    vtkexportmesh(
        File3,
        fens3, fes3,vectors = [("Displacement", u3.values)],
        scalars = [("Stress", s3.values), ("Error", err3.values)]
    )

    # u1_fens = meta1["fens_u"]
    # u2_fens = meta2["fens_u"]
    # u1_fes = meta1["fes_u"]
    # u2_fes = meta2["fes_u"]
    # file3 = "$filename/pt_union_left.vtk"
    # vtkexportmesh(
    #     file3,
    #     u1_fens, u1_fes,scalars = []
    # )
    # file4 = "$filename/pt_union_right.vtk"
    # vtkexportmesh(
    #     file4,
    #     u2_fens, u2_fes,scalars = []
    # )
    # println(u_i.values)
end

