using FinEtools
using FinEtoolsDeforLinear
using FinEtoolsDeforLinear.AlgoDeforLinearModule
using FinEtools.MeshExportModule
using LinearAlgebra
using FinEtools.AlgoBaseModule: matrix_blocked, vector_blocked
using SparseArrays
using Infiltrator
include("AL_solver.jl")

function run_hollow_shaft(mult=1, savevtk=false)
    N_elem1 = 2*mult
    N_elem2 = 8*mult
    lam_order = 1
    N_elemi = 8*mult

    E = 1.0
    nu = 1/3
    MR = DeforModelRed3D
    material1 = MatDeforElastIso(MR, 1.0, E, nu, 0.0)
    material2 = MatDeforElastIso(MR, 1.0, E, nu, 0.0)
    rule3d = TetRule(4)
    rule2d = TriRule(3)

    exact_u(x) = [0.0 0.0 0.0]

    xc = 0.5
    yc = 0.5
    print("Building meshes...\n")

    function radial_out(x)
        dx = x[1] - xc
        dy = x[2] - yc
        r = hypot(dx, dy)
        return [dx / r, dy / r, 0.0]
    end

    function radial_in(x)
        dx = x[1] - xc
        dy = x[2] - yc
        r = hypot(dx, dy)
        return [-dx / r, -dy / r, 0.0]
    end

    function L2circle(
        radius::Real,
        nCirc::Integer;
        center=(0.0, 0.0),
        z::Real=0.0,
        orientation::Symbol=:ccw,
    )
        @assert radius > 0.0
        @assert nCirc >= 3
        @assert orientation in (:ccw, :cw)

        xc, yc = center

        # Do not include 2π: the last element closes the circle.
        θ = collect(range(0.0, 2π; length=nCirc + 1))[1:end-1]

        if orientation == :cw
            θ = -θ
        end

        xyz = zeros(Float64, nCirc, 3)

        for i in 1:nCirc
            xyz[i, 1] = xc + radius * cos(θ[i])
            xyz[i, 2] = yc + radius * sin(θ[i])
            xyz[i, 3] = z
        end

        conn = zeros(Int, nCirc, 2)

        for i in 1:nCirc
            conn[i, 1] = i
            conn[i, 2] = mod1(i + 1, nCirc)
        end

        return FENodeSet(xyz), FESetL2(conn)
    end
    fens_ii, fes_ii = L2circle(0.25, N_elemi, center=(0.5, 0.5), z=0.0, orientation=:ccw)
    len = 0.5
    n_axial = 4*mult
    fens_i, fes_i = Q4extrudeL2(
            fens_ii,
            fes_ii,
            n_axial,
            (x, k) -> [
                x[1],
                x[2],
                k * len / n_axial,
            ],
        )
    fens_i.xyz[:, 3] .+= 0.25
    # @infiltrate

    # SD1 #########################################################################################################################
    ta = time()
    r1 = 0.25
    l1 = 1.0
    fens1, fes1 = T4cylindern(r1, l1, N_elem1, N_elem1*4)

    fens1.xyz[:, 1] .+= 0.5
    fens1.xyz[:, 2] .+= 0.5

    boundaryfes1 = meshboundary(fes1)
    # interface_fes1 = subset(boundaryfes1, selectelem(fens1, boundaryfes1, box=[0.25 ,0.75, 0.25, 0.75, 0, l1], inflate=1e-8))
    function notz(x)
        return [x[1], x[2], 0]
    end
    # interface_fes1_i = subset(boundaryfes1, selectelem(fens1, boundaryfes1, facing =true, direction =notz, tolerance=0.1))
    # interface_fes1 = subset(interface_fes1_i, selectelem(fens1, interface_fes1_i, box=[0.25 ,0.75, 0.25, 0.75, 0, l1], inflate=1e-8))

    interface_fes1_i = subset(boundaryfes1, selectelem(fens1, boundaryfes1, facing =true, direction = radial_out, tolerance=0.1))
    interface_fes1 = subset(interface_fes1_i, selectelem(fens1, interface_fes1_i, box=[0.25 ,0.75, 0.25, 0.75, 0.25, 0.75], inflate=1e-8))

    geom1 = NodalField(fens1.xyz)
    u1 = NodalField(zeros(size(fens1.xyz, 1), 3))



    dbc_node1 = selectnode(fens1, box=[0.5,0.5,0.5,0.5,0.5,0.5], inflate=1e-8)
    setebc!(u1, dbc_node1, 1, 0.0)
    setebc!(u1, dbc_node1, 2, 0.0)
    setebc!(u1, dbc_node1, 3, 0.0)
    dbc_node2 = selectnode(fens1, box=[0.625,0.625,0.5,0.5,0.5,0.5], inflate=1e-8)
    setebc!(u1, dbc_node2, 2, 0.0)
    setebc!(u1, dbc_node2, 3, 0.0)
    dbc_node3 = selectnode(fens1, box=[0.5,0.5,0.625,0.625,0.5,0.5], inflate=1e-8)
    setebc!(u1, dbc_node3, 3, 0.0)

    dbc_dofs1 = sort([
        3*dbc_node1 .- 2;  # ux
        3*dbc_node1 .- 1;  # uy
        3*dbc_node1;       # uz
        3*dbc_node2 .- 1;  # uy
        3*dbc_node2;       # uz
        3*dbc_node3        # uz
    ])


    applyebc!(u1)
    numberdofs!(u1)

    topsurf = subset(boundaryfes1, selectelem(fens1, boundaryfes1, box=[0., 1, 0., 1., l1, l1], inflate=1e-8))
    topfemm = FEMMBase(IntegDomain(topsurf, rule2d))
    function topfunc(forceout::Vector{T}, XYZ, tangents, feid, qpid) where {T}
        x = [-(XYZ[2]-0.5), XYZ[1]-0.5,  0.0]
        return x
    end

    fi_top = ForceIntensity(Float64, 3, topfunc)
    F1 = distribloads(topfemm, geom1, u1, fi_top, 2)

    botsurf = subset(boundaryfes1, selectelem(fens1, boundaryfes1, box=[0., 1, 0., 1., 0.0, 0.0], inflate=1e-8))
    botfemm = FEMMBase(IntegDomain(botsurf, rule2d))
    function botfunc(forceout::Vector{T}, XYZ, tangents, feid, qpid) where {T}
        x = [(XYZ[2]-0.5), -XYZ[1]+0.5,  0.0]
        return x
    end

    fi_bot = ForceIntensity(Float64, 3, botfunc)
    F1 += distribloads(botfemm, geom1, u1, fi_bot, 2)

    femm1 = FEMMDeforLinear(MR, IntegDomain(fes1, rule3d), material1)
    K1 = stiffness(femm1, geom1, u1)
    K1_ff = matrix_blocked(K1, nfreedofs(u1), nfreedofs(u1))[:ff]
    K1_fd = matrix_blocked(K1, nfreedofs(u1), nfreedofs(u1))[:fd]
    F1_ff = vector_blocked(F1, nfreedofs(u1))[:f] - K1_fd * gathersysvec(u1, :d)
    print("Time to build SD1: $(time() - ta) seconds\n")
    # SD2 #########################################################################################################################
    tb = time()
    r2 = 0.25
    l2 = 0.5
    fens2_, fes2_ = Q4annulus(0.25, 0.5, N_elem2, N_elem2*3, 2*pi)
    fens2_, fes2_ = mergenodes(fens2_, fes2_, 1e-8)
    nelem2_h = N_elem2

    fens2, fes2 = H8extrudeQ4(fens2_, fes2_, nelem2_h, (x, k) -> [x[1], x[2], k * l2 / nelem2_h])
    fens2.xyz[:, 1] .+=0.5
    fens2.xyz[:, 2] .+= 0.5
    fens2.xyz[:, 3] .+= 0.25

    boundaryfes2 = meshboundary(fes2)
    # interface_fes2 = subset(boundaryfes2, selectelem(fens2, boundaryfes2, box=[0.25, 0.75, 0.25, 0.75, 0.0, 0.5], inflate=1e-8))
    # @infiltrate
    interface_fes2 = subset(boundaryfes2, selectelem(fens2, boundaryfes2, facing =true, direction =radial_in, tolerance=0.1)) # intermediate facing z
    # interface_fes2 = subset(interface_fes2_i, selectelem(fens2, interface_fes2_i, box=[0.25, 0.75, 0.25, 0.75, 0.0, 0.5], inflate=1e-8))
    geom2 = NodalField(fens2.xyz)
    u2 = NodalField(zeros(size(fens2.xyz, 1), 3))
    # dbc_nodes2 = selectnode(fens2, box=[0.0, 1.0, 0.0, 1.0, 0.0, 0.0], inflate=1e-8)
    # setebc!(u2, dbc_nodes2, 1, 0.0)
    # setebc!(u2, dbc_nodes2, 2, 0.0)
    # setebc!(u2, dbc_nodes2, 3, 0.0)

    applyebc!(u2)
    numberdofs!(u2)

    femm2 = FEMMDeforLinear(MR, IntegDomain(fes2, GaussRule(3, 3)), material2)
    K2 = stiffness(femm2, geom2, u2)
    K2_ff = matrix_blocked(K2, nfreedofs(u2), nfreedofs(u2))[:ff]
    K2_fd = matrix_blocked(K2, nfreedofs(u2), nfreedofs(u2))[:fd]
    F2 = zeros(size(K2, 1))

    botsurf2 = subset(boundaryfes2, selectelem(fens2, boundaryfes2, box=[0., 1, 0., 1., 0.0, 0.0], inflate=1e-8))
    botfemm2 = FEMMBase(IntegDomain(botsurf2, GaussRule(2,2)))

    fi_bot2 = ForceIntensity(Float64, 3, botfunc)
    # F2 += distribloads(botfemm2, geom2, u2, fi_bot2, 2)
    F2_ff = vector_blocked(F2, nfreedofs(u2))[:f] - K2_fd * gathersysvec(u2, :d)
    print("Time to build SD2: $(time() - tb) seconds\n")

    # sekeleton ###############################################################################################################
    # fens_i, fes_i = T3circleseg(2*pi, r2, N_elem_i*6, N_elem_i)
    # fens_i.xyz[:, 1] .+= r1
    # fens_i.xyz[:, 2] .+= r1
    # # append z coordinate column
    # fens_i.xyz = hcat(fens_i.xyz, l1*ones(size(fens_i.xyz, 1)))



    # fes_i = deepcopy(interface_fes1)
    # fens_i = deepcopy(fens1)
    # connected = findunconnnodes(fens_i, fes_i)
    # fens_i, newnumbering = compactnodes(fens_i, connected)
    # fes_i = renumberconn!(fes_i, newnumbering)

    if lam_order==1
        u_i  = NodalField(zeros(size(fens_i.xyz, 1), 3))
    elseif lam_order==0
        u_i  = ElementalField(zeros(size(fes_i.conn, 1), 3))
    end
    geom_i = NodalField(fens_i.xyz)
    femm_i = FEMMDeforLinear(MR, IntegDomain(fes_i, TriRule(4)), material1)
    numberdofs!(u_i)
    mass_i = mass(femm_i, geom_i, u_i)

    print("Building coupling operators...\n")
    D1, meta1 = common_refinement(fens1, interface_fes1, fens_i, fes_i; lam_order=lam_order, h=0.05, dim_u=3, tri_order=2)
    D2, meta2 = common_refinement(fens2, interface_fes2, fens_i, fes_i; lam_order=lam_order, h=0.05, dim_u=3, tri_order=2)


    f_lams = -D1[:, dbc_dofs1] * gathersysvec(u1, :d)
    D1 = D1[:, setdiff(1:count(fens1)*3, dbc_dofs1)]
    # D2 = D2[:, setdiff(1:count(fens2)*3, dbc_dofs2)]

    # A = [K1_ff          spzeros(size(K1_ff,1), size(K2_ff,2))    D1';
    #      spzeros(size(K2_ff,1), size(K1_ff,2))     K2_ff          -D2';
    #      D1               -D2               spzeros(size(D1,1), size(D1,1))]
    # B = vcat(F1_ff, F2_ff, f_lams)
    # X = A \ B

    us, lambdas, X, stats = AL_solve(
                                        [K1_ff, K2_ff],
                                        [F1_ff, F2_ff],
                                        [
                                            [(1, +1.0, D1), (2, -1.0, D2)]
                                        ],
                                        [mass_i], [f_lams];
                                        gamma=1.0,
                                        tau=1e-3
                                    )

    scattersysvec!(u1, us[1])
    scattersysvec!(u2, us[2])
    scattersysvec!(u_i, lambdas[1])

    s1 = elemfieldfromintegpoints(femm1, geom1, u1,:vm, 1)
    s2 = elemfieldfromintegpoints(femm2, geom2, u2,:vm, 1)

    if savevtk
        # # export results
        err1 = L2error(femm1, geom1, u1, exact_u)
        err2 = L2error(femm2, geom2, u2, exact_u)
        filename = basename(@__FILE__)
        if !isdir(filename)
            mkdir(filename)
        else
            for file in readdir(filename)
                rm(joinpath(filename, file))
            end
        end
        File1 = "$filename/mesh_left.vtk"
        vtkexportmesh(
            File1,
            fens1, fes1,
            scalars = [ ("Stress", s1.values)],
            vectors = [("Displacement", u1.values)]

        )
        File2 = "$filename/mesh_right.vtk"
        vtkexportmesh(
            File2,
            fens2, fes2,
            scalars = [ ("Stress", s2.values)],
            vectors = [("Displacement", u2.values)]
        )
        File_skel = "$filename/mesh_skeleton.vtk"
        vtkexportmesh(
            File_skel,
            fens_i, fes_i,
            vectors = [("LagrangeMultiplier", u_i.values)]
        )
    end
    SE = (us[1]' * K1_ff * us[1] + us[2]' * K2_ff * us[2])/2
    return SE

end
SE_vector = Float64[]
for mult in [1,2,4,8,16]
    println("Running hollow shaft example with mult = $mult")
    SE = run_hollow_shaft(mult, true)
    println("SE = $SE")
    push!(SE_vector, SE)
end
println("SE_vector = $SE_vector")