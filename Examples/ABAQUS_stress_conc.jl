using FinEtools
using FinEtoolsDeforLinear
using FinEtoolsDeforLinear.AlgoDeforLinearModule
using FinEtools.MeshExportModule
using LinearAlgebra
using FinEtools.AlgoBaseModule: matrix_blocked, vector_blocked
using SparseArrays
using Krylov


# function run_stress_concentration(r, m=2)
#     println("Running with r = $r")
m=2
r=3
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

    left_data = import_ABAQUS("Examples/ABAQUS_meshes/l-r$r.inp")
    fens1 = left_data["fens"]
    fes1 = left_data["fesets"][1]
    geom1 = NodalField(fens1.xyz)
    u1 = NodalField(zeros(size(fens1.xyz, 1), 3))

    dbc_node1 = selectnode(fens1, box=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], inflate=1e-7)
    setebc!(u1, dbc_node1, 1, 0.0)
    setebc!(u1, dbc_node1, 2, 0.0)
    setebc!(u1, dbc_node1, 3, 0.0)
    dbc_node2 = selectnode(fens1, box=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0], inflate=1e-7)
    setebc!(u1, dbc_node2, 1, 0.0)
    # setebc!(u1, dbc_node2, 3, 0.0)
    dbc_node3 = selectnode(fens1, box=[0.0, 0.0, 0.0, 0.0, 0.5, 0.5], inflate=1e-7)
    setebc!(u1, dbc_node3, 1, 0.0)
    setebc!(u1, dbc_node3, 2, 0.0)

    dbc_nodes = sort([dbc_node1; dbc_node2; dbc_node3])
    applyebc!(u1)
    numberdofs!(u1)
    femm1 = FEMMDeforLinear(MR, IntegDomain(fes1, grule3d), material)
    K1 = stiffness(femm1, geom1, u1)
    K1_ff = matrix_blocked(K1, nfreedofs(u1), nfreedofs(u1))[:ff]
    K1_fd = matrix_blocked(K1, nfreedofs(u1), nfreedofs(u1))[:fd]

    mb1 = meshboundary(fes1)
    left_fes1 = subset(mb1, selectelem(fens1, mb1, box=[0.0, 0.0, 0.0, 1.0, 0.0, 1.0], inflate=1e-7))
    left_femm1 = FEMMBase(IntegDomain(left_fes1, grule2d))
    function fi_fun_l(forceout::Vector{T}, XYZ, tangents, feid, qpid) where {T}
        return [-2.5e-2, 0.0, 0.0]
    end
    fi_l = ForceIntensity(Float64, 3, fi_fun_l)
    F1 = distribloads(left_femm1, geom1, u1, fi_l,2)
    F1_ff = vector_blocked(F1, nfreedofs(u1))[:f] - K1_fd*gathersysvec(u1, :d)

    ifes1 = subset(mb1, selectelem(fens1, mb1, box=[0.35, 0.5, 0.1, 0.9, 0, 1], inflate=1e-5))
    connected = findunconnnodes(fens1, ifes1)
    fensi1, newnumbering = compactnodes(fens1, connected)
    fesi1 = renumberconn!(ifes1, newnumbering)

    ifes1 = subset(mb1, selectelem(fens1, mb1, box=[0.35, 0.5, 0.1, 0.9, 0, 1], inflate=1e-5))

    

    # ifes1b = subset(mb1, selectelem(fens1, mb1, box=[0.15, 0.3, 0.3, 0.3, 0, 1], inflate=1e-7))
    # ifes1t = subset(mb1, selectelem(fens1, mb1, box=[0.15, 0.3, 0.7, 0.7, 0, 1], inflate=1e-7))
    # ifes1c = subset(mb1, selectelem(fens1, mb1, box=[0.15, 0.15, 0.3, 0.7, 0, 1], inflate=1e-7))
    # ifes1 = FESetT3(vcat(connasarray(ifes1b), connasarray(ifes1c), connasarray(ifes1t)))

    D11,_ = common_refinement(fens1, ifes1, fensi1, fesi1; lam_order=lam_order, h=0.05, dim_u=3, tri_order=2)
    dbc_dofs = sort([3*dbc_node1 .- 2; 3*dbc_node1 .- 1; 3*dbc_node1;
                        3*dbc_node2 .- 2;
                        3*dbc_node3; 3*dbc_node3.-1 ])
    D11 = D11[:,setdiff(1:3*count(fens1), dbc_dofs)]

    ## right meshing ##############################################################################

    right_data = import_ABAQUS("Examples/ABAQUS_meshes/r-r$r.inp")
    fens3 = right_data["fens"]
    fes3 = right_data["fesets"][1]

    geom3 = NodalField(fens3.xyz)
    u3 = NodalField(zeros(size(fens3.xyz, 1), 3))
    femm3 = FEMMDeforLinear(MR, IntegDomain(fes3, grule3d), material)
    numberdofs!(u3)
    K3 = stiffness(femm3, geom3, u3)
    K3_ff = matrix_blocked(K3, nfreedofs(u3), nfreedofs(u3))[:ff]

    mb3 = meshboundary(fes3)

    right_fes3 = subset(mb3, selectelem(fens3, mb3, box=[1.0, 1.0, 0.0, 1.0, 0.0, 1.0], inflate=1e-7))
    right_femm3 = FEMMBase(IntegDomain(right_fes3, grule2d))
    function fi_fun_r(forceout::Vector{T}, XYZ, tangents, feid, qpid) where {T}
        return [5.0e-2, 0.0, 0.0]
    end
    fi_r = ForceIntensity(Float64, 3, fi_fun_r)
    F3 = distribloads(right_femm3, geom3, u3, fi_r,2)
    F3_ff = vector_blocked(F3, nfreedofs(u3))[:f]

    ifes32 = subset(mb3, selectelem(fens3, mb3, box=[0.65, 0.65, 0., 1., 0., 1.], inflate=1e-7))
    connected = findunconnnodes(fens3, ifes32)
    fensi2, newnumbering = compactnodes(fens3, connected)
    fesi2 = renumberconn!(ifes32, newnumbering)

    ifes32 = subset(mb3, selectelem(fens3, mb3, box=[0.65, 0.65, 0.0, 1., 0., 1.], inflate=1e-7))

    D32,_ = common_refinement(fens3, ifes32, fensi2, fesi2; lam_order=lam_order, h=0.05, dim_u=3, tri_order=2)

    # central meshing ################################################################################

    cent_data = import_ABAQUS("Examples/ABAQUS_meshes/c-r$r.inp")
    fens2 = cent_data["fens"]
    fes2 = cent_data["fesets"][1]

    geom2 = NodalField(fens2.xyz)
    u2 = NodalField(zeros(size(fens2.xyz, 1), 3))
    femm2 = FEMMDeforLinear(MR, IntegDomain(fes2, grule3d), material)
    numberdofs!(u2)
    K2 = stiffness(femm2, geom2, u2)
    K2_ff = matrix_blocked(K2, nfreedofs(u2), nfreedofs(u2))[:ff]
    F2 = zeros((nfreedofs(u2),1))


    mb2  = meshboundary(fes2)
    # ifes21 = subset(mb2, selectelem(fens2, mb2, box=[0.35, 0.5-1e-6, 0.1, 0.9, 1e-5, 0.5-1e-5], inflate=1e-7))

    ifes21c = subset(mb2, selectelem(fens2, mb2, box=[0.35, 0.35, 0.1, 0.9, 0., 0.5], inflate=1e-7))
    ifes21b  = subset(mb2, selectelem(fens2, mb2, box=[0.35, 0.5, 0.1, 0.1, 0., 0.5], inflate=1e-7))
    ifes21t  = subset(mb2, selectelem(fens2, mb2, box=[0.35, 0.5, 0.9, 0.9, 0., 0.5], inflate=1e-7))
    ifes21 = FESetQ4(vcat(connasarray(ifes21b), connasarray(ifes21c), connasarray(ifes21t)))
    D21,_ = common_refinement(fens2, ifes21, fensi1, fesi1; lam_order=lam_order, h=0.05, dim_u=3, tri_order=2)

    ifes22 = subset(mb2, selectelem(fens2, mb2, box=[0.65, 0.65, 0.0, 1.0, 0., 1.], inflate=1e-7))
    D22,_ = common_refinement(fens2, ifes22, fensi2, fesi2; lam_order=lam_order, h=0.05, dim_u=3, tri_order=2)




    # assembly and results ######################################################################################

    D = [D11 -D21 spzeros(size(D11,1),size(D32,2));
        spzeros(size(D22,1),size(D11,2)) -D22 D32;]
    # K = [K1_ff spzeros(size(K1_ff, 1), size(K2_ff, 2)) spzeros(size(K1_ff, 1), size(K3_ff, 2));
    #     spzeros(size(K2_ff, 1), size(K1_ff, 2)) K2_ff spzeros(size(K2_ff, 1), size(K3_ff, 2));
    #     spzeros(size(K3_ff, 1), size(K1_ff, 2)) spzeros(size(K3_ff, 1), size(K2_ff, 2)) K3_ff]

    K = blockdiag(K1_ff, K2_ff, K3_ff) 
    b = vec([F1_ff; F2; F3_ff; zeros(size(D, 1), 1)])
    A = [K D'; 
            D 0*sparse(I, size(D, 1), size(D, 1))]

    # bb = vec([F1_ff; F2; F3_ff;])
    # bc = zeros(size(D, 1), 1)
    # print(size(A))
    # print(rank(A))
    #  x = A\b

    @time x,_ = krylov_solve(Val(:minres),A,b, rtol=1e-7,verbose=1)
    scattersysvec!(u1, x[1:nfreedofs(u1)])
    scattersysvec!(u2, x[nfreedofs(u1)+1:nfreedofs(u1)+nfreedofs(u2)])
    scattersysvec!(u3, x[nfreedofs(u1)+nfreedofs(u2)+1:nfreedofs(u1)+nfreedofs(u2)+nfreedofs(u3)])
    st1 = fieldfromintegpoints(femm1, geom1, u1,:Cauchy, 1)
    st2 = fieldfromintegpoints(femm2, geom2, u2,:Cauchy, 1)
    st3 = fieldfromintegpoints(femm3, geom3, u3,:Cauchy, 1)
    # st1 = elemfieldfromintegpoints(femm1, geom1, u1,:vm, 1)
    # st2 = elemfieldfromintegpoints(femm2, geom2, u2,:vm, 1)
    # st3 = elemfieldfromintegpoints(femm3, geom3, u3,:vm, 1)
    # st1 = elemfieldfromintegpoints(femm1, geom1, u1,:princCauchy, 1)
    # st2 = elemfieldfromintegpoints(femm2, geom2, u2,:princCauchy, 1)
    # st3 = elemfieldfromintegpoints(femm3, geom3, u3,:princCauchy, 1)

    println("max stress = ", maximum(st2.values))

    # output #########################################################################################
    filename = basename(@__FILE__)

    # if !isdir(filename)
    #     mkdir(filename)
    # else
    #     for file in readdir(filename)
    #         rmr(joinpath(filename, file))
    #     end
    # end
    if isdir("$filename/r$r")
        for file in readdir("$filename/r$r")
            rm(joinpath("$filename/r$r", file))
        end
    else
        mkdir("$filename/r$r")
    end

    file_left = "$filename/r$r/left.vtk"
    vtkexportmesh(
        file_left,
        fens1, fes1,
        scalars = [ ("St", st1.values)],
        vectors = [("Displacement", u1.values)]
    )
    file_right = "$filename/r$r/right.vtk"
    vtkexportmesh(
        file_right,
        fens3, fes3,
        scalars = [ ("St", st3.values)],
        vectors = [("Displacement", u3.values)]
    )
    file_center = "$filename/r$r/center.vtk"
    vtkexportmesh(
        file_center,
        fens2, fes2,
        scalars = [ ("St", st2.values)],
        vectors = [("Displacement", u2.values)]
    )
    file_li = "$filename/r$r/left_interface.vtk"
    vtkexportmesh(
        file_li,
        fensi1, fesi1
    )
    file_ri = "$filename/r$r/right_interface.vtk"
    vtkexportmesh(
        file_ri,
        fensi2, fesi2
    )
    return maximum(st2.values)

    # file_11 = "$filename/union_11.vtk"
    # vtkexportmesh(
    #     file_11,
    #     meta11["fens_u"], meta11["fes_u"]
    # )
    # file_21 = "$filename/union_21.vtk"
    # vtkexportmesh(
    #     file_21,
    #     meta21["fens_u"], meta21["fes_u"]
    # )
    # file_22 = "$filename/union_22.vtk"
    # vtkexportmesh(
    #     file_22,
    #     meta22["fens_u"], meta22["fes_u"]
    # )
    # file_32 = "$filename/union_32.vtk"
    # vtkexportmesh(
    #     file_32,
    #     meta32["fens_u"], meta32["fes_u"]
    # )

# end
# s = []
# for r in 2:2
#     max_stress = run_stress_concentration(r,2)
#     push!(s, max_stress)
# end
# s = s[end-2:end]
# extrapolated_s = (s[2]^2 - s[3]*s[1])/(2*s[2] - s[1] - s[3])
# println("extrapolated s = ", extrapolated_s)