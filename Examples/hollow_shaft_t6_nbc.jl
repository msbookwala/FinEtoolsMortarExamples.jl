using FinEtools
using FinEtoolsDeforLinear
using FinEtoolsDeforLinear.AlgoDeforLinearModule
using FinEtools.MeshExportModule
using LinearAlgebra
using FinEtools.AlgoBaseModule: matrix_blocked, vector_blocked
using SparseArrays

N_elem1 = 4
N_elem2 = 3*2
N_elem_i = 3
lam_order = 0

E = 1000.0
nu = 1/3
MR = DeforModelRed3D
material = MatDeforElastIso(MR, 0.0, E, nu, 0.0)
rule3d = TetRule(4)
rule2d = TriRule(3)

exact_u(x) = [0.0 0.0 0.0]

function right_dbc_u(x)
    # rotate by 10 degrees
    x_ = x[1] - 0.5
    y_ = x[2] - 0.5
    theta = atan(y_, x_)
    r = sqrt(x_^2 + y_^2)
    u_rot = r * cos(theta + deg2rad(90)) - r * cos(theta)
    v_rot = r * sin(theta + deg2rad(90)) - r * sin(theta)
    return [u_rot, v_rot, 0.0]
end

# SD1 #########################################################################################################################
r1 = 0.25
l1 = 1.0
fens1, fes1 = T4cylindern(r1, l1, N_elem1, N_elem1*2)

fens1.xyz[:, 1] .+= 0.5
fens1.xyz[:, 2] .+= 0.5

boundaryfes1 = meshboundary(fes1)
interface_fes1 = subset(boundaryfes1, selectelem(fens1, boundaryfes1, box=[0.25 ,0.75, 0.25, 0.75, 0, l1], inflate=1e-8))

geom1 = NodalField(fens1.xyz)
u1 = NodalField(zeros(size(fens1.xyz, 1), 3))



dbc_node1 = selectnode(fens1, box=[0.5,0.5,0.5,0.5,0.0,0.0], inflate=1e-8)
setebc!(u1, dbc_node1, 1, 0.0)
setebc!(u1, dbc_node1, 2, 0.0)
setebc!(u1, dbc_node1, 3, 0.0)
dbc_node2 = selectnode(fens1, box=[0.625,0.625,0.5,0.5,0.0,0.0], inflate=1e-8)
setebc!(u1, dbc_node2, 2, 0.0)
setebc!(u1, dbc_node2, 3, 0.0)
dbc_node3 = selectnode(fens1, box=[0.5,0.5,0.625,0.625,0.0,0.0], inflate=1e-8)
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


femm1 = FEMMDeforLinear(MR, IntegDomain(fes1, rule3d), material)
K1 = stiffness(femm1, geom1, u1)
K1_ff = matrix_blocked(K1, nfreedofs(u1), nfreedofs(u1))[:ff]
K1_fd = matrix_blocked(K1, nfreedofs(u1), nfreedofs(u1))[:fd]
F1_ff = vector_blocked(F1, nfreedofs(u1))[:f] - K1_fd * gathersysvec(u1, :d)
# SD2 #########################################################################################################################
r2 = 0.25
l2 = 0.5
fens2_, fes2_ = Q4annulus(0.25, 0.5, N_elem2, N_elem2*3, 2*pi)

fens2, fes2 = H8extrudeQ4(fens2_, fes2_, N_elem2, (x, k) -> [x[1], x[2], k * l2 / N_elem2])
fens2, fes2 = mergenodes(fens2, fes2, 1e-8)
fens2.xyz[:, 1] .+=0.5
fens2.xyz[:, 2] .+= 0.5


boundaryfes2 = meshboundary(fes2)
interface_fes2 = subset(boundaryfes2, selectelem(fens2, boundaryfes2, box=[0.25, 0.75, 0.25, 0.75, 0, l1], inflate=1e-8))

geom2 = NodalField(fens2.xyz)
u2 = NodalField(zeros(size(fens2.xyz, 1), 3))
# dbc_nodes2 = selectnode(fens2, box=[0.0, 1.0, 0.0, 1.0, 0.0, 0.0], inflate=1e-8)
# setebc!(u2, dbc_nodes2, 1, 0.0)
# setebc!(u2, dbc_nodes2, 2, 0.0)
# setebc!(u2, dbc_nodes2, 3, 0.0)

applyebc!(u2)
numberdofs!(u2)

femm2 = FEMMDeforLinear(MR, IntegDomain(fes2, rule3d), material)
K2 = stiffness(femm2, geom2, u2)
K2_ff = matrix_blocked(K2, nfreedofs(u2), nfreedofs(u2))[:ff]
K2_fd = matrix_blocked(K2, nfreedofs(u2), nfreedofs(u2))[:fd]
F2 = zeros(size(K2, 1))
F2_ff = vector_blocked(F2, nfreedofs(u2))[:f] - K2_fd * gathersysvec(u2, :d)

# sekeleton ###############################################################################################################
# fens_i, fes_i = T3circleseg(2*pi, r2, N_elem_i*6, N_elem_i)
# fens_i.xyz[:, 1] .+= r1
# fens_i.xyz[:, 2] .+= r1
# # append z coordinate column
# fens_i.xyz = hcat(fens_i.xyz, l1*ones(size(fens_i.xyz, 1)))
fes_i = deepcopy(interface_fes2)
fens_i = deepcopy(fens2)
connected = findunconnnodes(fens_i, fes_i)
fens_i, newnumbering = compactnodes(fens_i, connected)
fes_i = renumberconn!(fes_i, newnumbering)

if lam_order==1
    u_i  = NodalField(zeros(size(fens_i.xyz, 1), 3))
elseif lam_order==0
    u_i  = ElementalField(zeros(size(fes_i.conn, 1), 3))
end
numberdofs!(u_i)

D1, meta1 = common_refinement(fens1, interface_fes1, fens_i, fes_i; lam_order=lam_order, h=0.05, dim_u=3)
D2, meta2 = common_refinement(fens2, interface_fes2, fens_i, fes_i; lam_order=lam_order, h=0.05, dim_u=3)

# dbc_dofs1 = sort([3*dbc_nodes1 .- 2; 3*dbc_nodes1 .- 1;  3*dbc_nodes1])
# dbc_dofs2 = sort([3*dbc_nodes2 .- 2; 3*dbc_nodes2 .- 1;  3*dbc_nodes2])

f_lams = -D1[:, dbc_dofs1] * gathersysvec(u1, :d)
D1 = D1[:, setdiff(1:count(fens1)*3, dbc_dofs1)]
# D2 = D2[:, setdiff(1:count(fens2)*3, dbc_dofs2)]
A = [K1_ff          spzeros(size(K1_ff,1), size(K2_ff,2))    D1';
     spzeros(size(K2_ff,1), size(K1_ff,2))     K2_ff          -D2';
     D1               -D2               spzeros(size(D1,1), size(D1,1))]
B = vcat(F1_ff, F2_ff, f_lams)
X = A \ B

scattersysvec!(u1, X[1:size(K1_ff,1)])
scattersysvec!(u2, X[size(K1_ff,1)+1 : size(K1_ff,1)+size(K2_ff,1)])
scattersysvec!(u_i, X[size(K1_ff,1)+size(K2_ff,1)+1 : end])

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
    # calars = [ ("Err", err1.values)],
     vectors = [("Displacement", u1.values)]
)
File2 = "$filename/mesh_right.vtk"
vtkexportmesh(
    File2,
    fens2, fes2,
    # scalars = [ ("Err", err2.values)],
     vectors = [("Displacement", u2.values)]
)
File_skel = "$filename/mesh_skeleton.vtk"
vtkexportmesh(
    File_skel,
    fens_i, fes_i,scalars = [],
    vectors = [("LagrangeMultiplier", u_i.values)]
)

# Fileu1 = "$filename/union_left.vtk"
# vtkexportmesh(
#     Fileu1,
#     meta1["fens_u"], meta1["fes_u"],scalars = []
# )
# Fileu2 = "$filename/union_right.vtk"
# vtkexportmesh(
#     Fileu2,
#     meta2["fens_u"], meta2["fes_u"],scalars = []
# )