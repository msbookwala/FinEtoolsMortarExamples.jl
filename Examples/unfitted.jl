using FinEtools
using FinEtools.AlgoBaseModule: solve_blocked!, matrix_blocked, vector_blocked
using FinEtoolsHeatDiff
using FinEtoolsHeatDiff.AlgoHeatDiffModule
using FinEtools.MeshExportModule.VTK: vtkexportmesh, T3, vtkexportvectors
using LinearAlgebra
using FinEtoolsDeforLinear
using FinEtoolsDeforLinear.AlgoDeforLinearModule


N_elem1 = 6
N_elem2 = 4
N_elem_i = 4
left_m = "t"
right_m = "t"
skew = 0.
lam_order = 1
bend=1.

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
depth1 = 0.5
if left_m == "h"
    fens1, fes1 = H8block(width1, height1, depth1, floor(Int, N_elem1), N_elem1, N_elem1)
    Rule1 = GaussRule(3,2)
else
    fens1, fes1 = T4block(width1, height1, depth1, floor(Int, N_elem1), N_elem1, N_elem1)
    Rule1 = TetRule(4)
end
fens1.xyz[:,2] .+= 0.25
fens1.xyz[:,3] .+= 0.25
boundaryfes1 = meshboundary(fes1)
edge_fes1 = subset(boundaryfes1, selectelem(fens1, boundaryfes1,  box=[width1,width1, 0.0,1.0, 0.0, 1.0], inflate=1e-8))


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
# setebc!(u2, [dbc_nodes2_3], 2, 0.0)
# setebc!(u2, [dbc_nodes2_3], 3, 0.0)

# box2 = [1.0,1.0,0.0,1.0,0.0,1.0]
# dbc_nodes2 = selectnode(fens2; box=box2, inflate=1e-8)
# for i in dbc_nodes2
#     setebc!(u2, [i],1, 0.0)
#     setebc!(u2, [i],2, 0.0)
#     setebc!(u2, [i],3, 0.0)
# end



applyebc!(u2)
numberdofs!(u2)

femm2 = FEMMDeforLinear(MR, IntegDomain(fes2, Rule2), material)
K2 = stiffness(femm2, geom2, u2)
K2_ff = matrix_blocked(K2, nfreedofs(u2), nfreedofs(u2))[:ff]

l2 = selectelem(fens2, meshboundary(fes2), box = [0.5,0.5, 0.0,1.0, 0.0, 1.0], inflate=1e-8)
el2femm = FEMMBase(IntegDomain(subset(meshboundary(fes2), l2), TriRule(3)))

function pfun_r(forceout::Vector{T}, XYZ, tangents, feid, qpid) where {T}
    if XYZ[2]<=0.25 || XYZ[2]>=0.75 || XYZ[3]<=0.25 || XYZ[3]>=0.75
        return Float64[t0, 0.0, 0.0]
    else
        return Float64[0.0, 0.0, 0.0]
    end
end
fi2 = ForceIntensity(Float64, 3, pfun_r)
F2 = distribloads(el2femm, geom2, u2, fi2, 2)
# error("")

l2_right = selectelem(fens2, meshboundary(fes2), box = [1.0,1.0, 0.0,1.0, 0.0, 1.0], inflate=1e-8)
el2femm_right = FEMMBase(IntegDomain(subset(meshboundary(fes2), l2_right), TriRule(3)))
fi2_right = ForceIntensity(Float64[-t0, 0.0, 0.0])
F2 += distribloads(el2femm_right, geom2, u2, fi2_right, 2) 

F2_ff = vector_blocked(F2, nfreedofs(u2))[:f]
##########################################################################################

xs_i = 0.5
ys_i = collect(linearspace(0.25, 0.75, N_elem_i+1))
zs_i = collect(linearspace(0.25, 0.75, N_elem_i+1))
fens_i, fes_i = T3blockx(ys_i, zs_i, :a)
fens_i.xyz = hcat(xs_i*ones(size(fens_i.xyz, 1), 1), fens_i.xyz)
# fens_i.xyz[:, 1] .+= bend * fens_i.xyz[:, 1].*(fens_i.xyz[:, 2] .- 0.5).^2

if lam_order==1
    u_i  = NodalField(zeros(size(fens_i.xyz, 1), 3))
else
    u_i  = ElementalField(zeros(size(fes_i.conn, 1), 3))
end

femm_i = FEMMDeforLinear(MR, IntegDomain(fes_i, TriRule(9)), material)
geom_i = NodalField(fens_i.xyz)
applyebc!(u_i)

numberdofs!(u_i)

# D1, Pi_NC1, Pi_phi1, M_u1 = build_D_matrix(fens_i, fes_i, fens1, edge_fes1; lam_order=lam_order,tol=1e-8)
# D2, Pi_NC2, Pi_phi2, M_u2 = build_D_matrix(fens_i, fes_i, fens2, edge_fes2; lam_order=lam_order,tol=1e-8)


D1, meta1 = common_refinement(fens1, edge_fes1, fens_i, fes_i; lam_order=lam_order, h=0.03, dim_u=3, tri_order=2)
D2, meta2 = common_refinement(fens2, edge_fes2, fens_i, fes_i; lam_order=lam_order, h=0.033, dim_u=3, tri_order=2)

dbc_dofs2 = sort([
        3*dbc_nodes2_1 .- 2;  # ux
        3*dbc_nodes2_1 .- 1;  # uy
        3*dbc_nodes2_1;       # uz

        3*dbc_nodes2_2 .- 2;  # ux
        3*dbc_nodes2_2;       # uz

        3*dbc_nodes2_3 .- 2        # ux
    ])

# dbc_dofs2 = sort([
#         3*dbc_nodes2 .- 2;  # ux
#         3*dbc_nodes2 .- 1;  # uy
#         3*dbc_nodes2;       # uz
#     ])

D2 = D2[:, setdiff(1:3*count(fens2), dbc_dofs2)]

A = [K1_ff          zeros(size(K1_ff,1), size(K2_ff,2))    D1';
     zeros(size(K2_ff,1), size(K1_ff,2))     K2_ff          -D2';
     D1               -D2               zeros(size(D1,1), size(D1,1))]
B = vcat(F1_ff, F2_ff, zeros(size(D1,1)))
X = A \ B

scattersysvec!(u1, X[1:size(K1_ff,1)])
scattersysvec!(u2, X[size(K1_ff,1)+1 : size(K1_ff,1)+size(K2_ff,1)])
scattersysvec!(u_i, X[size(K1_ff,1)+size(K2_ff,1)+1 : end])

# sol(x) = x[1]-1
err1 = L2error(femm1, geom1, u1, u_exact)
err2 = L2error(femm2, geom2, u2, u_exact)

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
File1 = "$filename/mesh_left.vtk"
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

u1_fens = meta1["fens_u"]
u2_fens = meta2["fens_u"]
u1_fes = meta1["fes_u"]
u2_fes = meta2["fes_u"]
file3 = "$filename/pt_union_left.vtk"
vtkexportmesh(
    file3,
    u1_fens, u1_fes,scalars = []
)
file4 = "$filename/pt_union_right.vtk"
vtkexportmesh(
    file4,
    u2_fens, u2_fes,scalars = []
)
# println(u_i.values)