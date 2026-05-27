using FinEtools
using FinEtoolsDeforLinear
using FinEtoolsDeforLinear.AlgoDeforLinearModule
using FinEtools.MeshExportModule
using LinearAlgebra
using FinEtools.AlgoBaseModule: matrix_blocked, vector_blocked
using SparseArrays

N_elem1 = 3
N_elem2 = 5
N_elem3 = 3
N_elem_i = 3
lam_order = 1
depth = 0.3

E = 1.0
nu = 1/3
MR = DeforModelRed3D
material = MatDeforElastIso(MR, 0.0, E, nu, 0.0)
trule3d = TetRule(4)
trule2d = TriRule(3)
grule3d = GaussRule(3, 3)
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

fesi1 = FESetT3(vcat(connasarray(fesi1_b), connasarray(fesi1_t), connasarray(fesi1_c))) 
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

fes1_2d = FESetT3(vcat(connasarray(fes1b), connasarray(fes1c), connasarray(fes1t)))
fens1, fes1 = T4extrudeT3(fens1_2d, fes1_2d, 2*N_elem1, (x, k) -> [x[1], x[2], k * depth / N_elem1/2])

mb1 = meshboundary(fes1)
ifes1b = subset(mb1, selectelem(fens1, mb1, box=[0.15, 0.3, 0.3, 0.3, 0, 1], inflate=1e-8))
ifes1t = subset(mb1, selectelem(fens1, mb1, box=[0.15, 0.3, 0.7, 0.7, 0, 1], inflate=1e-8))
ifes1c = subset(mb1, selectelem(fens1, mb1, box=[0.15, 0.15, 0.3, 0.7, 0, 1], inflate=1e-8))
ifes1 = FESetT3(vcat(connasarray(ifes1b), connasarray(ifes1c), connasarray(ifes1t)))

D12,meta12 = common_refinement(fens1, ifes1, fensi1, fesi1; lam_order=lam_order, h=0.05, dim_u=3)

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
fes2_2d = FESetQ4(vcat(connasarray(fes2b), 
                        connasarray(fes2t), 
                        connasarray(fes2r), 
                        connasarray(fes2bb), 
                        connasarray(fes2tt)))

N_elem2_depth = ceil(Int, 2*N2_inc*1.5)
fens2, fes2 = H8extrudeQ4(fens2_2d, fes2_2d, N_elem2_depth, (x, k) -> [x[1], x[2], k * depth / N_elem2_depth])
mb2  = meshboundary(fes2)
ifes21c = subset(mb2, selectelem(fens2, mb2, box=[0.15, 0.15, 0.3, 0.7, 0., 1.], inflate=1e-8))
ifes21b  = subset(mb2, selectelem(fens2, mb2, box=[0.15, 0.3, 0.3, 0.3, 0., 1.], inflate=1e-8))
ifes21t  = subset(mb2, selectelem(fens2, mb2, box=[0.15, 0.3, 0.7, 0.7, 0., 1.], inflate=1e-8))
ifes21 = FESetQ4(vcat(connasarray(ifes21b), connasarray(ifes21c), connasarray(ifes21t)))
D21,meta21 = common_refinement(fens2, ifes21, fensi1, fesi1; lam_order=lam_order, h=0.05, dim_u=3, tri_order=2)

ifes22 = subset(mb2, selectelem(fens2, mb2, box=[0.4, 0.4, 0.0, 1.0, 0., 1.], inflate=1e-8))
D22,meta22 = common_refinement(fens2, ifes22, fensi2, fesi2; lam_order=lam_order, h=0.05, dim_u=3, tri_order=2)

## right meshing ##############################################################################
N_elem3_depth = N_elem3
N_elem3_height = ceil(Int, N_elem3_depth*0.666)
fens3, fes3 = T4block(0.3, 0.2, depth, N_elem3, N_elem3_height, N_elem3_depth)
fens3.xyz[:, 1] .+= 0.4
fens3.xyz[:, 2] .+= 0.4
mb3 = meshboundary(fes3)
ifes32 = subset(mb3, selectelem(fens3, mb3, box=[0.4, 0.4, 0.0, 1., 0., 1.], inflate=1e-8))
D32,meta32 = common_refinement(fens3, ifes32, fensi2, fesi2; lam_order=lam_order, h=0.05, dim_u=3)

# output #########################################################################################
filename = basename(@__FILE__)

if !isdir(filename)
    mkdir(filename)
else
    for file in readdir(filename)
        rm(joinpath(filename, file))
    end
end
file_left = "$filename/left.vtk"
vtkexportmesh(
    file_left,
    fens1, fes1,
    # scalars = [ ("Err", err1.values)],
    #  vectors = [("Displacement", u1.values)]
)
file_right = "$filename/right.vtk"
vtkexportmesh(
    file_right,
    fens3, fes3,
    # scalars = [ ("Err", err1.values)],
    #  vectors = [("Displacement", u1.values)]
)
file_center = "$filename/center.vtk"
vtkexportmesh(
    file_center,
    fens2, fes2,
    # scalars = [ ("Err", err1.values)],
    #  vectors = [("Displacement", u1.values)]
)
file_li = "$filename/left_interface.vtk"
vtkexportmesh(
    file_li,
    fensi1, fesi1
)
file_ri = "$filename/right_interface.vtk"
vtkexportmesh(
    file_ri,
    fensi2, fesi2
)

file_11 = "$filename/union_11.vtk"
vtkexportmesh(
    file_11,
    meta12["fens_u"], meta12["fes_u"]
)
file_21 = "$filename/union_21.vtk"
vtkexportmesh(
    file_21,
    meta21["fens_u"], meta21["fes_u"]
)
file_22 = "$filename/union_22.vtk"
vtkexportmesh(
    file_22,
    meta22["fens_u"], meta22["fes_u"]
)
file_32 = "$filename/union_32.vtk"
vtkexportmesh(
    file_32,
    meta32["fens_u"], meta32["fes_u"]
)
