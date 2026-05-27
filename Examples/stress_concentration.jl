using FinEtools
using FinEtoolsDeforLinear
using FinEtoolsDeforLinear.AlgoDeforLinearModule
using FinEtools.MeshExportModule
using LinearAlgebra
using FinEtools.AlgoBaseModule: matrix_blocked, vector_blocked
using SparseArrays

N_elem1 = 1
N_elem2 = 3*3
N_elem_i = 3*3
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

# left meshing #########################################################################################################################

fens1b, fes1b = T3block(0.3,0.3,2*N_elem1, 2*N_elem1)
fens1t, fes1t = T3block(0.3,0.3,2*N_elem1, 2*N_elem1)
fens1t.xyz[:,2] .+= 0.7

fens1c, fes1c = T3block(0.15,0.4, N_elem1, 2*N_elem1)
fens1c.xyz[:,2] .+= 0.3

fens1_2d, output_fes1 = mergenmeshes(Tuple{FENodeSet, AbstractFESet}[
                                                                (fens1b, fes1b),
                                                                (fens1c, fes1c),
                                                                (fens1t, fes1t),
                                                                ], 1e-8)

fes1_2d = FESetT3(vcat(connasarray(fes1b), connasarray(fes1c), connasarray(fes1t)))
fens1, fes1 = T4extrudeT3(fens1_2d, fes1_2d, 2*N_elem1, (x, k) -> [x[1], x[2], k * depth / N_elem1/2])


# output ##########################################################################################################################
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