using FinEtools
using FinEtoolsDeforLinear
using FinEtoolsDeforLinear.AlgoDeforLinearModule
using FinEtools.MeshExportModule
using LinearAlgebra
using FinEtools.AlgoBaseModule: matrix_blocked, vector_blocked
using SparseArrays

N_elem1 = 2
N_elem2 = 2
N_elem_i = 2
lam_order = 1

ri = 1.0
rmid = 2.0
ro = 3.0



function Q4sphere(radius::T, nperradius::IT;
                  tolerance = 1.0e-8 * radius) where {T<:Number, IT<:Integer}

    # First octant mesh
    fens0, fes0 = Q4spheren(radius, nperradius)

    # Mirroring a Q4 reverses orientation unless we renumber.
    q4renumb = c -> c[[1, 4, 3, 2]]

    origin = T[0, 0, 0]

    function make_octant(sx, sy, sz)
        fens, fes = fens0, fes0

        if sx < 0
            fens, fes = mirrormesh(
                fens, fes,
                T[1, 0, 0], origin;
                renumb = q4renumb
            )
        end

        if sy < 0
            fens, fes = mirrormesh(
                fens, fes,
                T[0, 1, 0], origin;
                renumb = q4renumb
            )
        end

        if sz < 0
            fens, fes = mirrormesh(
                fens, fes,
                T[0, 0, 1], origin;
                renumb = q4renumb
            )
        end

        return fens, fes
    end

    S = eltype(fens0.xyz)
    meshes = Tuple{FENodeSet, AbstractFESet}[]

    for sx in (1, -1), sy in (1, -1), sz in (1, -1)
        push!(meshes, make_octant(sx, sy, sz))
    end

    # Merge coincident nodes on symmetry planes.
    fens, fess = mergenmeshes(meshes, tolerance)

    # Since all patches are Q4, collapse the returned element-set array
    # into a single Q4 finite-element set.
    conn = reduce(vcat, [stack(fes.conn, dims=1) for fes in fess])
    fes = FESetQ4(conn)

    return fens, fes
end

function H8hollowsphere(r::RT,
                        r2::RT2,
                        nperradius::IT,
                        nLayers::IT) where {RT<:Number, RT2<:Number, IT<:Integer}

    @assert r > 0 "Inner radius r must be positive."
    @assert r2 > r "Outer radius r2 must be larger than inner radius r."
    @assert nLayers >= 1 "nLayers must be at least 1."

    # Inner spherical surface, Q4 mesh
    fens_s, fes_s = Q4sphere(float(r), nperradius)

    # Radial extrusion from inner radius r to outer radius r2.
    function radial_extrusion(x, k)
        ρ = r + (r2 - r) * k / nLayers
        return vec(x) .* (ρ / norm(vec(x)))
    end

    # Extrude Q4 surface mesh into H8 solid shell mesh
    fens, fes = H8extrudeQ4(fens_s, fes_s, nLayers, radial_extrusion)

    return fens, fes
end


fens1, fes1 = H8hollowsphere(ri, rmid, N_elem1, 1)
fens2, fes2 = H8hollowsphere(rmid, ro, N_elem2, 1)
fensi, fesi = Q4sphere(rmid, N_elem_i)

# mesh boundaries
innerbox = [-ri, ri, -ri, ri, -ri, ri]
outerbox = [-ro, ro, -ro, ro, -ro, ro]
midbox = [-rmid, rmid, -rmid, rmid, -rmid, rmid]

mb1 = meshboundary(fes1)
mb2 = meshboundary(fes2)

fei1inner = selectelem(fens1, mb1, box=innerbox, inflate =1e-8)
fei1outer = setdiff(1:size(mb1.conn,1), fei1inner)
fe1inner = subset(mb1, fei1inner)
fe1outer = subset(mb1, fei1outer)

fei2inner = selectelem(fens2, mb2, box=midbox, inflate =1e-8)
fei2outer = setdiff(1:size(mb2.conn,1), fei2inner)
fe2inner = subset(mb2, fei2inner)
fe2outer = subset(mb2, fei2outer)
# translate
fens1.xyz.+=ro
fens2.xyz.+=ro
fensi.xyz.+=ro


u1 = NodalField(zeros(size(fens1.xyz)))
u2 = NodalField(zeros(size(fens2.xyz)))
geom1 = NodalField(fens1.xyz)
geom2 = NodalField(fens2.xyz)

numberdofs!(u1)
numberdofs!(u2)

# pressure on the inner surface of inner sphere
pfemm = FEMMBase(IntegDomain(fe1inner, GaussRule(2, 2)))
function pfun_r(forceout::Vector{T}, XYZ, tangents, feid, qpid) where {T}
    return [XYZ[1], XYZ[2], XYZ[3]]
end

fi_p = ForceIntensity(Float64, 3, pfun_r)
Fp = distribloads(pfemm,geom1, u1, fi_p,2)



# D1, meta1 = common_refinement(fens1, fe1outer, fensi, fesi; lam_order, tri_order = 1, h =1.)
# D2, meta2 = common_refinement(fens2, fe2inner, fensi, fesi; lam_order, tri_order = 1, h =1.)



# ###################################################################
# Export meshes for visualization
filename = basename(@__FILE__)

if !isdir(filename)
    mkdir(filename)
else
    for file in readdir(filename)
        rm(joinpath(filename, file))
    end
end

fn1 = "$filename/inner.vtk"
fn2 = "$filename/outer.vtk"
fn3 = "$filename/skeleton.vtk"
vtkexportmesh(fn1, fens1, fes1)
vtkexportmesh(fn2, fens2, fes2)
vtkexportmesh(fn3, fensi, fesi)

ufn1 = "$filename/inner_u.vtk"
ufn2 = "$filename/outer_u.vtk"
# vtkexportmesh(ufn1, meta1["fens_u"], meta1["fes_u"])
# vtkexportmesh(ufn2, meta2["fens_u"], meta2["fes_u"])

f1 = "$filename/inner_f.vtk"
f2 = "$filename/outer_f.vtk"
vtkexportmesh(f1, fens1, fe1outer)
vtkexportmesh(f2, fens2, fe2inner)