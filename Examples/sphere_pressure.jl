using FinEtools
using FinEtoolsDeforLinear
using FinEtoolsDeforLinear.AlgoDeforLinearModule
using FinEtools.MeshExportModule
using LinearAlgebra
using FinEtools.AlgoBaseModule: matrix_blocked, vector_blocked
using SparseArrays

multfactor = 4
N_elem1 = 2*multfactor
N_elem2 = 3*multfactor
N_elem_i = 2*multfactor
lam_order = 1



ri = 1.0
rmid = 2.0
ro = 3.0
p=1

E = 1.0
nu = 1/3
MR = DeforModelRed3D
material = MatDeforElastIso(MR, 0.0, E, nu, 0.0)
trule3d = TetRule(4)
trule2d = TriRule(3)
grule3d = GaussRule(3, 3)
grule2d = GaussRule(2, 3)

p_exact = function (x)
    return -ri^3 / (ro^3-ri^3) *p
end
u_exact = function (x)
    r = norm(x.-ro)
    return ri^3 / (ro^3-ri^3) * ((1-2*nu)*r+ (1+nu)*ro^3 / 2/r^2)*p/E
end

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

function T4hollowsphere(r::RT,
                        r2::RT2,
                        nperradius::IT,
                        nLayers::IT) where {RT<:Number, RT2<:Number, IT<:Integer}

    @assert r > 0 "Inner radius r must be positive."
    @assert r2 > r "Outer radius r2 must be larger than inner radius r."
    @assert nLayers >= 1 "nLayers must be at least 1."

    # Inner spherical surface, Q4 mesh
    fens_s, fes_s = Q4sphere(float(r), nperradius)
    fens_s, fes_s = Q4toT3(fens_s, fes_s)

    # Radial extrusion from inner radius r to outer radius r2.
    function radial_extrusion(x, k)
        ρ = r + (r2 - r) * k / nLayers
        return vec(x) .* (ρ / norm(vec(x)))
    end

    # Extrude Q4 surface mesh into H8 solid shell mesh
    fens, fes = T4extrudeT3(fens_s, fes_s, nLayers, radial_extrusion)

    return fens, fes
end



fens1, fes1 = H8hollowsphere(ri, rmid, N_elem1, ceil(Int, N_elem1/2))
fens2, fes2 = H8hollowsphere(rmid, ro, N_elem2, ceil(Int, N_elem2/2))
rule = grule3d
rule2d = grule2d

# fens1,fes1 = T4hollowsphere(ri, rmid, N_elem1, 4)
# fens2,fes2 = T4hollowsphere(rmid, ro, N_elem2, 4)
# rule = trule3d
# rule2d = trule2d


fensi, fesi = Q4sphere(rmid, N_elem_i)
fensi, fesi = Q4toT3(fensi, fesi)

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


dbc_node1 = selectnode(fens1, box=[ro+ri,ro+ri, ro,ro,ro,ro], inflate=1e-8)
setebc!(u1, dbc_node1, 1, 0.0)
setebc!(u1, dbc_node1, 2, 0.0)
setebc!(u1, dbc_node1, 3, 0.0)
dbc_node2 = selectnode(fens1, box=[ro+rmid,ro+rmid, ro,ro,ro,ro], inflate=1e-8)
setebc!(u1, dbc_node2, 2, 0.0)
setebc!(u1, dbc_node2, 3, 0.0)
dbc_node3 = selectnode(fens1, box=[ro,ro,ro+ri,ro+ri,ro,ro], inflate=1e-8)
setebc!(u1, dbc_node3, 3, 0.0)
# setebc!(u1, dbc_node3, 2, 0.0)
dbc_nodes = sort([dbc_node1; dbc_node2; dbc_node3])
applyebc!(u1)
applyebc!(u2)
numberdofs!(u1)
numberdofs!(u2)

femm1 = FEMMDeforLinear(MR, IntegDomain(fes1, rule), material)
K1 = stiffness(femm1, geom1, u1)
K1_ff = matrix_blocked(K1, nfreedofs(u1), nfreedofs(u1))[:ff]
K1_fd = matrix_blocked(K1, nfreedofs(u1), nfreedofs(u1))[:fd]

femm2 = FEMMDeforLinear(MR, IntegDomain(fes2, rule), material)
K2 = stiffness(femm2, geom2, u2)
K2_ff = matrix_blocked(K2, nfreedofs(u2), nfreedofs(u2))[:ff]
K2_fd = matrix_blocked(K2, nfreedofs(u2), nfreedofs(u2))[:fd]


# pressure on the inner surface of inner sphere
pfemm = FEMMBase(IntegDomain(fe1inner, rule2d))
function pfun_r(forceout::Vector{T}, XYZ, tangents, feid, qpid) where {T}
    x = [XYZ[1]-ro, XYZ[2]-ro, XYZ[3]-ro]
    return x * p/ norm(x) 
end

fi_p = ForceIntensity(Float64, 3, pfun_r)
Fp = distribloads(pfemm,geom1, u1, fi_p,2)

F1_ff = vector_blocked(Fp, nfreedofs(u1))[:f]
F2_ff = zeros(nfreedofs(u2))

println("Assembling system...")

D1, meta1 = common_refinement(fens1, fe1outer, fensi, fesi; lam_order, tri_order = 2, h =0.05, dim_u=3)
D2, meta2 = common_refinement(fens2, fe2inner, fensi, fesi; lam_order, tri_order = 2, h =0.05, dim_u=3)
println("Solving now")

dbc_dofs = sort([
    3*dbc_node1 .- 2;  # ux
    3*dbc_node1 .- 1;  # uy
    3*dbc_node1;       # uz

    3*dbc_node2 .- 1;  # uy
    3*dbc_node2;       # uz

    3*dbc_node3        # uz
])
                    # error("dbc_dofs = $dbc_dofs")
D1 = D1[:,setdiff(1:3*count(fens1), dbc_dofs)]



A = [K1_ff          spzeros(size(K1_ff,1), size(K2_ff,2))    D1';
     spzeros(size(K2_ff,1), size(K1_ff,2))     K2_ff          -D2';
     D1               -D2               spzeros(size(D1,1), size(D1,1))]
B = vcat(F1_ff, F2_ff, zeros(size(D1, 1)))
X = A \ B
println("Done solving")
scattersysvec!(u1, X[1:size(K1_ff,1)])
scattersysvec!(u2, X[size(K1_ff,1)+1 : size(K1_ff,1)+size(K2_ff,1)])
# scattersysvec!(u_i, X[size(K1_ff,1)+size(K2_ff,1)+1 : end])
st1 = elemfieldfromintegpoints(femm1, geom1, u1,:Cauchy, 1:6)
st2 = elemfieldfromintegpoints(femm2, geom2, u2,:Cauchy, 1:6)
# hss1 = ElementalField(sum(st1.values[:,1:3], dims=2) ./ 3)
# hss2 = ElementalField(sum(st2.values[:,1:3], dims=2) ./ 3)

# st1 = fieldfromintegpoints(femm1, geom1, u1,:Pressure, 1)
# st2 = fieldfromintegpoints(femm2, geom2, u2,:Pressure, 1)
# hss1 = NodalField(sum(st1.values[:,1:3], dims=2) ./ 3)
# hss2 = NodalField(sum(st2.values[:,1:3], dims=2) ./ 3)



# p1 = fieldfromintegpoints(femm1, geom1, u1,:Pressure, 1)
# p2 = fieldfromintegpoints(femm2, geom2, u2,:Pressure, 1)
p1 = elemfieldfromintegpoints(femm1, geom1, u1,:Pressure, 1)
p2 = elemfieldfromintegpoints(femm2, geom2, u2,:Pressure, 1)

p1error = L2error(femm1, geom1, p1, p_exact)
p2error = L2error(femm2, geom2, p2, p_exact)



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
vtkexportmesh(fn1, fens1, fes1, vectors = [("Disp", u1.values)], scalars = [#("Cauchy", st1.values), 
                                                                        ("Pressure", p1.values), 
                                                                        ("p_error", p1error.values)])
vtkexportmesh(fn2, fens2, fes2, vectors = [("Disp", u2.values)], scalars = [#("Cauchy", st2.values), 
                                                                        ("Pressure", p2.values), 
                                                                        ("p_error", p2error.values)])
vtkexportmesh(fn3, fensi, fesi)

ufn1 = "$filename/inner_u.vtk"
ufn2 = "$filename/outer_u.vtk"
vtkexportmesh(ufn1, meta1["fens_u"], meta1["fes_u"])
vtkexportmesh(ufn2, meta2["fens_u"], meta2["fes_u"])

f1 = "$filename/inner_f.vtk"
f2 = "$filename/outer_f.vtk"
vtkexportmesh(f1, fens1, fe1outer)
vtkexportmesh(f2, fens2, fe2inner)

p_l2error1 = sqrt(sum(p1error.values.^2))
p_l2error2 = sqrt(sum(p2error.values.^2))
total_l2error = sqrt(p_l2error1^2 + p_l2error2^2)
println("L2 error in pressure, inner sphere: $p_l2error1")
println("L2 error in pressure, outer sphere: $p_l2error2")
println("Total L2 error in pressure: $total_l2error")