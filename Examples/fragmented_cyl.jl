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

t0 = 0.05
E = 1.0
nu = 1/3
MR = DeforModelRed3D
material =  MatDeforElastIso(MR, 1.0, E, nu, 0.0)

sector_angle = 2pi / 5
inner_radius = 0.5
nradial = 5
ncircumferential = 5
depth = 0.5
nlayers = 5
lam_order = 1

function radial_interface_faces(fens, fes, theta, sector_angle; tolerance=0.1)
    bfes = meshboundary(fes)

    ids_start = selectelem(fens, bfes; facing=true,
        direction=x -> [sin(theta), -cos(theta), 0.0],
        tolerance=tolerance)

    theta_end = theta + sector_angle
    ids_end = selectelem(fens, bfes; facing=true,
        direction=x -> [-sin(theta_end), cos(theta_end), 0.0],
        tolerance=tolerance)

    return subset(bfes, ids_start), subset(bfes, ids_end)
end


# ------------------------------------------------------------------
# Sector 1: 0 to 72 degrees
# ------------------------------------------------------------------
nradial1 = 5
ncircumferential1 = 5
nlayers1 = 5
fens_c1, fes_c1 = T3circleseg(
    sector_angle,
    inner_radius,
    nradial1,
    ncircumferential1,
)

theta1 = 0.0
x1 = copy(fens_c1.xyz[:, 1])
y1 = copy(fens_c1.xyz[:, 2])

fens_c1.xyz[:, 1] .= cos(theta1) .* x1 .- sin(theta1) .* y1 .+ 0.5
fens_c1.xyz[:, 2] .= sin(theta1) .* x1 .+ cos(theta1) .* y1 .+ 0.5

fens1, fes1 = T4extrudeT3(
    fens_c1,
    fes_c1,
    nlayers1,
    (x, k) -> [x[1], x[2], k * depth / nlayers1],
)
bfes1 = meshboundary(fes1)

geom1 = NodalField(fens1.xyz)
u1 = NodalField(zeros(size(fens1.xyz)))
numberdofs!(u1)
femm1 = FEMMDeforLinear(MR, IntegDomain(fes1, TetRule(4)), material)
K1 = stiffness(femm1, geom1, u1)
K1_ff = matrix_blocked(K1, nfreedofs(u1), nfreedofs(u1))[:ff]


fil = ForceIntensity(Float64[t0, 0.0, 0.0])
fir = ForceIntensity(Float64[-t0, 0.0, 0.0])
l1e = subset(bfes1, selectelem(fens1, bfes1, box = [0.0,1.0,0.0,1.0,0.0,0.0], inflate=1e-8))
r1e = subset(bfes1, selectelem(fens1, bfes1, box = [0.0,1.0,0.0,1.0,depth,depth], inflate=1e-8))
femml1 = FEMMBase(IntegDomain(l1e, TriRule(3)))
femmr1 = FEMMBase(IntegDomain(r1e, TriRule(3)))
F1 = distribloads(femml1, geom1, u1, fil,2) +
     distribloads(femmr1, geom1, u1, fir,2)
F1_ff = vector_blocked(F1, nfreedofs(u1))[:f]

# ------------------------------------------------------------------
# Sector 2: 72 to 144 degrees
# ------------------------------------------------------------------
nradial2 = 4
ncircumferential2 = 4
nlayers2 = 4
fens_c2, fes_c2 = T3circleseg(
    sector_angle,
    inner_radius,
    nradial2,
    ncircumferential2,
)

theta2 = 1 * sector_angle
x2 = copy(fens_c2.xyz[:, 1])
y2 = copy(fens_c2.xyz[:, 2])

fens_c2.xyz[:, 1] .= cos(theta2) .* x2 .- sin(theta2) .* y2 .+ 0.5
fens_c2.xyz[:, 2] .= sin(theta2) .* x2 .+ cos(theta2) .* y2 .+ 0.5

fens2, fes2 = T4extrudeT3(
    fens_c2,
    fes_c2,
    nlayers2,
    (x, k) -> [x[1], x[2], k * depth / nlayers2],
)
bfes2 = meshboundary(fes2)

geom2 = NodalField(fens2.xyz)
u2 = NodalField(zeros(size(fens2.xyz)))
numberdofs!(u2)
femm2 = FEMMDeforLinear(MR, IntegDomain(fes2, TetRule(4)), material)

K2 = stiffness(femm2, geom2, u2)
K2_ff = matrix_blocked(K2, nfreedofs(u2), nfreedofs(u2))[:ff]
l2e = subset(bfes2, selectelem(fens2, bfes2, box = [0.0,1.0,0.0,1.0,0.0,0.0], inflate=1e-8))
r2e = subset(bfes2, selectelem(fens2, bfes2, box = [0.0,1.0,0.0,1.0,depth,depth], inflate=1e-8))
femml2 = FEMMBase(IntegDomain(l2e, TriRule(3)))
femmr2 = FEMMBase(IntegDomain(r2e, TriRule(3)))
F2 = distribloads(femml2, geom2, u2, fil,2) +
     distribloads(femmr2, geom2, u2, fir,2)
F2_ff = vector_blocked(F2, nfreedofs(u2))[:f]

# ------------------------------------------------------------------
# Sector 3: 144 to 216 degrees
# ------------------------------------------------------------------
nradial3 = 5
ncircumferential3 = 5
nlayers3 = 5
fens_c3, fes_c3 = T3circleseg(
    sector_angle,
    inner_radius,
    nradial3,
    ncircumferential3,
)

theta3 = 2 * sector_angle
x3 = copy(fens_c3.xyz[:, 1])
y3 = copy(fens_c3.xyz[:, 2])

fens_c3.xyz[:, 1] .= cos(theta3) .* x3 .- sin(theta3) .* y3 .+ 0.5
fens_c3.xyz[:, 2] .= sin(theta3) .* x3 .+ cos(theta3) .* y3 .+ 0.5

fens3, fes3 = T4extrudeT3(
    fens_c3,
    fes_c3,
    nlayers3,
    (x, k) -> [x[1], x[2], k * depth / nlayers3],
)
bfes3 = meshboundary(fes3)

geom3 = NodalField(fens3.xyz)
u3 = NodalField(zeros(size(fens3.xyz)))
numberdofs!(u3)
femm3 = FEMMDeforLinear(MR, IntegDomain(fes3, TetRule(4)), material)

K3 = stiffness(femm3, geom3, u3)
K3_ff = matrix_blocked(K3, nfreedofs(u3), nfreedofs(u3))[:ff]
l3e = subset(bfes3, selectelem(fens3, bfes3, box = [0.0,1.0,0.0,1.0,0.0,0.0], inflate=1e-8))
r3e = subset(bfes3, selectelem(fens3, bfes3, box = [0.0,1.0,0.0,1.0,depth,depth], inflate=1e-8))
femml3 = FEMMBase(IntegDomain(l3e, TriRule(3)))
femmr3 = FEMMBase(IntegDomain(r3e, TriRule(3)))
F3 = distribloads(femml3, geom3, u3, fil,2) +
     distribloads(femmr3, geom3, u3, fir,2)
F3_ff = vector_blocked(F3, nfreedofs(u3))[:f]
# ------------------------------------------------------------------
# Sector 4: 216 to 288 degrees
# ------------------------------------------------------------------
nradial4 = 4
ncircumferential4 = 4
nlayers4 = 4
fens_c4, fes_c4 = T3circleseg(
    sector_angle,
    inner_radius,
    nradial4,
    ncircumferential4,
)

theta4 = 3 * sector_angle
x4 = copy(fens_c4.xyz[:, 1])
y4 = copy(fens_c4.xyz[:, 2])

fens_c4.xyz[:, 1] .= cos(theta4) .* x4 .- sin(theta4) .* y4 .+ 0.5
fens_c4.xyz[:, 2] .= sin(theta4) .* x4 .+ cos(theta4) .* y4 .+ 0.5

fens4, fes4 = T4extrudeT3(
    fens_c4,
    fes_c4,
    nlayers4,
    (x, k) -> [x[1], x[2], k * depth / nlayers4],
)
bfes4 = meshboundary(fes4)

geom4 = NodalField(fens4.xyz)
u4 = NodalField(zeros(size(fens4.xyz)))
numberdofs!(u4)
femm4 = FEMMDeforLinear(MR, IntegDomain(fes4, TetRule(4)), material)
K4 = stiffness(femm4, geom4, u4)
K4_ff = matrix_blocked(K4, nfreedofs(u4), nfreedofs(u4))[:ff]

l4e = subset(bfes4, selectelem(fens4, bfes4, box = [0.0,1.0,0.0,1.0,0.0,0.0], inflate=1e-8))
r4e = subset(bfes4, selectelem(fens4, bfes4, box = [0.0,1.0,0.0,1.0,depth,depth], inflate=1e-8))
femml4 = FEMMBase(IntegDomain(l4e, TriRule(3)))
femmr4 = FEMMBase(IntegDomain(r4e, TriRule(3)))
F4 = distribloads(femml4, geom4, u4, fil,2) +
     distribloads(femmr4, geom4, u4, fir,2)
F4_ff = vector_blocked(F4, nfreedofs(u4))[:f]

# ------------------------------------------------------------------
# Sector 5: 288 to 360 degrees
# ------------------------------------------------------------------
nradial5 = 3
ncircumferential5 = 3
nlayers5 = 3
fens_c5, fes_c5 = T3circleseg(
    sector_angle,
    inner_radius,
    nradial5,
    ncircumferential5,
)

theta5 = 4 * sector_angle
x5 = copy(fens_c5.xyz[:, 1])
y5 = copy(fens_c5.xyz[:, 2])

fens_c5.xyz[:, 1] .= cos(theta5) .* x5 .- sin(theta5) .* y5 .+ 0.5
fens_c5.xyz[:, 2] .= sin(theta5) .* x5 .+ cos(theta5) .* y5 .+ 0.5

fens5, fes5 = T4extrudeT3(
    fens_c5,
    fes_c5,
    nlayers5,
    (x, k) -> [x[1], x[2], k * depth / nlayers5],
)
bfes5 = meshboundary(fes5)
if_fes5 = radial_interface_faces(fens5, fes5, theta5, sector_angle)

geom5 = NodalField(fens5.xyz)
u5 = NodalField(zeros(size(fens5.xyz)))
numberdofs!(u5)
femm5 = FEMMDeforLinear(MR, IntegDomain(fes5, TetRule(4)), material)

K5 = stiffness(femm5, geom5, u5)
K5_ff = matrix_blocked(K5, nfreedofs(u5), nfreedofs(u5))[:ff]
l5e = subset(bfes5, selectelem(fens5, bfes5, box = [0.0,1.0,0.0,1.0,0.0,0.0], inflate=1e-8))
r5e = subset(bfes5, selectelem(fens5, bfes5, box = [0.0,1.0,0.0,1.0,depth,depth], inflate=1e-8))
femml5 = FEMMBase(IntegDomain(l5e, TriRule(3)))
femmr5 = FEMMBase(IntegDomain(r5e, TriRule(3)))
F5 = distribloads(femml5, geom5, u5, fil,2) +
     distribloads(femmr5, geom5, u5, fir,2)
F5_ff = vector_blocked(F5, nfreedofs(u5))[:f]
# ------------------------------------------------------------------
# skel 1
# ------------------------------------------------------------------
nskelr =3
nskeld = 3

theta_i1 = 3 * sector_angle

rs_i1 = collect(linearspace(0.0, inner_radius, nskelr + 1))
zs_i1 = collect(linearspace(0.0, depth, nskeld + 1))

fens_i1, fes_i1 = T3blockx(rs_i1, zs_i1, :a)

r_i1 = copy(fens_i1.xyz[:, 1])
z_i1 = copy(fens_i1.xyz[:, 2])

fens_i1.xyz = hcat(
    0.5 .+ r_i1 .* cos(theta_i1),
    0.5 .+ r_i1 .* sin(theta_i1),
    z_i1,
)

geomi1 = NodalField(fens_i1.xyz)
if lam_order== 1
    u_i1 = NodalField(zeros(size(fens_i1.xyz)))
    numberdofs!(u_i1)
    femmi1 = FEMMDeforLinear(MR, IntegDomain(fes_i1, TriRule(4)), material)
    mi1 = mass(femmi1, geomi1, u_i1)
else
    u_i1 = ElementalField(zeros(size(fens_i1.conn, 1), 3))
    numberdofs!(u_i1)
    mi1 = p0_mass(fens_i1, fes_i1)
end

# ------------------------------------------------------------------
# skel 2
# ------------------------------------------------------------------
# Skeleton 2: circumference -> center -> circumference
# Rays at 0 and 144 degrees

theta_i2a = 0 * sector_angle
theta_i2b = 2 * sector_angle

ss_i2 = collect(linearspace(-inner_radius, inner_radius, 2 * nskelr + 1))
zs_i2 = collect(linearspace(0.0, depth, nskeld + 1))

fens_i2, fes_i2 = T3blockx(ss_i2, zs_i2, :a)

s_i2 = copy(fens_i2.xyz[:, 1])
z_i2 = copy(fens_i2.xyz[:, 2])

x_i2 = similar(s_i2)
y_i2 = similar(s_i2)

for i in eachindex(s_i2)
    if s_i2[i] <= 0.0
        r = -s_i2[i]
        x_i2[i] = 0.5 + r * cos(theta_i2a)
        y_i2[i] = 0.5 + r * sin(theta_i2a)
    else
        r = s_i2[i]
        x_i2[i] = 0.5 + r * cos(theta_i2b)
        y_i2[i] = 0.5 + r * sin(theta_i2b)
    end
end

fens_i2.xyz = hcat(x_i2, y_i2, z_i2)

geomi2 = NodalField(fens_i2.xyz)
if lam_order== 1
    u_i2 = NodalField(zeros(size(fens_i2.xyz)))
    numberdofs!(u_i2)
    femmi2 = FEMMDeforLinear(MR, IntegDomain(fes_i2, TriRule(4)), material)
    mi2 = mass(femmi2, geomi2, u_i2)
else
    u_i2 = ElementalField(zeros(size(fens_i2.conn, 1), 3))
    numberdofs!(u_i2)
    mi2 = p0_mass(fens_i2, fes_i2)
end

# ------------------------------------------------------------------
# skel 3
# ------------------------------------------------------------------
# Skeleton 3: circumference -> center -> circumference
# Rays at 72 and 288 degrees, giving a 144-degree smaller interior angle

theta_i3a = 1 * sector_angle
theta_i3b = 4 * sector_angle

ss_i3 = collect(linearspace(-inner_radius, inner_radius, 2 * nskelr + 1))
zs_i3 = collect(linearspace(0.0, depth, nskeld + 1))

fens_i3, fes_i3 = T3blockx(ss_i3, zs_i3, :a)

s_i3 = copy(fens_i3.xyz[:, 1])
z_i3 = copy(fens_i3.xyz[:, 2])

x_i3 = similar(s_i3)
y_i3 = similar(s_i3)

for i in eachindex(s_i3)
    if s_i3[i] <= 0.0
        r = -s_i3[i]
        x_i3[i] = 0.5 + r * cos(theta_i3a)
        y_i3[i] = 0.5 + r * sin(theta_i3a)
    else
        r = s_i3[i]
        x_i3[i] = 0.5 + r * cos(theta_i3b)
        y_i3[i] = 0.5 + r * sin(theta_i3b)
    end
end

fens_i3.xyz = hcat(x_i3, y_i3, z_i3)

geomi3 = NodalField(fens_i3.xyz)
if lam_order== 1
    u_i3 = NodalField(zeros(size(fens_i3.xyz)))
    numberdofs!(u_i3)
    femmi3 = FEMMDeforLinear(MR, IntegDomain(fes_i3, TriRule(4)), material)
    mi3 = mass(femmi3, geomi3, u_i3)
else
    u_i3 = ElementalField(zeros(size(fens_i3.conn, 1), 3))
    numberdofs!(u_i3)
    mi3 = p0_mass(fens_i3, fes_i3)
end


if_fes1s, if_fes1e = radial_interface_faces(fens1, fes1, theta1, sector_angle)
if_fes2s, if_fes2e = radial_interface_faces(fens2, fes2, theta2, sector_angle)
if_fes3s, if_fes3e = radial_interface_faces(fens3, fes3, theta3, sector_angle)
if_fes4s, if_fes4e = radial_interface_faces(fens4, fes4, theta4, sector_angle)
if_fes5s, if_fes5e = radial_interface_faces(fens5, fes5, theta5, sector_angle)

# Skeleton 1: 216 degrees
D13, meta13 = common_refinement(fens3, if_fes3e, fens_i1, fes_i1; lam_order=lam_order, tri_order=1, h=0.1, dim_u=3)
D14, meta14 = common_refinement(fens4, if_fes4s, fens_i1, fes_i1; lam_order=lam_order, tri_order=1, h=0.1, dim_u=3)

# Skeleton 2: 0 and 144 degrees
D21, meta21 = common_refinement(fens1, if_fes1s, fens_i2, fes_i2; lam_order=lam_order, tri_order=1, h=0.1, dim_u=3)
D25, meta25 = common_refinement(fens5, if_fes5e, fens_i2, fes_i2; lam_order=lam_order, tri_order=1, h=0.1, dim_u=3)
D22, meta22 = common_refinement(fens2, if_fes2e, fens_i2, fes_i2; lam_order=lam_order, tri_order=1, h=0.1, dim_u=3)
D23, meta23 = common_refinement(fens3, if_fes3s, fens_i2, fes_i2; lam_order=lam_order, tri_order=1, h=0.1, dim_u=3)

# Skeleton 3: 72 and 288 degrees
D31, meta31 = common_refinement(fens1, if_fes1e, fens_i3, fes_i3; lam_order=lam_order, tri_order=1, h=0.1, dim_u=3)
D32, meta32 = common_refinement(fens2, if_fes2s, fens_i3, fes_i3; lam_order=lam_order, tri_order=1, h=0.1, dim_u=3)
D34, meta34 = common_refinement(fens4, if_fes4e, fens_i3, fes_i3; lam_order=lam_order, tri_order=1, h=0.1, dim_u=3)
D35, meta35 = common_refinement(fens5, if_fes5s, fens_i3, fes_i3; lam_order=lam_order, tri_order=1, h=0.1, dim_u=3)

# direct solve
n1 = size(K1_ff, 1)
n2 = size(K2_ff, 1)
n3 = size(K3_ff, 1)
n4 = size(K4_ff, 1)
n5 = size(K5_ff, 1)

nl1 = size(D13, 1)
nl2 = size(D21, 1)
nl3 = size(D31, 1)

Z12 = spzeros(n1, n2); Z13 = spzeros(n1, n3); Z14 = spzeros(n1, n4); Z15 = spzeros(n1, n5)
Z23 = spzeros(n2, n3); Z24 = spzeros(n2, n4); Z25 = spzeros(n2, n5)
Z34 = spzeros(n3, n4); Z35 = spzeros(n3, n5)
Z45 = spzeros(n4, n5)

Z1l1 = spzeros(n1, nl1); Z2l1 = spzeros(n2, nl1); Z5l1 = spzeros(n5, nl1)
Z3l3 = spzeros(n3, nl3)
Zl12 = spzeros(nl1, nl2); Zl13 = spzeros(nl1, nl3); Zl23 = spzeros(nl2, nl3)

A = [
    K1_ff  Z12    Z13    Z14    Z15     Z1l1   D21'    D31'
    Z12'   K2_ff  Z23    Z24    Z25     Z2l1  -D22'   -D32'
    Z13'   Z23'   K3_ff  Z34    Z35     D13'   D23'    Z3l3
    Z14'   Z24'   Z34'   K4_ff  Z45    -D14'   spzeros(n4, nl2)  -D34'
    Z15'   Z25'   Z35'   Z45'   K5_ff   Z5l1  -D25'    D35'

    Z1l1'  Z2l1'  D13   -D14    Z5l1'   spzeros(nl1, nl1)  Zl12  Zl13
    D21   -D22     D23    spzeros(nl2, n4)  -D25  Zl12'  spzeros(nl2, nl2)  Zl23
    D31   -D32     Z3l3' -D34     D35   Zl13'  Zl23'  spzeros(nl3, nl3)
]

B = vcat(
    F1_ff,
    F2_ff,
    F3_ff,
    F4_ff,
    F5_ff,
    zeros(nl1 + nl2 + nl3),
)

println("rank A = $(rank(A))")
println("size A = $(size(A))")
error("")
# iterative solve

Ks = [K1_ff, K2_ff, K3_ff, K4_ff, K5_ff]
Fs = [F1_ff, F2_ff, F3_ff, F4_ff, F5_ff]

Ds = [
    [(3, +1.0, D13), (4, -1.0, D14)],

    [(1, +1.0, D21), (2, -1.0, D22),
     (3, +1.0, D23), (5, -1.0, D25)],

    [(1, +1.0, D31), (2, -1.0, D32),
     (4, -1.0, D34), (5, +1.0, D35)],
]

Ms = [mi1, mi2, mi3]

us, lambdas, X, stats = AL_solve(
    Ks,
    Fs,
    Ds,
    Ms;
    gamma=1.0,
    tau=1e-3,
    rtol=1e-8,
    atol=1e-10,
    itmax=5000,
)

scattersysvec!(u1, us[1])
scattersysvec!(u2, us[2])
scattersysvec!(u3, us[3])
scattersysvec!(u4, us[4])
scattersysvec!(u5, us[5])

scattersysvec!(u_i1, lambdas[1])
scattersysvec!(u_i2, lambdas[2])
scattersysvec!(u_i3, lambdas[3])

# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------
filename = basename(@__FILE__)
if !isdir(filename)
    mkdir(filename)
else
    for file in readdir(filename)
        rm(joinpath(filename, file))
    end
end
vtkexportmesh("$filename/1.vtk", fens1, fes1, vectors=[("u", u1.values)])
vtkexportmesh("$filename/2.vtk", fens2, fes2, vectors=[("u", u2.values)])
vtkexportmesh("$filename/3.vtk", fens3, fes3, vectors=[("u", u3.values)])
vtkexportmesh("$filename/4.vtk", fens4, fes4, vectors=[("u", u4.values)])
vtkexportmesh("$filename/5.vtk", fens5, fes5, vectors=[("u", u5.values)])

vtkexportmesh("$filename/skeleton1.vtk", fens_i1, fes_i1)
vtkexportmesh("$filename/skeleton2.vtk", fens_i2, fes_i2)
vtkexportmesh("$filename/skeleton3.vtk", fens_i3, fes_i3)