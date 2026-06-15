using FinEtools
using FinEtoolsDeforLinear
using FinEtoolsDeforLinear.AlgoDeforLinearModule
using FinEtools.MeshExportModule
using LinearAlgebra
using FinEtools.AlgoBaseModule: matrix_blocked, vector_blocked
using SparseArrays

function run_stress_concentration(r, m=2;
)
    println("Running with r = $r")

    mult = floor(Int, m^r)

    n_left_depth = 1*mult
    n_center_depth = 6 *mult
    n_right_depth = 1*mult
    n_interface1_depth = 2*mult
    n_interface2_depth = 2*mult

    lam_order = 0


    depth = 0.3
    rfillet = 0.05
    h_right = 0.5

    left_length = 0.8
    right_length = 0.8

    x0 = left_length
    x1 = x0 + 0.15
    x2 = x0 + 0.20
    x3 = x0 + 0.25
    x4 = x3 + right_length

    ymid = 0.5
    yR0 = ymid - h_right / 2       # 0.25
    yR1 = ymid + h_right / 2       # 0.75
    yL0 = yR0 - 2 * rfillet        # 0.15
    yL1 = yR1 + 2 * rfillet        # 0.85
    h_left_opening = yL1 - yL0     # 0.70


    # Convert the five controls above to target edge sizes.
    # Counts in each direction are then length/target-size, so elements are
    # roughly square-ish within each subdomain/interface.
    h_left_mesh = depth / n_left_depth
    h_center_mesh = depth / n_center_depth
    h_right_mesh = depth / n_right_depth
    h_interface1_mesh = depth / n_interface1_depth
    h_interface2_mesh = depth / n_interface2_depth

    nel(L, h; minval=1) = max(minval, ceil(Int, L / h))

    # Left subdomain counts
    nx_left_long = nel(x0, h_left_mesh)
    nx_left_short = nel(x1 - x0, h_left_mesh)
    # Use one cap count for both top and bottom. The top cap is created by
    # mirroring the bottom cap, so the left mesh is visually/geometrically
    # symmetric about y = 0.5.
    ny_left_cap = max(nel(yL0, h_left_mesh), nel(1.0 - yL1, h_left_mesh))
    ny_left_opening = nel(h_left_opening, h_left_mesh)
    nz_left = nel(depth, h_left_mesh)

    # Center subdomain counts
    nx_center_short = nel(x1 - x0, h_center_mesh)
    nx_center_web = nel(x3 - x2, h_center_mesh)
    ny_center_right = 2 * nx_center_short
    ny_center_fillet_strip = nel(rfillet, h_center_mesh)
    nz_center = nel(depth, h_center_mesh)
    nfillet = nel(rfillet, h_center_mesh; minval=2)

    # Right subdomain counts
    nx_right = nel(right_length, h_right_mesh)
    ny_right = nel(h_right, h_right_mesh)
    nz_right = nel(depth, h_right_mesh)

    # Skeleton/interface counts
    nx_i1_short = nel(x1 - x0, h_interface1_mesh)
    ny_i1_opening = nel(h_left_opening, h_interface1_mesh)
    nz_i1 = nel(depth, h_interface1_mesh)

    ny_i2 = nel(h_right, h_interface2_mesh)
    nz_i2 = nel(depth, h_interface2_mesh)

    # Spatial-grid sizes for common refinement search. Keep them tied to the
    # actual local interface resolution instead of using a fixed ad hoc value.
    h_cr1 = min(h_left_mesh, h_center_mesh, h_interface1_mesh)
    h_cr2 = min(h_center_mesh, h_right_mesh, h_interface2_mesh)

    E = 1.0
    nu = 1 / 3
    MR = DeforModelRed3D
    material = MatDeforElastIso(MR, 0.0, E, nu, 0.0)
    trule3d = TetRule(4)
    trule2d = TriRule(3)
    grule3d = GaussRule(3, 4)
    grule2d = GaussRule(2, 3)

    # Left skeleton/interface --------------------------------------------------
    fensi1_b, fesi1_b = T3block(x1 - x0, depth, nx_i1_short, nz_i1)
    fensi1_b.xyz = hcat(fensi1_b.xyz[:, 1] .+ x0,
                        yL0 * ones(size(fensi1_b.xyz, 1)),
                        fensi1_b.xyz[:, 2])

    fensi1_t, fesi1_t = T3block(x1 - x0, depth, nx_i1_short, nz_i1)
    fensi1_t.xyz = hcat(fensi1_t.xyz[:, 1] .+ x0,
                        yL1 * ones(size(fensi1_t.xyz, 1)),
                        fensi1_t.xyz[:, 2])

    fensi1_c, fesi1_c = T3block(depth, h_left_opening, nz_i1, ny_i1_opening)
    fensi1_c.xyz = hcat(x0 * ones(size(fensi1_c.xyz, 1)),
                        fensi1_c.xyz[:, 2] .+ yL0,
                        fensi1_c.xyz[:, 1])

    fensi1, fesi1out = mergenmeshes(Tuple{FENodeSet, AbstractFESet}[
        (fensi1_b, fesi1_b),
        (fensi1_t, fesi1_t),
        (fensi1_c, fesi1_c),
    ], 1e-8)
    fesi1 = FESetT3(vcat(connasarray.(fesi1out)...))

    # Right skeleton/interface -------------------------------------------------
    fensi2, fesi2 = T3block(depth, h_right, nz_i2, ny_i2)
    fensi2.xyz = hcat(x3 * ones(size(fensi2.xyz, 1)),
                      fensi2.xyz[:, 2] .+ yR0,
                      fensi2.xyz[:, 1])

    # Left subdomain -----------------------------------------------------------
    # Split bottom/top blocks at x0 so the notch interface x0--x1 is an
    # actual mesh line in the boundary mesh. Counts are based on hmesh.
    fens1bL, fes1bL = T3block(x0, yL0, nx_left_long, ny_left_cap)
    fens1bR, fes1bR = T3block(x1 - x0, yL0, nx_left_short, ny_left_cap)
    fens1bR.xyz[:, 1] .+= x0

    fens1c, fes1c = T3block(x0, h_left_opening, nx_left_long, ny_left_opening)
    fens1c.xyz[:, 2] .+= yL0

    # Build the top cap by mirroring the bottom cap. This keeps the left
    # subdomain symmetric without making it conform to the center subdomain.
    fens1tL, fes1tL = mirrormesh(fens1bL, fes1bL,
                                 [0.0, 1.0], [0.0, ymid],
                                 renumb=(c) -> c[[1, 3, 2]])
    fens1tR, fes1tR = mirrormesh(fens1bR, fes1bR,
                                 [0.0, 1.0], [0.0, ymid],
                                 renumb=(c) -> c[[1, 3, 2]])

    fens1_2d, output_fes1 = mergenmeshes(Tuple{FENodeSet, AbstractFESet}[
        (fens1bL, fes1bL),
        (fens1bR, fes1bR),
        (fens1c, fes1c),
        (fens1tL, fes1tL),
        (fens1tR, fes1tR),
    ], 1e-8)

    fes1_2d = FESetT3(vcat(connasarray.(output_fes1)...))
    fens1, fes1 = T4extrudeT3(fens1_2d, fes1_2d, nz_left,
                              (x, k) -> [x[1], x[2], k * depth / nz_left])

    geom1 = NodalField(fens1.xyz)
    u1 = NodalField(zeros(size(fens1.xyz, 1), 3))

    dbc_node1 = selectnode(fens1, box=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], inflate=1e-8)
    setebc!(u1, dbc_node1, 1, 0.0)
    setebc!(u1, dbc_node1, 2, 0.0)
    setebc!(u1, dbc_node1, 3, 0.0)

    dbc_node2 = selectnode(fens1, box=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0], inflate=1e-8)
    setebc!(u1, dbc_node2, 1, 0.0)

    dbc_node3 = selectnode(fens1, box=[0.0, 0.0, 0.0, 0.0, depth, depth], inflate=1e-8)
    setebc!(u1, dbc_node3, 1, 0.0)
    setebc!(u1, dbc_node3, 2, 0.0)

    dbc_nodes = sort([dbc_node1; dbc_node2; dbc_node3])
    applyebc!(u1)
    numberdofs!(u1)

    femm1 = FEMMDeforLinear(MR, IntegDomain(fes1, trule3d), material)
    K1 = stiffness(femm1, geom1, u1)
    K1_ff = matrix_blocked(K1, nfreedofs(u1), nfreedofs(u1))[:ff]
    K1_fd = matrix_blocked(K1, nfreedofs(u1), nfreedofs(u1))[:fd]

    mb1 = meshboundary(fes1)
    left_fes1 = subset(mb1, selectelem(fens1, mb1,
                                       box=[0.0, 0.0, 0.0, 1.0, 0.0, depth],
                                       inflate=1e-8))
    left_femm1 = FEMMBase(IntegDomain(left_fes1, trule2d))

    function fi_fun_l(forceout::Vector{T}, XYZ, tangents, feid, qpid) where {T}
        return [-1.0e-2, 0.0, 0.0]
    end
    fi_l = ForceIntensity(Float64, 3, fi_fun_l)
    F1 = distribloads(left_femm1, geom1, u1, fi_l, 2)
    F1_ff = vector_blocked(F1, nfreedofs(u1))[:f] - K1_fd * gathersysvec(u1, :d)

    ifes1b = subset(mb1, selectelem(fens1, mb1,
                                    box=[x0, x1, yL0, yL0, 0.0, depth],
                                    inflate=1e-8))
    ifes1t = subset(mb1, selectelem(fens1, mb1,
                                    box=[x0, x1, yL1, yL1, 0.0, depth],
                                    inflate=1e-8))
    ifes1c = subset(mb1, selectelem(fens1, mb1,
                                    box=[x0, x0, yL0, yL1, 0.0, depth],
                                    inflate=1e-8))
    @assert count(ifes1b) > 0 "left bottom interface selection is empty"
    @assert count(ifes1c) > 0 "left vertical interface selection is empty"
    @assert count(ifes1t) > 0 "left top interface selection is empty"
    ifes1 = FESetT3(vcat(connasarray(ifes1b), connasarray(ifes1c), connasarray(ifes1t)))

    D11, _ = common_refinement(fens1, ifes1, fensi1, fesi1;
                               lam_order=lam_order, h=h_cr1, dim_u=3)

    dbc_dofs = sort([3 * dbc_node1 .- 2;
                     3 * dbc_node1 .- 1;
                     3 * dbc_node1;
                     3 * dbc_node2 .- 2;
                     3 * dbc_node3;
                     3 * dbc_node3 .- 1])
    D11 = D11[:, setdiff(1:3 * count(fens1), dbc_dofs)]

    # Center subdomain ---------------------------------------------------------
    # Center counts use the same hmesh as the left/right subdomains.
    # This removes the previous ad hoc values N2_inc, 5*N2_inc, and 0.75*0.8.

    # Keep the original local fillet construction. Only the vertical placement
    # changes. With rfillet=0.05, the local top fillet before translation starts
    # at y=0.25; therefore the top shift below places it at yR1--(yR1+rfillet).
    fens2bb, fes2bb = Q4block(x1 - x0, rfillet, nx_center_short, ny_center_fillet_strip)
    fens2tt, fes2tt = Q4block(x1 - x0, rfillet, nx_center_short, ny_center_fillet_strip)

    fens_wh, fes_wh = Q4elliphole(rfillet, rfillet, 0.2, 0.3,
                                  nfillet, nfillet, nx_center_short)

    fens2b, fes2b = mirrormesh(fens_wh, fes_wh,
                               [1.0, 0.0], [0.1, 0.1],
                               renumb=(c) -> c[[1, 4, 3, 2]])

    fens2t, fes2t = mirrormesh(fens2b, fes2b,
                               [0.0, 1.0], [0.15, 0.15],
                               renumb=(c) -> c[[1, 4, 3, 2]])

    # Bottom/top straight strips.
    fens2bb.xyz[:, 1] .+= x0
    fens2bb.xyz[:, 2] .+= yL0

    fens2tt.xyz[:, 1] .+= x0
    fens2tt.xyz[:, 2] .+= yR1 + rfillet

    # Bottom fillet: yR0-rfillet -- yR0.
    fens2b.xyz[:, 1] .+= x0
    fens2b.xyz[:, 2] .+= yR0 - rfillet

    # Top fillet: yR1 -- yR1+rfillet.
    fens2t.xyz[:, 1] .+= x0
    fens2t.xyz[:, 2] .+= yR1 - (0.15 + 2 * rfillet)

    # Right vertical web of the center piece.
    fens2r, fes2r = Q4block(x3 - x2, h_right, nx_center_web, ny_center_right)
    fens2r.xyz[:, 1] .+= x2
    fens2r.xyz[:, 2] .+= yR0

    fens2_2d, output_fes2 = mergenmeshes(Tuple{FENodeSet, AbstractFESet}[
        (fens2b, fes2b),
        (fens2t, fes2t),
        (fens2r, fes2r),
        (fens2bb, fes2bb),
        (fens2tt, fes2tt),
    ], 1e-8)
    fes2_2d = FESetQ4(vcat(connasarray.(output_fes2)...))

    fens2, fes2 = H8extrudeQ4(fens2_2d, fes2_2d, nz_center,
                              (x, k) -> [x[1], x[2], k * depth / nz_center])

    geom2 = NodalField(fens2.xyz)
    u2 = NodalField(zeros(size(fens2.xyz, 1), 3))
    femm2 = FEMMDeforLinear(MR, IntegDomain(fes2, grule3d), material)
    numberdofs!(u2)
    K2 = stiffness(femm2, geom2, u2)
    K2_ff = matrix_blocked(K2, nfreedofs(u2), nfreedofs(u2))[:ff]
    F2 = zeros((nfreedofs(u2), 1))

    mb2 = meshboundary(fes2)
    ifes21c = subset(mb2, selectelem(fens2, mb2,
                                     box=[x0, x0, yL0, yL1, 0.0, depth],
                                     inflate=1e-8))
    ifes21b = subset(mb2, selectelem(fens2, mb2,
                                     box=[x0, x1, yL0, yL0, 0.0, depth],
                                     inflate=1e-8))
    ifes21t = subset(mb2, selectelem(fens2, mb2,
                                     box=[x0, x1, yL1, yL1, 0.0, depth],
                                     inflate=1e-8))
    @assert count(ifes21b) > 0 "center bottom-left interface selection is empty"
    @assert count(ifes21c) > 0 "center vertical-left interface selection is empty"
    @assert count(ifes21t) > 0 "center top-left interface selection is empty"
    ifes21 = FESetQ4(vcat(connasarray(ifes21b), connasarray(ifes21c), connasarray(ifes21t)))

    D21, _ = common_refinement(fens2, ifes21, fensi1, fesi1;
                               lam_order=lam_order, h=h_cr1, dim_u=3, tri_order=2)

    ifes22 = subset(mb2, selectelem(fens2, mb2,
                                    box=[x3, x3, yR0, yR1, 0.0, depth],
                                    inflate=1e-8))
    @assert count(ifes22) > 0 "center/right interface selection is empty"
    D22, _ = common_refinement(fens2, ifes22, fensi2, fesi2;
                               lam_order=lam_order, h=h_cr2, dim_u=3, tri_order=2)

    # Right subdomain ----------------------------------------------------------
    fens3, fes3 = T4block(right_length, h_right, depth,
                          nx_right, ny_right, nz_right)
    fens3.xyz[:, 1] .+= x3
    fens3.xyz[:, 2] .+= yR0

    geom3 = NodalField(fens3.xyz)
    u3 = NodalField(zeros(size(fens3.xyz, 1), 3))
    femm3 = FEMMDeforLinear(MR, IntegDomain(fes3, trule3d), material)
    numberdofs!(u3)
    K3 = stiffness(femm3, geom3, u3)
    K3_ff = matrix_blocked(K3, nfreedofs(u3), nfreedofs(u3))[:ff]

    mb3 = meshboundary(fes3)
    right_fes3 = subset(mb3, selectelem(fens3, mb3,
                                        box=[x4, x4, yR0, yR1, 0.0, depth],
                                        inflate=1e-8))
    right_femm3 = FEMMBase(IntegDomain(right_fes3, trule2d))

    # Balance total x force: left traction is -1e-2 over height 1.0; right face
    # height is 0.5, hence right traction is +2e-2.
    function fi_fun_r(forceout::Vector{T}, XYZ, tangents, feid, qpid) where {T}
        return [2.0e-2, 0.0, 0.0]
    end
    fi_r = ForceIntensity(Float64, 3, fi_fun_r)
    F3 = distribloads(right_femm3, geom3, u3, fi_r, 2)
    F3_ff = vector_blocked(F3, nfreedofs(u3))[:f]

    ifes32 = subset(mb3, selectelem(fens3, mb3,
                                    box=[x3, x3, yR0, yR1, 0.0, depth],
                                    inflate=1e-8))
    @assert count(ifes32) > 0 "right/center interface selection is empty"
    D32, _ = common_refinement(fens3, ifes32, fensi2, fesi2;
                               lam_order=lam_order, h=h_cr2, dim_u=3)

    # Assembly -----------------------------------------------------------------
    D = [D11 -D21 spzeros(size(D11, 1), size(D32, 2));
         spzeros(size(D22, 1), size(D11, 2)) -D22 D32]

    K = blockdiag(K1_ff, K2_ff, K3_ff)
    F = [F1_ff; F2; F3_ff; zeros(size(D, 1), 1)]
    A = [K D';
         D spzeros(size(D, 1), size(D, 1))]

    x = A \ F

    scattersysvec!(u1, x[1:nfreedofs(u1)])
    scattersysvec!(u2, x[nfreedofs(u1)+1:nfreedofs(u1)+nfreedofs(u2)])
    scattersysvec!(u3, x[nfreedofs(u1)+nfreedofs(u2)+1:nfreedofs(u1)+nfreedofs(u2)+nfreedofs(u3)])

    st1 = elemfieldfromintegpoints(femm1, geom1, u1, :vm, 1)
    st2 = elemfieldfromintegpoints(femm2, geom2, u2, :vm, 1)
    st3 = elemfieldfromintegpoints(femm3, geom3, u3, :vm, 1)

    println("max stress = ", maximum(st2.values))

    # Output -------------------------------------------------------------------
    filename = basename(@__FILE__)
    outdir = "$filename/r$r"
    if isdir(outdir)
        for file in readdir(outdir)
            rm(joinpath(outdir, file))
        end
    else
        mkpath(outdir)
    end

    vtkexportmesh(joinpath(outdir, "left.vtk"),
                  fens1, fes1,
                  scalars=[("St", st1.values)],
                  vectors=[("Displacement", u1.values)])

    vtkexportmesh(joinpath(outdir, "center.vtk"),
                  fens2, fes2,
                  scalars=[("St", st2.values)],
                  vectors=[("Displacement", u2.values)])

    vtkexportmesh(joinpath(outdir, "right.vtk"),
                  fens3, fes3,
                  scalars=[("St", st3.values)],
                  vectors=[("Displacement", u3.values)])

    vtkexportmesh(joinpath(outdir, "left_interface.vtk"), fensi1, fesi1)
    vtkexportmesh(joinpath(outdir, "right_interface.vtk"), fensi2, fesi2)

    return maximum(st2.values)
end

s = []
for r in 0:4
    max_stress = run_stress_concentration(r, 2)
    push!(s, max_stress)
end

# s = s[end-2:end]
# extrapolated_s = (s[2]^2 - s[3] * s[1]) / (2 * s[2] - s[1] - s[3])
# println("extrapolated s = ", extrapolated_s)
