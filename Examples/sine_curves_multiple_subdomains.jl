using FinEtools
using FinEtoolsDeforLinear
using FinEtoolsDeforLinear.AlgoDeforLinearModule
using FinEtools.MeshExportModule
import LinearAlgebra: cholesky
using FinEtools.AlgoBaseModule: matrix_blocked, vector_blocked
using SparseArrays

# check if folder exists
filename = basename(@__FILE__)
if !isdir(filename)
    mkdir(filename)
else
    for file in readdir(filename)
        rm(joinpath(filename, file))
    end
end


E = 1.0
nu = 1/3
MR = DeforModelRed3D
material = MatDeforElastIso(MR, 0.0, E, nu, 0.0)

lam_orders = [1,1,1]
bends = [0.1, 0.1, 0.1]*0
function bend_map(xyz, bends)
    X0 = copy(xyz)   # important: always use original/reference coordinates

    s = sin.(pi .* X0[:, 1]) .* sin.(pi .* X0[:, 2]) .* sin.(pi .* X0[:, 3])

    xyz[:, 1] .= X0[:, 1] .+ bends[1] .* s
    xyz[:, 2] .= X0[:, 2] .+ bends[2] .* s
    xyz[:, 3] .= X0[:, 3] .+ bends[3] .* s

    return xyz
end

alpha0, alpha1, alpha2, alpha3, beta0, beta1, beta2, beta3, gamma0, gamma1, gamma2, gamma3 =
    1.0 / 30, 1.0 / 34, -1.0 / 21, -1.0 / 51, -1.0 / 26, -1.0 / 35, 1.0 / 29, -1.0 / 31, -1.0 / 27, -1.0 / 33, -1.0 / 28, 1.0 / 32
ux(x, y, z) = alpha0 + alpha1 * x + alpha2 * y + alpha3 * z
uy(x, y, z) = beta0 + beta1 * x + beta2 * y + beta3 * z
uz(x, y, z) = gamma0 + gamma1 * x + gamma2 * y + gamma3 * z
exact_u(x) = [ux(x[1], x[2], x[3]), uy(x[1], x[2], x[3]), uz(x[1], x[2], x[3])]

boundary_boxes = [[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], #bottom
                  [0.0, 1.0, 1.0, 1.0, 0.0, 1.0], #top
                  [0.0, 0.0, 0.0, 1.0, 0.0, 1.0], #left
                  [1.0, 1.0, 0.0, 1.0, 0.0, 1.0], #right
                  [0.0, 1.0, 0.0, 1.0, 0.0, 0.0], #back
                  [0.0, 1.0, 0.0, 1.0, 1.0, 1.0], #front
                  ]

interface_boxes = [[0.5, 0.5, 0.0, 1.0, 0.0, 1.0], # x_normal
                   [0.0, 1.0, 0.5, 0.5, 0.0, 1.0], # y_normal
                   [0.0, 1.0, 0.0, 1.0, 0.5, 0.5], # z_normal
                   ]

# ####### interface meshes #####################################

nelem_is = [5,4,6]
perumutes = [[1,2,3], #x normal
            [2,1,3], #y normal
             [3,2,1]] #z normal
fes_is = []
fens_is = []
u_is = []
f_lams = []
for i in 1:3
    i_linspace = collect(linearspace(0.0, 1.0, nelem_is[i]+1))
    
    fens_i, fes_i = T3blockx(i_linspace, i_linspace, :a)
    c_linspace = 0.5*ones(size(fens_i.xyz, 1), 1)
    fens_i.xyz = hcat(c_linspace, fens_i.xyz)
    fens_i.xyz = fens_i.xyz[:, perumutes[i]]

    bend_map(fens_i.xyz, bends) 

    push!(fes_is, fes_i)
    push!(fens_is, fens_i)
    if lam_orders[i] == 0
        u_i  = ElementalField(zeros(count(fes_i), 3)) # Lagrange multipliers field
    else
        u_i  = NodalField(zeros(size(fens_i.xyz, 1), 3  )) # Lagrange multipliers field
    end
    numberdofs!(u_i)
    push!(u_is, u_i)
    push!(f_lams, zeros(nfreedofs(u_i)))

    fn = "$filename/interface_$(i).vtk"
    # vtkexportmesh(
    #     fn,
    #     fens_i,
    #     fes_i;
    #     scalars = [
    #     ],
    #     vectors = [
    #     ]
    # )
end
###################################################


N_elemes = [3,4,5,6,7,8,9,10]
# N_elemes = [20,20,20,20,20,20,20,20]
types = ["t", "t", "t", "t", "t", "t", "t", "t"]
index = 1
dbc_nodes_list = []
fens_list = []
fes_list = []
u_list = []
geom_list = []
bondaryfes_list = []
interface_fes_list = []
K_ff_list = []
F_ff_list = []
D_list_list = []
meta_list_list = []
femm_list = []
D_signs = [[-1, 1, -1, 1, -1, 1, -1, 1], # x normal
           [-1, -1, 1, 1, -1, -1, 1, 1], # y normal
           [-1,-1, -1, -1, 1, 1, 1, 1]] # z normal
for zi in 1:2
    for yi in 1:2
        for xi in 1:2
            
            print(index)
            if types[index] == "h"
                fens, fes = H8block(0.5,0.5,0.5, N_elemes[index], N_elemes[index], N_elemes[index])
                Rule = GaussRule(3,2)
                Rule2D = GaussRule(2,2)
            else
                fens, fes = T4block(0.5, 0.5, 0.5, N_elemes[index], N_elemes[index], N_elemes[index])
                Rule = TetRule(4)
                Rule2D = TriRule(3)
            end
            fens.xyz[:, 1] .+= (xi-1)*0.5
            fens.xyz[:, 2] .+= (yi-1)*0.5
            fens.xyz[:, 3] .+= (zi-1)*0.5

            

            boundaryfes = meshboundary(fes)
            dbc_nodes = []
            for box in boundary_boxes
                append!(dbc_nodes, selectnode(fens; box=box, inflate=1e-8))
            end
            dbc_nodes = sort(unique(dbc_nodes))

            interface_fes_vv = []
            for box in interface_boxes
                push!(interface_fes_vv, subset(boundaryfes, selectelem(fens, boundaryfes, box=box, inflate=1e-8)))
            end
            interface_fes = vcat(interface_fes_vv...)

            # xyz = deepcopy(fens.xyz)
            
            # fens.xyz+=  stack([bends[1]*sin.(pi.*xyz[:,1]).*sin.(pi.*xyz[:,2]).*sin.(pi.*xyz[:,3]),
            #                    bends[2]*sin.(pi.*xyz[:,1]).*sin.(pi.*xyz[:,2]).*sin.(pi.*xyz[:,3]),
            #                    bends[3]*sin.(pi.*xyz[:,1]).*sin.(pi.*xyz[:,2]).*sin.(pi.*xyz[:,3])])

            # fens.xyz[:,1]+=bends[1]*sin.(pi.*fens.xyz[:,1]).*sin.(pi.*fens.xyz[:,2]).*sin.(pi.*fens.xyz[:,3])
            # fens.xyz[:,2]+=bends[2]*sin.(pi.*fens.xyz[:,1]).*sin.(pi.*fens.xyz[:,2]).*sin.(pi.*fens.xyz[:,3])
            # fens.xyz[:,3]+=bends[3]*sin.(pi.*fens.xyz[:,1]).*sin.(pi.*fens.xyz[:,2]).*sin.(pi.*fens.xyz[:,3])

            bend_map(fens.xyz, bends)
            geom = NodalField(fens.xyz)
            u = NodalField(zeros(size(fens.xyz, 1), 3)) # displacement field
            for i in dbc_nodes
                setebc!(u, [i], 1, ux(fens.xyz[i, :]...))
                setebc!(u, [i], 2, uy(fens.xyz[i, :]...))
                setebc!(u, [i], 3, uz(fens.xyz[i, :]...))
            end
            applyebc!(u)
            numberdofs!(u)  
            femm = FEMMDeforLinear(MR, IntegDomain(fes, Rule), material)
            K = stiffness(femm, geom, u)
            K_ff = matrix_blocked(K, nfreedofs(u), nfreedofs(u))[:ff]
            K_fd = matrix_blocked(K, nfreedofs(u), nfreedofs(u))[:fd]
            F = zeros(size(K, 1))
            F_ff = vector_blocked(F, nfreedofs(u))[:f] - K_fd * gathersysvec(u, :d)



            D_list = []
            meta_list = []
            for i in 1:3
                
                D, meta = common_refinement(fens, interface_fes[i], fens_is[i], fes_is[i]; lam_order=lam_orders[i], h=0.3, dim_u=3)
                D = D_signs[i][index] * D
                dbc_dofs = sort([3*dbc_nodes.-2; 3*dbc_nodes.-1; 3*dbc_nodes])
                global f_lams[i] += -(D[:, dbc_dofs] * gathersysvec(u, :d))
                # push!(f_lams, -D[:, dbc_dofs] * gathersysvec(u, :d))
                D = D[:, setdiff(1:3*count(fens), dbc_dofs)]
                push!(D_list, D)
                # push!(meta_list, meta)

                # filename = "curved_mult_sd/union_$(index)_$(i).vtk"
                # vtkexportmesh(
                #     filename,
                #     meta["fens_u"],
                #     meta["fes_u"];
                #     scalars = [
                #     ],
                #     vectors = [
                #     ]
                # )
            end

            push!(dbc_nodes_list, dbc_nodes)
            push!(fens_list, fens)
            push!(fes_list, fes)
            push!(u_list, u)
            push!(geom_list, geom)
            push!(interface_fes_list, interface_fes)
            push!(bondaryfes_list, boundaryfes)
            push!(K_ff_list, K_ff)
            push!(F_ff_list, F_ff)
            push!(D_list_list, D_list)
            push!(meta_list_list, meta_list)
            push!(femm_list, femm)
            global index += 1

        end
    end
end

K_mat = []
for i in 1:8
    row = []
    for j in 1:8
        if i == j
            push!(row, K_ff_list[i])
        else
            push!(row, spzeros(Float64, size(K_ff_list[i],1), size(K_ff_list[j],2)))

        end
    end
    row = hcat(row...)
    push!(K_mat, row)
end
K_mat = vcat(K_mat...)

D_mat = []
for i in 1:3
    row = []
    for j in 1:8
            push!(row, D_list_list[j][i])
    end
    row = hcat(row...)
    push!(D_mat, row)
end

D_mat = vcat(D_mat...)


f_lams_vec  = vcat(f_lams...)
F_vec = vcat(F_ff_list...)

A = [K_mat  D_mat';
     D_mat  spzeros(Float64, size(f_lams_vec, 1), size(f_lams_vec, 1))]
b = [vcat(F_ff_list...);
     f_lams_vec]
# X,_ = cg(A, b)
X = A\b

offset = 0
for i in 1:8
    ndofs_j = size(K_ff_list[i],1)
    scattersysvec!(u_list[i], X[offset+1:offset+ndofs_j])
    global offset += ndofs_j
    
    err = L2error(femm_list[i], geom_list[i], u_list[i], exact_u)

    fn = "$filename/mesh_$(i).vtk"
            vtkexportmesh(
                fn,
                fens_list[i],
                fes_list[i];
                scalars = [
                     ("Err", err.values)
                ],
                vectors = [ ("u", u_list[i].values)
                ]
            )
            

end

for i in 1:3
    ndofs_i = size(D_list_list[1][i], 1)
    u_i = u_is[i]
    scattersysvec!(u_i, X[offset+1:offset+ndofs_i])
    global offset += ndofs_i

    fn = "$filename/mesh_interface_$(i).vtk"
            vtkexportmesh(
                fn,
                fens_is[i],
                fes_is[i];
                scalars = [
                ],
                vectors = [
                    ("u", u_i.values)
                ]
            )
end
