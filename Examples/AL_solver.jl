using LinearAlgebra
using SparseArrays
using LinearOperators
using Krylov
using IncompleteLU: ilu
using AlgebraicMultigrid
function blockdiag_sparse(Ks)
    n = sum(size(K, 1) for K in Ks)
    I = Int[]
    J = Int[]
    V = Float64[]
    off = 0
    for K in Ks
        i, j, v = findnz(K)
        append!(I, i .+ off)
        append!(J, j .+ off)
        append!(V, v)
        off += size(K, 1)
    end
    return sparse(I, J, V, n, n)
end

function AL_solve(Ks, Fs, Ds, Ms, Gs=nothing;
                  gamma=1.0,
                  tau=1e-3,
                  rtol=1e-8,
                  atol=1e-10,
                  itmax=5000)

    nb = length(Ks)
    ns = [size(K, 1) for K in Ks]
    offs = cumsum([0; ns[1:end-1]])

    K0 = blockdiag_sparse(Ks)
    F = reduce(vcat, Fs)

    # Ds[s] is vector of (block_id, sign, Dblock)
    Drows = Int[]
    Dcols = Int[]
    Dvals = Float64[]

    lam_offsets = cumsum([0; [size(M, 1) for M in Ms][1:end-1]])
    nlam = sum(size(M, 1) for M in Ms)
    if Gs === nothing
        G = zeros(nlam)
    else
        G = reduce(vcat, Gs)
    end

    for s in eachindex(Ds)
        for (bid, sgn, Db) in Ds[s]
            i, j, v = findnz(Db)
            append!(Drows, i .+ lam_offsets[s])
            append!(Dcols, j .+ offs[bid])
            append!(Dvals, sgn .* v)
        end
    end

    D = sparse(Drows, Dcols, Dvals, nlam, size(K0, 1))

    Winvs = SparseMatrixCSC[]
    for M in Ms
        mlump = vec(sum(M, dims=2))
        push!(Winvs, spdiagm(0 => 1.0 ./ mlump.^2))
    end
    Winv = blockdiag_sparse(Winvs)

    K = K0 + gamma * D' * Winv * D
    K = sparse(0.5 * (K + K'))
    print("Computing ILU preconditioner...\n")
    FK = ilu(K, τ=tau)
    print("Done.\n")

    nT = size(K, 1)
    n = nT + nlam

    function Pinv_mul!(y, x)
        xu = @view x[1:nT]
        xl = @view x[nT+1:n]

        yu = @view y[1:nT]
        yl = @view y[nT+1:n]

        yl_tmp = -gamma * (Winv * xl)
        rhsu = xu - D' * yl_tmp

        yu .= FK \ rhsu
        yl .= yl_tmp

        return y
    end

    Pinv = LinearOperator(Float64, n, n, false, false, Pinv_mul!)

    A = [
        K D'
        D spzeros(nlam, nlam)
    ]

    Fhat = F + gamma * D' * Winv * G
    B = vcat(Fhat, G)

    X, stats = gmres(A, B; M=Pinv, rtol=rtol, atol=atol, itmax=itmax, verbose=1)

    us = Vector{Vector{Float64}}()
    for i in eachindex(Ks)
        push!(us, X[offs[i]+1 : offs[i]+ns[i]])
    end

    lambdas = Vector{Vector{Float64}}()
    for s in eachindex(Ms)
        nls = size(Ms[s], 1)
        push!(lambdas, X[nT+lam_offsets[s]+1 : nT+lam_offsets[s]+nls])
    end

    return us, lambdas, X, stats
end

function AL_solve_cg(Ks, Fs, Ds, Ms, Gs=nothing;
                  gamma=10.0,
                  tau=1e-3,
                  rtol=1e-8,
                  atol=1e-10,
                  itmax=5000)

    nb = length(Ks)
    ns = [size(K, 1) for K in Ks]
    offs = cumsum([0; ns[1:end-1]])

    K0 = blockdiag_sparse(Ks)
    F = reduce(vcat, Fs)

    # Ds[s] is vector of (block_id, sign, Dblock)
    Drows = Int[]
    Dcols = Int[]
    Dvals = Float64[]

    lam_offsets = cumsum([0; [size(M, 1) for M in Ms][1:end-1]])
    nlam = sum(size(M, 1) for M in Ms)
    if Gs === nothing
        G = zeros(nlam)
    else
        G = reduce(vcat, Gs)
    end

    for s in eachindex(Ds)
        for (bid, sgn, Db) in Ds[s]
            i, j, v = findnz(Db)
            append!(Drows, i .+ lam_offsets[s])
            append!(Dcols, j .+ offs[bid])
            append!(Dvals, sgn .* v)
        end
    end

    D = sparse(Drows, Dcols, Dvals, nlam, size(K0, 1))

    Winvs = SparseMatrixCSC[]
    for M in Ms
        mlump = vec(sum(M, dims=2))
        push!(Winvs, spdiagm(0 => 1.0 ./ mlump.^2))
    end
    Winv = blockdiag_sparse(Winvs)

    K = K0 + gamma * D' * Winv * D
    K = sparse(0.5 * (K + K'))

    # FK = ilu(K, τ=tau)

    nT = size(K, 1)
    n = nT + nlam
    ml = smoothed_aggregation(K)
    Pamg = aspreconditioner(ml)

    function Pinv_mul!(y, x)
        xu = @view x[1:nT]
        xl = @view x[nT+1:n]

        yu = @view y[1:nT]
        yl = @view y[nT+1:n]

        yl_tmp = -gamma * (Winv * xl)
        rhsu = xu - D' * yl_tmp

        yu .= cg(K, rhsu; M=Pamg, ldiv=true, rtol=1e-2, atol=0.0, itmax=10)[1]
        yl .= yl_tmp

        return y
    end

    Pinv = LinearOperator(Float64, n, n, false, false, Pinv_mul!)

    A = [
        K D'
        D spzeros(nlam, nlam)
    ]

    # B = vcat(F, G)
    Fhat = F + gamma * D' * Winv * G
    B = vcat(Fhat, G)

    X, stats = gmres(A, B; M=Pinv, rtol=rtol, atol=atol, itmax=itmax, verbose=1)

    us = Vector{Vector{Float64}}()
    for i in eachindex(Ks)
        push!(us, X[offs[i]+1 : offs[i]+ns[i]])
    end

    lambdas = Vector{Vector{Float64}}()
    for s in eachindex(Ms)
        nls = size(Ms[s], 1)
        push!(lambdas, X[nT+lam_offsets[s]+1 : nT+lam_offsets[s]+nls])
    end

    return us, lambdas, X, stats
end

