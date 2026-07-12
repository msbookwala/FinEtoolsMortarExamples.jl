using Plots
using LaTeXStrings
using Measures
using Printf

# ------------------------------------------------------------
# Data: stress concentration example
# ------------------------------------------------------------

h = [
    0.04,
    0.02,
    0.014142136,
    0.01,
    0.007071068,
    0.005,
    # 0.003535534,
]

sigma_max = [
    0.110845979,
    0.123534793,
    0.129699214,
    0.133800190,
    0.137198274,
    0.139048756,
    # 0.139412212,
]

# ------------------------------------------------------------
# Richardson extrapolation from last 3 values
# sigma_(h) = sigma__ext + C h^p
# ------------------------------------------------------------

h1, h2, h3 = h[end-2:end]
sigma_1, sigma_2, sigma_3 = sigma_max[end-2:end]

r = h1 / h2
p = log((sigma_1 - sigma_2) / (sigma_2 - sigma_3)) / log(r)
sigma_ext = sigma_3 + (sigma_3 - sigma_2) / (r^p - 1)

println("Estimated convergence order p = ", p)
println("Richardson extrapolated stress = ", sigma_ext)

# Absolute and normalized errors
err_abs = abs.(sigma_ext .- sigma_max)
err_norm = err_abs ./ abs(sigma_ext)

# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------

default(
    fontfamily = "Computer Modern",
    linewidth = 2.5,
    markersize = 6,
    framestyle = :box,
    grid = true,
    minorgrid = true,
    legendfontsize = 10,
    tickfontsize = 10,
    guidefontsize = 12,
    titlefontsize = 12,
    dpi = 300,
)

# ------------------------------------------------------------
# 1. Normalized error, log-log
# ------------------------------------------------------------



# Optional reference slope using the Richardson-estimated order
href = h[end-2:end]
eref = err_norm[end-2] .* (href ./ href[1]).^p

# plot!(
#     plt1,
#     href,
#     eref;
#     linestyle = :dash,
#     marker = :none,
#     label = latexstring(@sprintf("slope %.2f", p)),
# )

plt1 = plot(
    h,
    err_norm;
    xscale = :log10,
    yscale = :log10,
    marker = :circle,
    label = "Normalized error, p = $(round(p, digits=2))",
    xlabel = L"Element size, $h$",
    ylabel = "Normalized error",
    title = "Error convergence",
    xflip = true,
    margin = 5mm,
    legend = :bottomleft
)

savefig(plt1, "stress_conc_error_convergence.pdf")

# ------------------------------------------------------------
# 2. Max stress convergence to extrapolated value
# ------------------------------------------------------------

plt2 = plot(
    h,
    sigma_max;
    xscale = :log10,
    marker = :circle,
    label = L"\sigma_{\max,h}",
    xlabel = L"Element size, $h$",
    ylabel = L"Maximum principal stress, $\sigma_{\max}$",
    title = "Convergence of maximum stress",
    xflip = true,
    margin = 5mm,
    legend = :bottomright
)

hline!(
    plt2,
    [sigma_ext];
    linestyle = :dot,
    linewidth = 2.5,
    label = latexstring(@sprintf("\\sigma_{\\mathrm{ext}}")),
)

savefig(plt2, "stress_conc_max_convergence.pdf")