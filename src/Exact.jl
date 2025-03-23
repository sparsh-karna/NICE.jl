using LinearAlgebra
using SciMLBase
using ForwardDiff
using NonlinearSolve

"""
    solve(rxn_system, K_eqs; maxiters=1000, abstol=1.0e-9, reltol=0.0)

Solve the system deterministically.

This method solves a linear least squares problem to get an initial guess for
``\\mathbf{\\xi}``,
```math
{\\left(c_{mn}\\right)}^T \\left(\\xi_m^{\\left(0\\right)}\\right) =
    \\left(a_n - a_n^{\\left(0\\right)}\\right)
```
and then optimizes the equation for the simultaneous equilibria,
```math
K_\\text{eq}^{\\left(m\\right)} =
    \\prod_{n=1}^N {\\left( a_n^{\\left(0\\right)} +
    \\sum_{m=1}^M c_{mn} \\xi_m \\right)}^{c_{mn}}
```
using user-provided ``K_\\text{eq}`` values and the Newton's method + Trust
Region method. The function updates `rxn_system.concs` with the equilibrium
concentrations if the optimization succeeds, and returns a solution object
containing the optimized reaction extents and convergence details.
"""
function solve(
    rxn_system::ReactionSystem,
    K_eqs::AbstractVector{Float64};
    maxiters::Integer=1000,
    abstol::Real=1.0e-9,
    reltol::Real=0.0,
)
    # Define the objective function
    function f_K_eqs(f, ξ, _)
        for i in 1:rxn_system.n_reaction
            f_i1 = K_eqs[i]
            f_i2 = 1.0
            for j in 1:rxn_system.n_species
                # Reactants
                if rxn_system.stoich[j, i] < 0
                    f_i1 *= (
                        rxn_system.concs_init[j] + rxn_system.stoich[j, :] · ξ
                    )^-rxn_system.stoich[j, i]
                    # Products
                elseif rxn_system.stoich[j, i] > 0
                    f_i2 *= (
                        rxn_system.concs_init[j] + rxn_system.stoich[j, :] · ξ
                    )^rxn_system.stoich[j, i]
                end
            end
            f[i] = f_i1 - f_i2
        end
        nothing
    end
    # Compute reaction extents
    ξ = rxn_system.stoich \ (rxn_system.concs - rxn_system.concs_init)
    # Create objective function and Jacobian (automatic, with autodiff)
    problem = NonlinearSolve.NonlinearProblem(
        f_K_eqs, ξ;
        maxiters=maxiters, abstol=abstol, reltol=reltol,
    )
    # Run the nonlinear optimization
    solution = NonlinearSolve.solve(problem, NonlinearSolve.TrustRegion())
    # If nonlinear optimization was successful, update `rxn_system.concs`
    if SciMLBase.successful_retcode(solution)
        rxn_system.concs .= rxn_system.concs_init
        rxn_system.concs .+= rxn_system.stoich * solution.u
    end
    # Return the nonlinear solution
    solution
end

"""
    solve(rxn_system, K_eqs; max, φ=1.0, rate_consts=:forward, maxiters=1000, abstol=1.0e-9, reltol=0.0)

Solve the system deterministically.

This method solves a linear least squares problem to get an initial guess for
``\\mathbf{\\xi}``,
```math
{\\left(c_{mn}\\right)}^T \\left(\\xi_m^{\\left(0\\right)}\\right) =
    \\left(a_n - a_n^{\\left(0\\right)}\\right)
```
and then optimizes the equation for the simultaneous equilibria,
```math
K_\\text{eq}^{\\left(m\\right)} =
    \\prod_{n=1}^N {\\left( a_n^{\\left(0\\right)} +
    \\sum_{m=1}^M c_{mn} \\xi_m \\right)}^{c_{mn}}
```
using Newton's method + Trust Region. Instead of specifying the ``K_\\text{eq}``
values directly, they are approximated here using the forward or reverse rate
constants and a parameter ``\\phi``. If `:forward` is chosen, equilibrium constants
are derived as ``K_\\text{eq} = (k_f - 2\\phi)/(k_f - \\phi)``; if `:reverse`, as
``K_\\text{eq} = (k_r + \\phi)/k_r``. The function updates `rxn_system.concs`
upon successful convergence and returns a solution object.
"""
function solve(
    rxn_system::ReactionSystem;
    φ::Real=1.0,
    rate_consts::Symbol=:forward,
    maxiters::Integer=1000,
    abstol::Real=1.0e-9,
    reltol::Real=0.0,
)
    if rate_consts === :forward
        # Evaluate K_eqs from forward rate constants and φ_fwd
        K_eqs = (
            (rxn_system.fwd_rate_consts .- 2 * φ) ./
            (rxn_system.fwd_rate_consts - φ)
        )
    elseif rate_consts === :reverse
        # Evaluate K_eqs from reverse rate constants and φ_rev
        K_eqs = (rxn_system.rev_rate_consts .+ φ) ./ rxn_system.rev_rate_consts
    else
        throw(ArgumentError("invalid rate_consts value $rate_consts \
                             (must be :forward or :reverse)"))
    end
    solve(rxn_system, K_eqs; maxiters=maxiters, abstol=abstol, reltol=reltol)
end

"""
    hybrid_solve(rxn_system, K_eqs; n_iter=Int(1e+8), chunk_iter=Int(1e+4), ε=1.0e-4, ε_scale=1.0, ε_concs=0.0, tol_ε=0.0, maxiters=1000, abstol=1.0e-9, reltol=0.0)

Combines stochastic simulation and deterministic solving to find equilibrium
concentrations. First, it runs a Net-Event Kinetic Monte Carlo simulation via
`simulate` to approach equilibrium, then refines the result using the
deterministic `solve` method with provided ``K_\\text{eq}`` values. The function
updates `rxn_system.concs` and returns the final nonlinear solution object.
"""
function hybrid_solve(
    rxn_system::ReactionSystem,
    K_eqs::AbstractVector{Float64};
    n_iter::Integer=Int(1e+8),
    chunk_iter::Integer=Int(1e+4),
    ε::Real=1.0e-4,
    ε_scale::Real=1.0,
    ε_concs::Real=0.0,
    tol_ε::Real=0.0,
    maxiters::Integer=1000,
    abstol::Real=1.0e-9,
    reltol::Real=0.0,
)
    simulate(rxn_system; n_iter=n_iter, chunk_iter=chunk_iter, ε=ε, ε_scale=ε_scale, ε_concs=ε_concs, tol_ε=tol_ε)
    solve(rxn_system, K_eqs; maxiters=maxiters, abstol=abstol, reltol=reltol)
end