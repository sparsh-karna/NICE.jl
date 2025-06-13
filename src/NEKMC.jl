using LinearAlgebra

"""
    simulate(rxn_system;
             n_iter=Int(1e+8), chunk_iter=Int(1e+4),
             ε=1.0e-4, ε_scale=1.0, ε_concs=0.0,
             tol_t=Inf, tol_ε=0.0, tol_concs=0.0)

Run a *N*et-*E*vent *K*inetic *M*onte *C*arlo (NEKMC) simulation to find the
equilibrium concentrations of the reaction system.
"""
function simulate(
    rxn_system::ReactionSystem;
    n_iter::Integer=Int(1e+8),
    chunk_iter::Integer=Int(1e+4),
    ε::Real=1.0e-4,
    ε_scale::Real=1.0,
    ε_concs::Real=0.0,
    tol_t::Real=Inf,
    tol_ε::Real=0.0,
    tol_concs::Real=0.0,
)
    # Allocate probability vector, Δconcs vector, Δt
    pvec = zeros(Float64, rxn_system.n_reaction)
    Δconcs = similar(rxn_system.concs)
    # Run simulation
    for _ in 1:chunk_iter:n_iter
        Δtime = 0.0
        for _ in 1:chunk_iter
            # Set Δconcs to concentration at step (n_iter - 1)
            Δconcs .= rxn_system.concs
            # Update rates of reaction
            update_rates(rxn_system)
            # Select and carry out random reaction
            i_rxn = select_reaction(rxn_system, pvec)
            Δtime += do_reaction(rxn_system, i_rxn, ε)
        end
        rxn_system.time += Δtime
        # Check for time convergence
        if Δtime > tol_t
            return :TimeConvergence
        end
        # Compute actual Δconcs at step (n_iter)
        Δconcs .-= rxn_system.concs
        norm_Δconcs = √(Δconcs · Δconcs)
        # Check for concentration convergence
        if norm_Δconcs < tol_concs
            return :ConcentrationConvergence
            # Check for decrease in concentration step size
        elseif norm_Δconcs < ε * ε_concs
            ε *= ε_scale
        end
        # Check for concentration step size convergence
        if ε < tol_ε
            return :StepSizeConvergence
        end
        rxn_system.n_iter += 1
    end
    :IterationLimit
end

function update_rates(
    rxn_system::ReactionSystem,
)
    for i in 1:rxn_system.n_reaction
        rev_rate = rxn_system.rev_rate_consts[i]
        fwd_rate = rxn_system.fwd_rate_consts[i]
        for j in 1:rxn_system.n_species
            s = rxn_system.stoich[j, i]
            if s >= 0.
                rev_rate *= rxn_system.concs[j]^s
            else
                # fwd_rate *= rxn_system.concs[j] ^ abs(s)
                fwd_rate *= rxn_system.concs[j]^(-s)
            end
        end
        rxn_system.rev_rates[i] = rev_rate
        rxn_system.fwd_rates[i] = fwd_rate
        rxn_system.net_rates[i] = fwd_rate - rev_rate
    end
end

function select_reaction(
    rxn_system::ReactionSystem,
    pvec::AbstractVector{Float64},
)
    # Update probability vector
    p = 0.
    for i in 1:rxn_system.n_reaction
        p += abs(rxn_system.net_rates[i])
        pvec[i] = p
    end
    # Select random reaction
    p *= rand(Float64)
    for i in 1:rxn_system.n_reaction
        if pvec[i] > p
            return i
        end
    end
    # sentinel return value
    rxn_system.n_reaction
end

function do_reaction(
    rxn_system::ReactionSystem,
    i_rxn::Integer,
    ε::Real,
)
    rate = rxn_system.net_rates[i_rxn]
    if rate >= 0
        for j = 1:rxn_system.n_species
            rxn_system.concs[j] += rxn_system.stoich[j, i_rxn] * ε
        end
        return ε / rate
    else
        for j = 1:rxn_system.n_species
            rxn_system.concs[j] -= rxn_system.stoich[j, i_rxn] * ε
        end
        return -ε / rate
    end
end

"""
    objective_function(rxn_system, xi)

Computes the objective function f_r(ξ) for each reaction r in the system.
Returns a vector of length R, where each element f_r represents the residual
of the equilibrium equation for reaction r.

Arguments:
- rxn_system::ReactionSystem: The reaction system with stoichiometry, initial concentrations, and equilibrium constants.
- xi::Vector{Float64}: The reaction progress vector ξ, length R.
"""
function objective_function(rxn_system::ReactionSystem, xi::Vector{Float64})
    R = rxn_system.n_reaction  # Number of reactions
    N = rxn_system.n_species   # Number of species
    f = zeros(Float64, R)      # Objective function values f_r

    # Compute updated activities: a_i = a_i^(0) + sum(ν_i^(s) * ξ^(s))
    activities = copy(rxn_system.concs)  # Initial activities (a_i^(0))
    for i in 1:N
        for s in 1:R
            activities[i] += rxn_system.stoich[i, s] * xi[s]
        end
    end

    # Compute f_r for each reaction
    for r in 1:R
        # First term: K^(r) * product over species where ν_i^(r) < 0
        term1 = rxn_system.keq_vals[r]
        for i in 1:N
            if rxn_system.stoich[i, r] < 0
                term1 *= activities[i]^(-rxn_system.stoich[i, r])
            end
        end

        # Second term: product over species where ν_i^(r) > 0
        term2 = 1.0
        for i in 1:N
            if rxn_system.stoich[i, r] > 0
                term2 *= activities[i]^rxn_system.stoich[i, r]
            end
        end

        f[r] = term1 - term2
    end

    return f
end

"""
    jacobian_function(rxn_system, xi)

Computes the Jacobian matrix J_rs = ∂f_r/∂ξ^(s) for the system of equations.
Returns an R×R matrix, where R is the number of reactions.

Arguments:
- rxn_system::ReactionSystem: The reaction system with stoichiometry, initial concentrations, and equilibrium constants.
- xi::Vector{Float64}: The reaction progress vector ξ, length R.
"""
function jacobian_function(rxn_system::ReactionSystem, xi::Vector{Float64})
    R = rxn_system.n_reaction
    N = rxn_system.n_species
    J = zeros(Float64, R, R)

    # Compute updated activities: a_i = a_i^(0) + sum(ν_i^(s) * ξ^(s))
    activities = copy(rxn_system.concs)
    for i in 1:N
        for s in 1:R
            activities[i] += rxn_system.stoich[i, s] * xi[s]
        end
    end

    # Compute Jacobian J_rs
    for r in 1:R
        for s in 1:R
            # First term: K^(r) * sum over species where ν_i^(r) < 0
            term1 = 0.0
            for i in 1:N
                if rxn_system.stoich[i, r] < 0
                    # Product over all j ≠ i
                    prod = 1.0
                    for j in 1:N
                        if j != i
                            if rxn_system.stoich[j, r] < 0
                                prod *= activities[j]^(-rxn_system.stoich[j, r])
                            end
                        end
                    end
                    # Compute the derivative term
                    term1 += prod * (-rxn_system.stoich[i, r]) * activities[i]^(-rxn_system.stoich[i, r] - 1) * rxn_system.stoich[i, s]
                end
            end
            term1 *= rxn_system.keq_vals[r]

            # Second term: sum over species where ν_i^(r) > 0
            term2 = 0.0
            for i in 1:N
                if rxn_system.stoich[i, r] > 0
                    # Product over all j ≠ i
                    prod = 1.0
                    for j in 1:N
                        if j != i
                            if rxn_system.stoich[j, r] > 0
                                prod *= activities[j]^rxn_system.stoich[j, r]
                            end
                        end
                    end
                    # Compute the derivative term
                    term2 += prod * rxn_system.stoich[i, r] * activities[i]^(rxn_system.stoich[i, r] - 1) * rxn_system.stoich[i, s]
                end
            end

            J[r, s] = term1 - term2
        end
    end

    return J
end