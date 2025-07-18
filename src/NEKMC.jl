using LinearAlgebra

"""
    simulate(rxn_system;
             n_iter=Int(1e+8), ε=1.0e-4, ε_tol=1.0e-12, ε_mult=0.1,
             n_check=100, n_avg=100)

Run a *N*et-*E*vent *K*inetic *M*onte *C*arlo (NEKMC) simulation to find the
equilibrium concentrations of the reaction system.
"""

function simulate(
    rxn_system::ReactionSystem;
    n_iter::Integer=Int(1e+10), # A more reasonable default for testing
    ε_tol::Real=1.0e-12,
    min_concs_factor::Real=0.01 # Step size is a fraction of the smallest concentration
)
    # Working arrays
    rev_stoich = max.(rxn_system.stoich, 0.0)
    fwd_stoich = max.(-rxn_system.stoich, 0.0)
    pvec = zeros(Float64, rxn_system.n_reaction)

    rxn_system.n_iter = 0

    while rxn_system.n_iter < n_iter
        # --- FIX: Rates calculated at the START of the loop ---
        rxn_system.fwd_rates .= rxn_system.fwd_rate_consts .* vec(prod(rxn_system.concs .^ fwd_stoich; dims=1))
        rxn_system.rev_rates .= rxn_system.rev_rate_consts .* vec(prod(rxn_system.concs .^ rev_stoich; dims=1))
        rxn_system.net_rates .= rxn_system.fwd_rates .- rxn_system.rev_rates

        total_abs_rate = sum(abs.(rxn_system.net_rates))

        # --- FIX: Robust stopping condition ---
        if total_abs_rate < ε_tol
            break # System has reached equilibrium
        end

        # --- FIX: Stable, adaptive step-size (ε) calculation ---
        # The step size is limited by the smallest non-zero concentration.
        # This naturally prevents overshooting and creating negative concentrations.
        min_conc = Inf
        for c in rxn_system.concs
            if c > 1e-14 # Avoid using zero concentrations
                min_conc = min(min_conc, c)
            end
        end
        ε = min_conc * min_concs_factor

        # Select a reaction randomly, weighted by absolute net rate
        cumsum!(pvec, abs.(rxn_system.net_rates))
        i_rxn = searchsortedfirst(pvec, pvec[end] * rand(Float64))

        # Apply the reaction step
        direction = sign(rxn_system.net_rates[i_rxn])

        # Use a temporary vector for the proposed new concentrations
        concs_proposal = rxn_system.concs .+ (ε * direction) .* rxn_system.stoich[:, i_rxn]

        # Rejection sampling to ensure no negative concentrations
        if any(x -> x < 0, concs_proposal)
            # This step was too large, but the adaptive ε will shrink it next iteration.
            # We simply skip the concentration update for this step.
            rxn_system.n_iter += 1
            continue
        end
        rxn_system.concs .= concs_proposal

        # Update time using a standard KMC formula
        rxn_system.time += -log(rand(Float64)) / total_abs_rate

        rxn_system.n_iter += 1
    end
end