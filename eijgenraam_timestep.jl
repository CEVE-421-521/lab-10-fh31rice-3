# ============================================================================
# Timestepping Eijgenraam Dike Model (Lab 10)
# ============================================================================
#
# Annual-timestep reformulation of the Eijgenraam et al. (2014) dike heightening
# model, in the spirit of Garner & Keller (2018). Differences from Lab 8:
#
#   - Lab 8 used a single-heightening policy with an analytical expected-loss
#     integral.
#   - Lab 10 steps year-by-year so adaptive policies can react to observed
#     water levels.
#
# The only uncertainty in this lab is the **sea-level rise trajectory**,
# parameterized by a polynomial z_t = a + b t + c t² (following GK18 Eq. 5).
# Storm surge, the starting dike height, the discount rate, and the economic
# parameters are all FIXED so students can focus on one source of uncertainty
# at a time. Within a scenario, annual storm surges are still drawn randomly
# from a fixed Norfolk-VA-calibrated GEV (aleatory noise), so two scenarios
# with the same SLR coefficients will still differ in their year-to-year
# surge sequence.
#
# Policies:
#   - StaticDikePolicy   : heighten once at t=1 by u_heighten cm.
#   - AdaptiveDikePolicy : each year, if (dike height - water level) < buffer,
#                          heighten to restore the buffer + an extra freeboard.
#                          This is the 2-variable constant-DPS rule from GK18,
#                          capped at a physically plausible 100 cm/yr.
#
# References:
#   Eijgenraam et al. (2014), Interfaces 44(1):7-21.
#   Garner & Keller (2018), Environmental Modelling & Software 107:96-104.
# ============================================================================

using SimOptDecisions
using Distributions
using Random

# ----------------------------------------------------------------------------
# Ring 15 economic parameters (Eijgenraam et al. 2014). Matches Lab 8.
# ----------------------------------------------------------------------------

const RING15 = (
    c     = 125.6422,     # fixed investment cost (M€)
    b     = 1.1268,       # variable investment cost (M€/cm)
    lam   = 0.0098,       # investment cost exponent (1/cm)
    gamma = 0.035,        # economic growth rate
    rho   = 0.015,        # risk-free rate
    zeta  = 0.003764,     # loss growth per cm of heightening
)

# ----------------------------------------------------------------------------
# Config — ALL fixed (non-uncertain) parameters live here.
# ----------------------------------------------------------------------------

Base.@kwdef struct DikeConfig{T<:AbstractFloat} <: SimOptDecisions.AbstractConfig
    # Eijgenraam economics
    c::T         = RING15.c
    b::T         = RING15.b
    lam::T       = RING15.lam
    gamma::T     = RING15.gamma
    rho::T       = RING15.rho
    zeta::T      = 0.0   # Ring 15 value is 0.003764; zeroed out for this lab
                          # to remove the "growth in the protected area" term,
                          # which otherwise produces a non-monotone region
                          # where small heightenings increase damage.

    # Initial dike height (cm) and horizon (years)
    H0::T        = 250.0   # existing dike (cm) — adequate for typical year-1
                            # surges so the adaptive rule doesn't eat a
                            # year-1 flood just because of the info lag
    horizon::Int = 100

    # Fixed discount rate (GK18 uses 4%)
    discount_rate::T = 0.04

    # Fixed initial damage-per-flood (M€), mid-range from Lab 8
    V0::T        = 15_000.0

    # Fixed Norfolk-like GEV annual-max surge parameters (meters). Fit to
    # the Sewells Point, VA posterior return levels from the Bayesian GEV
    # used in the elevation-robustness study (see references.bib entry
    # doss-gollin_subjective:2023). Return levels at μ=0.9, σ=0.20, ξ=0.10:
    #   2-yr  ≈ 1.00 m (3.3 ft)
    #   10-yr ≈ 1.40 m (4.6 ft)
    #  100-yr ≈ 2.07 m (6.8 ft)
    #  500-yr ≈ 2.65 m (8.7 ft)
    mu_surge_m::T    = 0.9
    sigma_surge_m::T = 0.2
    xi_surge::T      = 0.10
end

# ----------------------------------------------------------------------------
# Scenario — the ONLY uncertain input is the water-level time series, driven
# by an uncertain polynomial SLR trajectory. Storm surge realizations within
# a scenario are aleatory (drawn from the fixed GEV in config).
# ----------------------------------------------------------------------------

SimOptDecisions.@scenariodef DikeScenario begin
    @timeseries water_levels          # cm, annual maximum water level
end

"""
Sample a single scenario: draw polynomial SLR coefficients `(a, b, c)` for
`z_t = a + b t + c t²`, then build a 100-year water-level trajectory as
MSL(t) + surge(t), where surges come from the fixed GEV in `config`.

The SLR coefficient distributions are chosen so that year-100 MSL spans
roughly 30–200 cm across scenarios, bracketing IPCC-AR5 projections.
"""
function sample_dike_scenario(rng::AbstractRNG, config::DikeConfig)
    # --- Polynomial SLR coefficients (cm, cm/yr, cm/yr²) ---
    a = rand(rng, Normal(0.0, 5.0))                                   # initial offset
    b = rand(rng, truncated(Normal(0.3, 0.1); lower=0.05))             # linear rate
    c = rand(rng, truncated(Normal(0.005, 0.003); lower=0.0))          # acceleration

    # --- Annual maximum surge from fixed GEV (meters → cm) ---
    surge_dist_m = GeneralizedExtremeValue(
        config.mu_surge_m, config.sigma_surge_m, config.xi_surge
    )

    water_levels = [
        a + b * t + c * t^2 + 100.0 * rand(rng, surge_dist_m)
        for t in 1:(config.horizon)
    ]

    return DikeScenario(; water_levels=water_levels)
end

# ----------------------------------------------------------------------------
# State and Action.
# ----------------------------------------------------------------------------

struct DikeState{T<:AbstractFloat} <: SimOptDecisions.AbstractState
    height_cm::T
    n_failures::Int
end

struct DikeAction{T<:AbstractFloat} <: SimOptDecisions.AbstractAction
    heighten_cm::T
end

# ----------------------------------------------------------------------------
# Outcome — per-scenario summary statistics.
# ----------------------------------------------------------------------------

SimOptDecisions.@outcomedef DikeOutcome begin
    @continuous investment_cost     # M€, discounted
    @continuous expected_damages    # M€, discounted
    @continuous reliability         # fraction of years without overtopping
end

# ----------------------------------------------------------------------------
# Policies.
# ----------------------------------------------------------------------------

# Static: heighten once at t=1. Bounds chosen to cover "do nothing" through
# "over-engineer". Use your recommended value from Lab 8 as a comparison point.
SimOptDecisions.@policydef StaticDikePolicy begin
    @continuous u_heighten 0.0 500.0   # up to 5 m
end

# Adaptive buffer/freeboard rule (2-variable constant DPS, the degenerate
# case of Garner & Keller's 10-variable formulation). Note: this specific
# parameterization is NOT a strict superset of the static policy class —
# the buffer/freeboard rule has no way to represent "do a one-shot year-1
# heightening and then stay put", so static and adaptive can each beat the
# other on different segments of the trade-off.
#
# Bounds are intentionally set to the range that actually produces
# non-dominated policies. The Eijgenraam cost function grows roughly
# exponentially with heightening, so values above ~400 cm push costs into
# the "overbuilt" region where every policy is dominated and the optimizer
# wastes evaluations.
SimOptDecisions.@policydef AdaptiveDikePolicy begin
    @continuous buffer    0.0 400.0   # up to 4 m
    @continuous freeboard 0.0 300.0   # up to 3 m
end

# ----------------------------------------------------------------------------
# Simulation callbacks.
# ----------------------------------------------------------------------------

SimOptDecisions.initialize(cfg::DikeConfig, ::DikeScenario, ::AbstractRNG) =
    DikeState(cfg.H0, 0)

SimOptDecisions.time_axis(cfg::DikeConfig, ::DikeScenario) = 1:(cfg.horizon)

function SimOptDecisions.get_action(
    policy::StaticDikePolicy, state::DikeState, t::SimOptDecisions.TimeStep, ::DikeScenario
)
    if SimOptDecisions.is_first(t)
        return DikeAction(SimOptDecisions.value(policy.u_heighten))
    end
    return DikeAction(0.0)
end

function SimOptDecisions.get_action(
    policy::AdaptiveDikePolicy,
    state::DikeState,
    t::SimOptDecisions.TimeStep,
    scenario::DikeScenario,
)
    # Realistic information lag: decisions for year `t` are made from the
    # observation at year `t-1`. You can't heighten the dike during the
    # storm you're observing. In year 1 there is no prior observation, so
    # no action is taken.
    SimOptDecisions.is_first(t) && return DikeAction(0.0)
    wl = SimOptDecisions.value(scenario.water_levels)
    y_prev = wl[t.t - 1]
    buf = SimOptDecisions.value(policy.buffer)
    fb  = SimOptDecisions.value(policy.freeboard)
    gap = state.height_cm - y_prev
    if gap < buf
        return DikeAction(max(0.0, (buf - gap) + fb))
    end
    return DikeAction(0.0)
end

"""
Eijgenraam-style investment cost to heighten the dike by `u` cm:
`(c + b u) exp(λ u)`. Zero if `u == 0`.

This matches the formula used in Lab 8, which depends only on the incremental
heightening `u` rather than the paper's `exp(λ (H + u))`. The Lab 8 form
offsets a unit inconsistency between the Eijgenraam (2014) Table 1 coefficients
and the Garner & Keller (2018) water-level units, and keeps per-event costs
in a reasonable range even as the dike accumulates over a long horizon.
"""
function eijgenraam_invest(u::Real, c::Real, b::Real, lam::Real)
    u <= 0 && return 0.0
    return (c + b * u) * exp(lam * u)
end

function SimOptDecisions.run_timestep(
    state::DikeState,
    action::DikeAction,
    t::SimOptDecisions.TimeStep,
    cfg::DikeConfig,
    scenario::DikeScenario,
    ::AbstractRNG,
)
    u = action.heighten_cm
    new_height = state.height_cm + u

    invest = eijgenraam_invest(u, cfg.c, cfg.b, cfg.lam)

    # Flood this year if water exceeds the (now-raised) dike.
    water = scenario.water_levels[t]
    flooded = water > new_height

    # Damage value grows with the protected economy (γ - ρ) and with prior
    # heightening (ζ). V0 sets the scale (fixed in the config).
    V_t = cfg.V0 * exp((cfg.gamma - cfg.rho) * t.t) *
          exp(cfg.zeta * (new_height - cfg.H0))
    damage = flooded ? V_t : 0.0

    new_state = DikeState(new_height, state.n_failures + (flooded ? 1 : 0))
    return (new_state, (investment=invest, damage=damage, flooded=flooded))
end

function SimOptDecisions.compute_outcome(
    step_records::Vector, cfg::DikeConfig, ::DikeScenario
)
    delta = cfg.discount_rate
    T = length(step_records)
    inv_disc = sum(step_records[t].investment * exp(-delta * t) for t in 1:T)
    dmg_disc = sum(step_records[t].damage      * exp(-delta * t) for t in 1:T)
    n_floods = sum(r.flooded for r in step_records)
    return DikeOutcome(;
        investment_cost  = inv_disc,
        expected_damages = dmg_disc,
        reliability      = 1.0 - n_floods / T,
    )
end
