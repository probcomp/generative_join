import jax.numpy as jnp
import jax
import genjax
import polars as pl
import numpy as np
from itertools import product
from jax.scipy.special import logit
from genjax import static_gen_fn
from generative_join.modeling import (
    aggregate_mean,
    importance_sample,
    estimate_mu_std,
    make_conditions_dict,
    estimate_logprobs,
)


@static_gen_fn
def gen_parametrized_z(
    male_p,
    high_life_expectancy_p,
    connected_p,
    supports_medicare_p,
    vote_republican_p,
    high_income_p,
    support_high_taxes_p,
    attended_elite_college_p,
    region_logits,
):

    male = genjax.bernoulli(logit(male_p)) @ "male"
    high_life_expectancy = (
        genjax.bernoulli(logit(high_life_expectancy_p)) @ "high_life_expectancy"
    )
    connected = genjax.bernoulli(logit(connected_p)) @ "connected"
    supports_medicare = (
        genjax.bernoulli(logit(supports_medicare_p)) @ "supports_medicare"
    )
    vote_republican = genjax.bernoulli(logit(vote_republican_p)) @ "vote_republican"
    high_income = genjax.bernoulli(logit(high_income_p)) @ "high_income"
    support_high_taxes = (
        genjax.bernoulli(logit(support_high_taxes_p)) @ "support_high_taxes"
    )
    attended_elite_college = (
        genjax.bernoulli(logit(attended_elite_college_p)) @ "attended_elite_college"
    )
    region = genjax.categorical(jnp.log(region_logits)) @ "region"

    return {
        "male": male,
        "high_life_expectancy": high_life_expectancy,
        "connected": connected,
        "supports_medicare": supports_medicare,
        "vote_republican": vote_republican,
        "high_income": high_income,
        "support_high_taxes": support_high_taxes,
        "attended_elite_college": attended_elite_college,
        "region": region,
    }


@static_gen_fn
def gen_z1():
    z = gen_parametrized_z.inline(
        0.6,
        0.75,
        0.75,
        0.75,
        0.75,
        0.75,
        0.75,
        0.75,
        jnp.array([0.05, 0.15, 0.6, 0.2]),
    )
    return z


@static_gen_fn
def gen_z2():
    z = gen_parametrized_z.inline(
        0.4,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        jnp.array([0.6, 0.2, 0.05, 0.15]),
    )
    return z


switch = genjax.switch_combinator(gen_z1, gen_z2)


@static_gen_fn
def gen_z():
    is_model2 = genjax.bernoulli(logit(0.7)) @ "model"
    is_model2 = jnp.array(is_model2, dtype=int)
    z = switch(is_model2) @ "z"
    return z


@static_gen_fn
def make_health_data(key, gen_z, n_sample, n_importance_samples):
    # model for OI life expectancy data
    # https://opportunityinsights.org/wp-content/uploads/2018/04/health_ineq_online_table_7_readme.pdf
    n_importance_samples = n_importance_samples.const

    conditions = product(
        [0, 1],
        [0, 1],
        [0, 1, 2, 3],
    )
    conditions = np.array([c for c in conditions])

    condition_names = ["high_income", "male", "region"]
    keys = jax.random.split(key, len(conditions))

    trs, ws = jax.vmap(importance_sample, in_axes=(0, None, 0, None, None))(
        keys, gen_z, conditions, condition_names, n_importance_samples
    )

    logprob_conditions = jax.vmap(estimate_logprobs)(ws)

    group_sizes = (
        genjax.multinomial(jnp.array(n_sample, float), logprob_conditions) @ "group_sizes"
    )
    mus, stds = jax.vmap(estimate_mu_std, in_axes=(0, 0, None))(
        trs, ws, "high_life_expectancy"
    )

    map_aggregate_mean = genjax.map_combinator(in_axes=(0, 0, 0))(aggregate_mean)
    means = map_aggregate_mean(mus, stds, group_sizes) @ "map_aggregate_mean"

    # make dictionary with high_income, male, region, mean(high_life_expectancy), and counts
    conditions_dict = make_conditions_dict(conditions, condition_names)
    conditions_dict["mean(high_life_expectancy)"] = means
    conditions_dict["count"] = group_sizes

    return conditions_dict

def test_gen_z():
    # Two branches for a branching submodel.
    key = jax.random.PRNGKey(0)
    tr = gen_z.simulate(key, ())
    retval = tr.get_retval()

    for k, v in retval.items():
        assert k in [
            "attended_elite_college",
            "connected",
            "high_income",
            "high_life_expectancy",
            "male",
            "region",
            "support_high_taxes",
            "supports_medicare",
            "vote_republican",
        ]
        assert v == 0 or v == 1


def test_importance_sampling():
    key = jax.random.PRNGKey(0)
    observations = genjax.choice_map(
        {
            "model": 1,
            "z": {
                "attended_elite_college": 1,
                "connected": 1,
                "high_income": 1,
                "high_life_expectancy": 1,
                "male": 1,
                "region": 0,
                "support_high_taxes": 1,
                "supports_medicare": 1,
                "vote_republican": 1,
            },
        }
    )
    tr, w = gen_z.importance(key, observations, ())
    p = 0.7 * 0.6 * 0.4 * (0.25**7)
    assert jnp.isclose(w, jnp.log(p))


def test_aggregate_mean():
    key = jax.random.PRNGKey(0)
    conditions = [0, 1, 2]
    condition_names = ("high_income", "connected", "region")
    n_sample = 100000
    target_name = "high_life_expectancy"
    trs, ws = importance_sample(key, gen_z, conditions, condition_names, n_sample)
    stochastic_mean, stochastic_std = estimate_mu_std(trs, ws, target_name)

    p_conditions_given_group1 = 0.25 * 0.75 * 0.6
    p_group1_given_conditions = p_conditions_given_group1 * 0.3
    p_conditions_given_group2 = 0.25 * 0.75 * 0.05
    p_group2_given_conditions = p_conditions_given_group2 * 0.7
    p_group1_given_conditions, p_group2_given_conditions = (
        p_group1_given_conditions
        / (p_group1_given_conditions + p_group2_given_conditions),
        p_group2_given_conditions
        / (p_group1_given_conditions + p_group2_given_conditions),
    )

    mean = p_group1_given_conditions * 0.75 + p_group2_given_conditions * 0.25
    std = jnp.sqrt(mean * (1 - mean))

    assert jnp.isclose(stochastic_mean, mean, rtol=3e-3)
    assert jnp.isclose(stochastic_std, std, rtol=3e-3)


def test_make_health_data():
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 2)
    n_sample = 1000
    n_importance_samples = 10000
    tr = make_health_data.simulate(keys[0], (keys[1], gen_z, n_sample, genjax.Pytree.const(n_importance_samples)))
    df_dict = tr.get_retval()
    df_dict = {k: np.array(v) for k, v in df_dict.items()}
    df = pl.from_dict(df_dict)

    assert True # smoke test