import jax.numpy as jnp
import jax
import genjax
import polars as pl
import numpy as np
from generative_join.modeling import (
    importance_sample,
    estimate_mu_std,
    mixture_model,
)
from generative_join.synthetic_data import (
    component_logprobs, component_params, gen_parametrized_z, make_health_data
)

def test_gen_z():
    # Two branches for a branching submodel.
    key = jax.random.PRNGKey(0)
    tr = mixture_model.simulate(
        key, (component_logprobs, component_params, gen_parametrized_z)
    )
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
            "component": 1,
            "attended_elite_college": 1,
            "connected": 1,
            "high_income": 1,
            "high_life_expectancy": 1,
            "male": 1,
            "region": 0,
            "support_high_taxes": 1,
            "supports_medicare": 1,
            "vote_republican": 1,
        }
    )
    tr, w = mixture_model.importance(
        key, observations, (component_logprobs, component_params, gen_parametrized_z)
    )
    p = 0.7 * 0.6 * 0.4 * (0.25**7)
    assert jnp.isclose(w, jnp.log(p))


def test_aggregate_mean():
    key = jax.random.PRNGKey(0)
    conditions = [0, 1, 2]
    condition_names = ("high_income", "connected", "region")
    n_sample = 100000
    target_name = "high_life_expectancy"
    trs, ws = importance_sample(
        key,
        mixture_model,
        (component_logprobs, component_params, gen_parametrized_z),
        conditions,
        condition_names,
        n_sample,
    )
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

    # should this be more precise? investigate
    assert jnp.isclose(stochastic_mean, mean, rtol=7e-3)
    assert jnp.isclose(stochastic_std, std, rtol=5e-3)


def test_make_health_data():
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 2)
    n_sample = 1000
    n_importance_samples = 10000
    tr = make_health_data.simulate(
        keys[0],
        (
            keys[1],
            mixture_model,
            (component_logprobs, component_params, gen_parametrized_z),
            n_sample,
            genjax.Pytree.const(n_importance_samples),
        ),
    )
    df_dict = tr.get_retval()
    df_dict = {k: np.array(v) for k, v in df_dict.items()}
    df = pl.from_dict(df_dict)

    assert True  # smoke test
