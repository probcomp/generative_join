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
from generative_join.metropolis_adjusted_langevin_algorithm import MetropolisAdjustedLangevinAlgorithm
from tests.test_synthetic_data import gen_parametrized_z, component_params, component_logprobs, make_health_data
from generative_join.modeling import mixture_model

@static_gen_fn
def make_social_data(key, gen_z, gen_z_args, n_sample, n_importance_samples):
    # model for OI social connectedness data
    n_importance_samples = n_importance_samples.const

    conditions = product(
        [0, 1],
        [0, 1, 2, 3],
    )
    conditions = np.array([c for c in conditions])

    condition_names = ["attended_elite_college", "region"]
    keys = jax.random.split(key, len(conditions))

    trs, ws = jax.vmap(importance_sample, in_axes=(0, None, None, 0, None, None))(
        keys, gen_z, gen_z_args, conditions, condition_names, n_importance_samples
    )

    logprob_conditions = jax.vmap(estimate_logprobs)(ws)

    group_sizes = (
        genjax.multinomial(jnp.array(n_sample, float), logprob_conditions) @ "social_group_sizes"
    )
    mus, stds = jax.vmap(estimate_mu_std, in_axes=(0, 0, None))(
        trs, ws, "connected"
    )

    map_aggregate_mean = genjax.map_combinator(in_axes=(0, 0, 0))(aggregate_mean)
    means = map_aggregate_mean(mus, stds, group_sizes) @ "social_map_aggregate_mean"

    # make dictionary with high_income, male, region, mean(high_life_expectancy), and counts
    conditions_dict = make_conditions_dict(conditions, condition_names)
    conditions_dict["mean(connected)"] = means
    conditions_dict["count"] = group_sizes

    return conditions_dict

@static_gen_fn
def make_politics_data(gen_z, gen_z_args, n_sample):
    # model for cces
    map_array = jnp.ones(n_sample.const)
    map_gen_z = genjax.map_combinator(in_axes=(None, None, 0))(aux_gen_z)
    sample_dict = map_gen_z(gen_z, gen_z_args, map_array) @ "politics_samples"

    return {k: v for k, v in sample_dict.items() if k in ["male", "region", "supports_medicare", "vote_republican"]}
    
@static_gen_fn
def aux_gen_z(gen_z, gen_z_args, _):
    # hack while the repeat combinator in genjax remains buggy
    return gen_z.inline(*gen_z_args)

def trace_to_df(tr):
    tr_dict = {k: np.array(v) for k, v in tr.get_retval().items()}
    return pl.from_dict(tr_dict)

@static_gen_fn
def model(key, n_sample, n_importance_samples):
    n_sample = n_sample.const
    n_importance_samples = n_importance_samples.const
    keys = jax.random.split(key, 2)

    component_logprobs = genjax.normal([0., 0.], [1., 1.]) @ "logits"

    args = (component_logprobs, component_params, gen_parametrized_z)
    # fix: if I don't inline the calls to health/social data, I get an UnexpectedTracer error when doing
    # IS on model. 
    health_data = make_health_data.inline(keys[0], mixture_model, args, n_sample, genjax.Pytree.const(n_importance_samples))
    social_data = make_social_data.inline(keys[1], mixture_model, args, n_sample, genjax.Pytree.const(n_importance_samples))
    politics_data = make_politics_data.inline(mixture_model, args, genjax.Pytree.const(n_sample))

    return health_data, social_data, politics_data

def test_mala():
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 6)
    n_sample = 1000
    n_importance_samples = 10000
    args = (component_logprobs, component_params, gen_parametrized_z)

    health_tr = make_health_data.simulate(keys[0], (keys[1], mixture_model, args, n_sample, genjax.Pytree.const(n_importance_samples)))
    social_tr = make_social_data.simulate(keys[2], (keys[3], mixture_model, args, n_sample, genjax.Pytree.const(n_importance_samples)))
    politics_tr = make_politics_data.simulate(keys[4], (mixture_model, args, genjax.Pytree.const(n_sample)))

    obs, _ = health_tr.get_choices().merge(social_tr.get_choices())
    obs, _ = obs.merge(politics_tr.get_choices())

    keys = jax.random.split(keys[5], 3)
    tr, w = model.importance(keys[0], obs, (keys[1], genjax.Pytree.const(n_sample), genjax.Pytree.const(n_importance_samples)))
    selection = genjax.select("logits")
    mala_alg = MetropolisAdjustedLangevinAlgorithm(selection, 1e-3)
    new_tr, accepted = mala_alg(keys[2], tr)

    assert accepted