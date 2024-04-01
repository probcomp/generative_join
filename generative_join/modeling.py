from genjax import static_gen_fn
import genjax
import jax.numpy as jnp
import jax
from itertools import product
from jax.scipy.special import logsumexp
from typing import Any
import numpy as np


@static_gen_fn
def aggregate(key, gen_z, gen_z_args, group_by_vars: dict[str, tuple[Any]], aggregation_vars: dict[str, tuple[str]], 
        n_samples: int, n_importance_samples=genjax.Pytree.const(1000), count: bool = True):
    n_importance_samples =  n_importance_samples.const
    conditions = product(*group_by_vars.values())
    conditions = np.array([c for c in conditions])

    condition_names = [c for c in group_by_vars.keys()]
    keys = jax.random.split(key, len(conditions))

    trs, ws = jax.vmap(importance_sample, in_axes=(0, None, None, 0, None, None))(
        keys, gen_z, gen_z_args, conditions, condition_names, n_importance_samples
    )

    logprob_conditions = jax.vmap(estimate_logprobs)(ws)
    counts = (
        genjax.multinomial(jnp.array(n_samples, float), logprob_conditions)
        @ "counts"
    )
    conditions_dict = make_conditions_dict(conditions, condition_names)

    # TODO vmap this
    for var in aggregation_vars.keys():
        mus, stds = jax.vmap(estimate_mu_std, in_axes=(0, 0, None))(
            trs, ws, var
        )

        map_aggregate_mean = genjax.map_combinator(in_axes=(0, 0, 0))(aggregate_mean)
        means = map_aggregate_mean(mus, stds, counts) @ var
        conditions_dict[f"mean({var})"] = means

    conditions_dict["count"] = counts

    return conditions_dict



@static_gen_fn
def aggregate_mean(mu, std, count):
    sigma = std / jnp.sqrt(count)

    return genjax.normal(mu, sigma) @ "mean"


def importance_sample(key, gen_fn, gen_fn_args, conditions, condition_names, n_samples):
    observations = make_obs(conditions, condition_names)
    keys = jax.random.split(key, n_samples)
    return jax.vmap(gen_fn.importance, in_axes=(0, None, None))(
        keys, observations, gen_fn_args
    )


def estimate_mu_std(trs, ws, target):
    retval = trs.get_retval()

    exp_ws = jnp.exp(ws - jnp.max(ws))
    ws_norm = exp_ws / jnp.sum(exp_ws)

    mu = jnp.sum(ws_norm * retval[target])
    std = jnp.sqrt(jnp.sum(ws_norm * ((retval[target] - mu) ** 2)))

    return mu, std


def estimate_logprobs(w):
    return logsumexp(w) - jnp.log(w.shape[0])


def make_conditions_dict(conditions, condition_names):
    return {name: cond for cond, name in zip(conditions.T, condition_names)}


def make_obs(conditions, condition_names):
    return genjax.choice_map(
        {name: cond for cond, name in zip(conditions, condition_names)}
    )


@static_gen_fn
def mixture_model(component_logprobs, component_params, gen_fn):
    component = genjax.categorical(component_logprobs) @ "component"
    component_param = {k: v[component] for k, v in component_params.items()}
    # would like to inline this, but leads to unexpected keyword arg
    value = gen_fn.inline(*component_param.values())
    return value
