from genjax import static_gen_fn
import genjax
import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp


@static_gen_fn
def aggregate_mean(mu, std, count):
    sigma = std / jnp.sqrt(count)

    return genjax.normal(mu, sigma) @ "mean"


def importance_sample(key, gen_fn, conditions, condition_names, n_samples):
    observations = make_obs(conditions, condition_names)
    keys = jax.random.split(key, n_samples)
    return jax.vmap(gen_fn.importance, in_axes=(0, None, None))(keys, observations, ())


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
        {"z": {name: cond for cond, name in zip(conditions, condition_names)}}
    )
