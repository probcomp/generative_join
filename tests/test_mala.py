import jax
import jax.numpy as jnp
import genjax
from jax.scipy.special import expit
from jax_tqdm import scan_tqdm
from genjax import static_gen_fn
from collections import OrderedDict
from generative_join.metropolis_adjusted_langevin_algorithm import MetropolisAdjustedLangevinAlgorithm
from generative_join.synthetic_data import gen_parametrized_z, component_params, component_logprobs, make_health_data, make_politics_data, make_social_data
from generative_join.modeling import mixture_model

@static_gen_fn
def bernoulli_prior():
    return genjax.normal([0., 0.], [1., 1.]) @ "logp"

@static_gen_fn
def categorical_prior():
    return genjax.normal(jnp.zeros((2, 4)), jnp.ones((2, 4))) @ "logp"

@static_gen_fn
def model(key, n_sample, n_importance_samples):
    n_sample = n_sample.const
    n_importance_samples = n_importance_samples.const
    keys = jax.random.split(key, 2)

    component_logprobs = genjax.normal([0., 0.], [1., 1.]) @ "logits"
    gen_component_params = OrderedDict({
        k: categorical_prior() @ k if k == "region_logits" else bernoulli_prior() @ k
        for k in component_params.keys()
    })

    args = (component_logprobs, gen_component_params, gen_parametrized_z)
    # fix: if I don't inline the calls to health/social data, I get an UnexpectedTracer error when doing
    # IS on model. 
    health_data = make_health_data(keys[0], mixture_model, args, n_sample, genjax.Pytree.const(n_importance_samples)) @ "health"
    social_data = make_social_data(keys[1], mixture_model, args, n_sample, genjax.Pytree.const(n_importance_samples)) @ "social"
    politics_data = make_politics_data(mixture_model, args, genjax.Pytree.const(n_sample)) @ "politics"

    return health_data, social_data, politics_data

def test_mala():
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 6)
    n_sample = 1000
    n_importance_samples = 10000
    n_mala_iters = 100
    args = (component_logprobs, component_params, gen_parametrized_z)

    health_tr = make_health_data.simulate(keys[0], (keys[1], mixture_model, args, n_sample, genjax.Pytree.const(n_importance_samples)))
    social_tr = make_social_data.simulate(keys[2], (keys[3], mixture_model, args, n_sample, genjax.Pytree.const(n_importance_samples)))
    politics_tr = make_politics_data.simulate(keys[4], (mixture_model, args, genjax.Pytree.const(n_sample)))

    obs = genjax.choice_map({
        "health": health_tr.get_choices(),
        "social": social_tr.get_choices(),
        "politics": politics_tr.get_choices()
    })

    keys = jax.random.split(keys[5], 3)
    tr, w = model.importance(keys[0], obs, (keys[1], genjax.Pytree.const(n_sample), genjax.Pytree.const(n_importance_samples)))

    @scan_tqdm(n_mala_iters)
    def mala_iter(tr, _):
        selection = genjax.select("logits")
        mala_alg = MetropolisAdjustedLangevinAlgorithm(selection, 1e-3)

        accepted_arr = jnp.zeros(len(component_params.keys()) + 1)

        tr, accepted = mala_alg(keys[2], tr)
        accepted_arr = accepted_arr.at[0].set(accepted)

        for i, k in enumerate(component_params.keys()):
            selection = genjax.select((k, "logp"))
            mala_alg = MetropolisAdjustedLangevinAlgorithm(selection, 1e-3)
            tr, accepted = mala_alg(keys[2], tr)
            accepted_arr = accepted_arr.at[i+1].set(accepted)

        return tr, accepted_arr

    new_tr, accepted = jax.lax.scan(mala_iter, tr, jnp.arange(n_mala_iters))

    assert jnp.all(accepted)

    val = jnp.exp(new_tr["logits"]) / jnp.sum(jnp.exp(new_tr["logits"]))
    assert jnp.allclose(val, jnp.array([.3, .7]), atol=.15)

    # todo: run mala for more iterations to tighten bound
    for k in component_params.keys():
        if k == "region_logits":
            val = jnp.exp(new_tr[k]["logp"]) / jnp.sum(jnp.exp(new_tr[k]["logp"]), axis=-1, keepdims=True)
            gt_val = jnp.exp(component_params[k]) / jnp.sum(jnp.exp(component_params[k]), axis=-1, keepdims=True)
            assert jnp.allclose(val, gt_val, atol=.15)
        else:
            val = expit(new_tr[k]["logp"])
            gt_val = expit(component_params[k])
            assert jnp.allclose(val, gt_val, atol=.15)