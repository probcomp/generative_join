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

    politics_choices = politics_tr.get_choices().filter(genjax.select(("politics_samples", "component")).complement())

    obs = genjax.choice_map({
        "health": health_tr.get_choices(),
        "social": social_tr.get_choices(),
        "politics": politics_choices,
    })

    keys = jax.random.split(keys[5], 4)
    tr, w = model.importance(keys[0], obs, (keys[1], genjax.Pytree.const(n_sample), genjax.Pytree.const(n_importance_samples)))
    choicemap = tr["politics"]["politics_samples"]
    choicemap = choicemap.filter(genjax.select("component").complement())

    def get_component_logprobs(key, gen_fn, gen_fn_args, component, choicemap):
        chm, _ = choicemap.inner.merge(genjax.choice_map({"component": component}))
        _, w = gen_fn.importance(key, chm, gen_fn_args)
        return w

    def get_all_component_logprobs(key, args):
        map_keys = jax.random.split(key, n_sample)
        map_component_logprobs = jax.vmap(get_component_logprobs, in_axes=(0, None, None, None, 0))
        map_component_logprobs = jax.vmap(map_component_logprobs, in_axes=(None, None, None, 0, None))
        return map_component_logprobs(
            map_keys,
            mixture_model, 
            args,
            jnp.array([0, 1]),
            choicemap
        )

    def sample_components(key, tr, args):
        keys = jax.random.split(key)
        result = get_all_component_logprobs(keys[0], args)
        components = jax.random.categorical(keys[1], result, axis=0)

        new_tr = tr.get_choices().filter(genjax.select(("politics", "politics_samples", "component")).complement())
        new_tr, _ = new_tr.merge(
            genjax.choice_map({"politics": {"politics_samples": 
                genjax.vector_choice_map({"component": components})}})
        )
        return new_tr

    # @scan_tqdm(n_mala_iters)
    def mala_iter(tr, key):
        key, subkey = jax.random.split(key)
        selection = genjax.select("logits")
        mala_alg = MetropolisAdjustedLangevinAlgorithm(selection, 1e-3)

        accepted_arr = jnp.zeros(len(component_params.keys()) + 1)

        tr, accepted = mala_alg(key, tr)
        accepted_arr = accepted_arr.at[0].set(accepted)

        for i, k in enumerate(component_params.keys()):
            key, subkey = jax.random.split(subkey)
            selection = genjax.select((k, "logp"))
            mala_alg = MetropolisAdjustedLangevinAlgorithm(selection, 1e-3)
            tr, accepted = mala_alg(key, tr)
            accepted_arr = accepted_arr.at[i+1].set(accepted)

        key, subkey = jax.random.split(subkey)
        gen_component_params = OrderedDict({
            k: tr[k]["logp"]
            for k in component_params.keys()
        })
        args = (tr["logits"], gen_component_params, gen_parametrized_z)
        chm = sample_components(key, tr, args)

        key, subkey = jax.random.split(subkey)
        gen_fn = tr.get_gen_fn()
        new_tr, w = gen_fn.importance(key, chm, tr.get_args())

        return new_tr, accepted_arr

    new_tr, accepted = jax.lax.scan(mala_iter, tr, jax.random.split(keys[2], n_mala_iters))

    # todo make this assertion be about a certain acceptance rate
    # assert jnp.all(accepted)

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