import jax.numpy as jnp
import genjax
from collections import OrderedDict
from jax.scipy.special import logit
from genjax import static_gen_fn
from generative_join.modeling import aggregate

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
    male = genjax.bernoulli(male_p) @ "male"
    high_life_expectancy = (
        genjax.bernoulli(high_life_expectancy_p) @ "high_life_expectancy"
    )
    connected = genjax.bernoulli(connected_p) @ "connected"
    supports_medicare = (
        genjax.bernoulli(supports_medicare_p) @ "supports_medicare"
    )
    vote_republican = genjax.bernoulli(vote_republican_p) @ "vote_republican"
    high_income = genjax.bernoulli(high_income_p) @ "high_income"
    support_high_taxes = (
        genjax.bernoulli(support_high_taxes_p) @ "support_high_taxes"
    )
    attended_elite_college = (
        genjax.bernoulli(attended_elite_college_p) @ "attended_elite_college"
    )
    region = genjax.categorical(region_logits) @ "region"

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


component_params = OrderedDict(
    {
        "male_p": logit(jnp.array([0.6, 0.4])),
        "high_life_expectancy_p": logit(jnp.array([0.75, 0.25])),
        "connected_p": logit(jnp.array([0.75, 0.25])),
        "supports_medicare_p": logit(jnp.array([0.75, 0.25])),
        "vote_republican_p": logit(jnp.array([0.75, 0.25])),
        "high_income_p": logit(jnp.array([0.75, 0.25])),
        "support_high_taxes_p": logit(jnp.array([0.75, 0.25])),
        "attended_elite_college_p": logit(jnp.array([0.75, 0.25])),
        "region_logits": jnp.log(jnp.array([[0.05, 0.15, 0.6, 0.2], [0.6, 0.2, 0.05, 0.15]])),
    }
)


component_logprobs = jnp.log(jnp.array([0.3, 0.7]))


@static_gen_fn
def make_health_data(key, gen_z, gen_z_args, n_samples, n_importance_samples):
    # model for OI life expectancy data
    # https://opportunityinsights.org/wp-content/uploads/2018/04/health_ineq_online_table_7_readme.pdf

    return aggregate.inline(
        key,
        gen_z,
        gen_z_args,
        {
            "high_income": [0, 1],
            "male": [0, 1],
            "region": [0, 1, 2, 3],
        },
        {
            "high_life_expectancy": "mean"
        },
        n_samples,
    )

@static_gen_fn
def make_social_data(key, gen_z, gen_z_args, n_samples, n_importance_samples):
    # model for OI social connectedness data
    return aggregate.inline(
        key,
        gen_z,
        gen_z_args,
        {
            "attended_elite_college": [0, 1],
            "region": [0, 1, 2, 3],
        },
        {
            "connected": "mean"
        },
        n_samples,
    )

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

