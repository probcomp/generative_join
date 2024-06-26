# Copyright 2023 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.datatypes.generative import Selection, Trace
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.typing import FloatArray
from genjax._src.core.pytree import Pytree
from genjax.typing import typecheck
from genjax._src.generative_functions.distributions.tensorflow_probability import normal
from generative_join.kernel import MCMCKernel

@typecheck
def choice_grad(gen_fn, trace: Trace, selection: Selection):
    fixed = trace.strip().filter(selection.complement())
    choice = trace.strip().filter(selection)
    scorer, _ = gen_fn.unzip(fixed)
    grad, nograd = Pytree.tree_grad_split(
        (choice, trace.get_args()),
    )
    choice_gradient_tree, _ = jax.grad(scorer)(grad, nograd)
    return choice_gradient_tree


class MetropolisAdjustedLangevinAlgorithm(MCMCKernel):
    selection: Selection
    tau: FloatArray

    def flatten(self):
        return (self.selection, self.tau), ()

    def _grad_step_no_none(self, v1, v2):
        if v2 is None:
            return v1
        else:
            return v1 + self.tau * v2

    def _random_split_like_tree(self, rng_key, target=None, treedef=None):
        if treedef is None:
            treedef = jtu.tree_structure(target)
        keys = jax.random.split(rng_key, treedef.num_leaves)
        return jtu.tree_unflatten(treedef, keys)

    def _tree_random_normal_fixed_std(self, rng_key, mu, std):
        keys_tree = self._random_split_like_tree(rng_key, mu)
        return jax.tree_map(
            lambda m, k: normal.sample(k, m, std),
            mu,
            keys_tree,
        )

    def _tree_logpdf_normal_fixed_std(self, values, mu, std):
        logpdf_tree = jax.tree_map(
            lambda v, m: normal.logpdf(v, m, std),
            values,
            mu,
        )
        leaves = jnp.array(jtu.tree_leaves(logpdf_tree))
        return leaves.sum()

    def apply(self, key, trace: Trace):
        args = trace.get_args()
        gen_fn = trace.get_gen_fn()
        std = jnp.sqrt(2 * self.tau)
        argdiffs = Diff.tree_diff_no_change(args)

        # Forward proposal.
        forward_gradient_trie = choice_grad(gen_fn, trace, self.selection)
        forward_values = trace.strip().filter(self.selection)
        forward_mu = jtu.tree_map(
            self._grad_step_no_none,
            forward_values,
            forward_gradient_trie,
        )

        key, sub_key = jax.random.split(key)
        proposed_values = self._tree_random_normal_fixed_std(sub_key, forward_mu, std)
        forward_score = self._tree_logpdf_normal_fixed_std(
            proposed_values, forward_mu, std
        )

        # Get model weight.
        key, sub_key = jax.random.split(key)
        (new_trace, weight, _, _) = gen_fn.update(
            sub_key, trace, proposed_values, argdiffs
        )

        # Backward proposal.
        backward_gradient_trie = choice_grad(gen_fn, new_trace, self.selection)
        backward_mu = jtu.tree_map(
            self._grad_step_no_none,
            proposed_values,
            backward_gradient_trie,
        )
        backward_score = self._tree_logpdf_normal_fixed_std(
            forward_values, backward_mu, std
        )

        alpha = weight - forward_score + backward_score
        check = jnp.log(jax.random.uniform(key)) < alpha
        return jax.lax.cond(
            check,
            lambda *args: (new_trace, True),
            lambda *args: (trace, False),
        )

    def reversal(self):
        return self