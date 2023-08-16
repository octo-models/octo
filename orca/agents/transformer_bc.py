import copy
from functools import partial
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
from orca.common.common import JaxRLTrainState, nonpytree_field
from orca.common.typing import Batch, PRNGKey
from orca.networks.transformer_policy import TransformerPolicy


class TransformerBCAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    lr_schedule: Any = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        def loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            info = self.state.apply_fn(
                {"params": params},
                batch["observations"],
                batch["goals"],
                batch["actions"],
                train=True,
                rngs={"dropout": key},
            )
            return info["loss"], info

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )

        # log learning rates
        info["lr"] = self.lr_schedule(self.state.step)

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        *,
        seed: PRNGKey,
        temperature: float = 1.0,
        argmax=False
    ) -> jnp.ndarray:
        if len(observations["image"].shape) == 4:
            # unbatched input from evaluation
            observations = jax.tree_map(lambda x: x[None], observations)
            goals = jax.tree_map(lambda x: x[None], goals)
        actions = self.state.apply_fn(
            {"params": self.state.params},
            observations,
            goals,
            method="predict_action",
            train=False,
            argmax=argmax,
            rng=seed,
            temperature=temperature,
        )
        return actions[0]

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        return self.state.apply_fn(
            {"params": self.state.params},
            batch["observations"],
            batch["goals"],
            batch["actions"],
        )

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        goals: FrozenDict,
        # Model architecture
        observation_tokenizer_defs: Sequence[nn.Module],
        task_tokenizer_defs: Sequence[nn.Module],
        policy_kwargs: dict = {},
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        decay_steps: int = 1000000,
        # Load pretrained weights
        pretrained_weights: Sequence[Any] = [],
    ):
        # time sequence length is the observation history length
        if len(observations["image"].shape) == 5:
            # batched input
            time_sequence_length = observations["image"].shape[1]
        else:
            # unbatched input
            time_sequence_length = observations["image"].shape[0]

        model_def = TransformerPolicy(
            observation_tokenizers=observation_tokenizer_defs,
            task_tokenizers=task_tokenizer_defs,
            action_dim=actions.shape[-1],
            time_sequence_length=time_sequence_length,
            **policy_kwargs
        )

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )
        tx = optax.adam(lr_schedule)

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng, observations, goals, actions)["params"]

        for loader in pretrained_weights:
            params = loader(params)

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )

        return cls(state, lr_schedule)
