import functools
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct
from orca.common.typing import Params, PRNGKey

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

default_init = nn.initializers.xavier_uniform


def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension.

    Args:
        batch: A pytree of arrays.
        sharding: A jax Sharding object with shape (num_devices,).
    """
    return jax.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class ModuleDict(nn.Module):
    """
    Utility class for wrapping a dictionary of modules. This is useful when you have multiple modules that you want to
    initialize all at once (creating a single `params` dictionary), but you want to be able to call them separately
    later. As a bonus, the modules may have sub-modules nested inside them that share parameters (e.g. an image encoder)
    and Flax will automatically handle this without duplicating the parameters.

    To initialize the modules, call `init` with no `name` kwarg, and then pass the example arguments to each module as
    additional kwargs. To call the modules, pass the name of the module as the `name` kwarg, and then pass the arguments
    to the module as additional args or kwargs.

    Example usage:
    ```
    shared_encoder = Encoder()
    actor = Actor(encoder=shared_encoder)
    critic = Critic(encoder=shared_encoder)

    model_def = ModuleDict({"actor": actor, "critic": critic})
    params = model_def.init(rng_key, actor=example_obs, critic=(example_obs, example_action))

    actor_output = model_def.apply({"params": params}, example_obs, name="actor")
    critic_output = model_def.apply({"params": params}, example_obs, action=example_action, name="critic")
    ```
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f"When `name` is not specified, kwargs must contain the arguments for each module. "
                    f"Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}"
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class JaxRLTrainState(struct.PyTreeNode):
    """
    Custom TrainState class to replace `flax.training.train_state.TrainState`.

    Adds support for holding target params and updating them via polyak
    averaging. Adds the ability to hold an rng key for dropout.

    Also generalizes the TrainState to support an arbitrary pytree of
    optimizers, `txs`. When `apply_gradients()` is called, the `grads` argument
    must have `txs` as a prefix. This is backwards-compatible, meaning `txs` can
    be a single optimizer and `grads` can be a single tree with the same
    structure as `self.params`.

    Also adds a convenience method `apply_loss_fns` that takes a pytree of loss
    functions with the same structure as `txs`, computes gradients, and applies
    them using `apply_gradients`.

    Attributes:
        step: The current training step.
        apply_fn: The function used to apply the model.
        params: The model parameters.
        target_params: The target model parameters.
        txs: The optimizer or pytree of optimizers.
        opt_states: The optimizer state or pytree of optimizer states.
        rng: The internal rng state.
    """

    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Params
    target_params: Params
    txs: Any = struct.field(pytree_node=False)
    opt_states: Any
    rng: PRNGKey

    @staticmethod
    def _tx_tree_map(*args, **kwargs):
        return jax.tree_map(
            *args,
            is_leaf=lambda x: isinstance(x, optax.GradientTransformation),
            **kwargs,
        )

    def target_update(self, tau: float) -> "JaxRLTrainState":
        """
        Performs an update of the target params via polyak averaging. The new
        target params are given by:

            new_target_params = tau * params + (1 - tau) * target_params
        """
        new_target_params = jax.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau), self.params, self.target_params
        )
        return self.replace(target_params=new_target_params)

    def apply_gradients(self, *, grads: Any) -> "JaxRLTrainState":
        """
        Only difference from flax's TrainState is that `grads` must have
        `self.txs` as a tree prefix (i.e. where `self.txs` has a leaf, `grads`
        has a subtree with the same structure as `self.params`.)
        """
        updates_and_new_states = self._tx_tree_map(
            lambda tx, opt_state, grad: tx.update(grad, opt_state, self.params),
            self.txs,
            self.opt_states,
            grads,
        )
        updates = self._tx_tree_map(lambda _, x: x[0], self.txs, updates_and_new_states)
        new_opt_states = self._tx_tree_map(
            lambda _, x: x[1], self.txs, updates_and_new_states
        )

        # not the cleanest, I know, but this flattens the leaves of `updates`
        # into a list where leaves are defined by `self.txs`
        updates_flat = []
        self._tx_tree_map(
            lambda _, update: updates_flat.append(update), self.txs, updates
        )

        # apply all the updates additively
        updates_acc = jax.tree_map(
            lambda *xs: jnp.sum(jnp.array(xs), axis=0), *updates_flat
        )
        new_params = optax.apply_updates(self.params, updates_acc)

        return self.replace(
            step=self.step + 1, params=new_params, opt_states=new_opt_states
        )

    def apply_loss_fns(
        self, loss_fns: Any, pmap_axis: str = None, has_aux: bool = False
    ) -> Union["JaxRLTrainState", Tuple["JaxRLTrainState", Any]]:
        """
        Convenience method to compute gradients based on `self.params` and apply
        them using `apply_gradients`. `loss_fns` must have the same structure as
        `txs`, and each leaf must be a function that takes two arguments:
        `params` and `rng`.

        This method automatically provides fresh rng to each loss function and
        updates this train state's internal rng key.

        Args:
            loss_fns: loss function or pytree of loss functions with same
                structure as `self.txs`. Each loss function must take `params`
                as the first argument and `rng` as the second argument, and return
                a scalar value.
            pmap_axis: if not None, gradients (and optionally auxiliary values)
                will be averaged over this axis
            has_aux: if True, each `loss_fn` returns a tuple of (loss, aux) where
                `aux` is a pytree of auxiliary values to be returned by this
                method.

        Returns:
            If `has_aux` is True, returns a tuple of (new_train_state, aux).
            Otherwise, returns the new train state.
        """
        # create a pytree of rngs with the same structure as `loss_fns`
        treedef = jax.tree_util.tree_structure(loss_fns)
        new_rng, *rngs = jax.random.split(self.rng, treedef.num_leaves + 1)
        rngs = jax.tree_util.tree_unflatten(treedef, rngs)

        # compute gradients
        grads_and_aux = jax.tree_map(
            lambda loss_fn, rng: jax.grad(loss_fn, has_aux=has_aux)(self.params, rng),
            loss_fns,
            rngs,
        )

        # update rng state
        self = self.replace(rng=new_rng)

        # average across devices if necessary
        if pmap_axis is not None:
            grads_and_aux = jax.lax.pmean(grads_and_aux, axis_name=pmap_axis)

        if has_aux:
            grads = jax.tree_map(lambda _, x: x[0], loss_fns, grads_and_aux)
            aux = jax.tree_map(lambda _, x: x[1], loss_fns, grads_and_aux)
            return self.apply_gradients(grads=grads), aux
        else:
            return self.apply_gradients(grads=grads_and_aux)

    @classmethod
    def create(
        cls, *, apply_fn, params, txs, target_params=None, rng=jax.random.PRNGKey(0)
    ):
        """
        Initializes a new train state.

        Args:
            apply_fn: The function used to apply the model, typically `model_def.apply`.
            params: The model parameters, typically from `model_def.init`.
            txs: The optimizer or pytree of optimizers.
            target_params: The target model parameters.
            rng: The rng key used to initialize the rng chain for `apply_loss_fns`.
        """
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            target_params=target_params,
            txs=txs,
            opt_states=cls._tx_tree_map(lambda tx: tx.init(params), txs),
            rng=rng,
        )
