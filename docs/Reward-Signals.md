# Reward Signals

## Enabling Reward Signals

Reward signals, like other hyperparameters, are defined in the trainer config `.yaml` file. An
example is provided in `config/trainer_config.yaml` and `config/gail_config.yaml`. To enable a reward signal, add it to the
`reward_signals:` section under the behavior name. For instance, to enable the extrinsic signal
in addition to a small curiosity reward and a GAIL reward signal, you would define your `reward_signals` as follows:

```yaml
reward_signals:
    extrinsic:
        strength: 1.0
        gamma: 0.99
    curiosity:
        strength: 0.02
        gamma: 0.99
        encoding_size: 256
    gail:
        strength: 0.01
        gamma: 0.99
        encoding_size: 128
        demo_path: Project/Assets/ML-Agents/Examples/Pyramids/Demos/ExpertPyramid.demo
```

Each reward signal should define at least two parameters, `strength` and `gamma`, in addition
to any class-specific hyperparameters. Note that to remove a reward signal, you should delete
its entry entirely from `reward_signals`. At least one reward signal should be left defined
at all times.

## Reward Signal Types
As part of the toolkit, we provide three reward signal types as part of hyperparameters - Extrinsic, Curiosity, and GAIL.

### Extrinsic Reward Signal

The `extrinsic` reward signal is simply the reward given by the
[environment](Learning-Environment-Design.md). Remove it to force the agent
to ignore the environment reward.

#### Strength

`strength` is the factor by which to multiply the raw
reward. Typical ranges will vary depending on the reward signal.

Typical Range: `1.0`

#### Gamma

`gamma` corresponds to the discount factor for future rewards. This can be
thought of as how far into the future the agent should care about possible
rewards. In situations when the agent should be acting in the present in order
to prepare for rewards in the distant future, this value should be large. In
cases when rewards are more immediate, it can be smaller.

Typical Range: `0.8` - `0.995`

### Curiosity Reward Signal

#### Strength

In this case, `strength` corresponds to the magnitude of the curiosity reward generated
by the intrinsic curiosity module. This should be scaled in order to ensure it is large enough
to not be overwhelmed by extrinsic reward signals in the environment.
Likewise it should not be too large to overwhelm the extrinsic reward signal.

Typical Range: `0.001` - `0.1`

#### Gamma

`gamma` corresponds to the discount factor for future rewards.

Typical Range: `0.8` - `0.995`

#### (Optional) Encoding Size

`encoding_size` corresponds to the size of the encoding used by the intrinsic curiosity model.
This value should be small enough to encourage the ICM to compress the original
observation, but also not too small to prevent it from learning to differentiate between
demonstrated and actual behavior.

Default Value: `64`

Typical Range: `64` - `256`

#### (Optional) Learning Rate

`learning_rate` is the learning rate used to update the intrinsic curiosity module.
This should typically be decreased if training is unstable, and the curiosity loss is unstable.

Default Value: `3e-4`

Typical Range: `1e-5` - `1e-3`

### GAIL Reward Signal

#### Strength

`strength` is the factor by which to multiply the raw reward. Note that when using GAIL
with an Extrinsic Signal, this value should be set lower if your demonstrations are
suboptimal (e.g. from a human), so that a trained agent will focus on receiving extrinsic
rewards instead of exactly copying the demonstrations. Keep the strength below about 0.1 in those cases.

Typical Range: `0.01` - `1.0`

#### Gamma

`gamma` corresponds to the discount factor for future rewards.

Typical Range: `0.8` - `0.9`

#### Demo Path

`demo_path` is the path to your `.demo` file or directory of `.demo` files. See the [imitation learning guide](Training-Imitation-Learning.md).

#### (Optional) Encoding Size

`encoding_size` corresponds to the size of the hidden layer used by the discriminator.
This value should be small enough to encourage the discriminator to compress the original
observation, but also not too small to prevent it from learning to differentiate between
demonstrated and actual behavior. Dramatically increasing this size will also negatively affect
training times.

Default Value: `64`

Typical Range: `64` - `256`

#### (Optional) Learning Rate

`learning_rate` is the learning rate used to update the discriminator.
This should typically be decreased if training is unstable, and the GAIL loss is unstable.

Default Value: `3e-4`

Typical Range: `1e-5` - `1e-3`

#### (Optional) Use Actions

`use_actions` determines whether the discriminator should discriminate based on both
observations and actions, or just observations. Set to `True` if you want the agent to
mimic the actions from the demonstrations, and `False` if you'd rather have the agent
visit the same states as in the demonstrations but with possibly different actions.
Setting to `False` is more likely to be stable, especially with imperfect demonstrations,
but may learn slower.

Default Value: `false`

#### (Optional) Variational Discriminator Bottleneck

`use_vail` enables a [variational bottleneck](https://arxiv.org/abs/1810.00821) within the
GAIL discriminator. This forces the discriminator to learn a more general representation
and reduces its tendency to be "too good" at discriminating, making learning more stable.
However, it does increase training time. Enable this if you notice your imitation learning is
unstable, or unable to learn the task at hand.

Default Value: `false`
