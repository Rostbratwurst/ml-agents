# ML-Agents Toolkit Overview

**The Unity Machine Learning Agents Toolkit** (ML-Agents Toolkit) is an
open-source Unity plugin that enables games and simulations to serve as
environments for training intelligent agents. Agents can be trained using
reinforcement learning, imitation learning, neuroevolution, or other machine
learning methods through a simple-to-use Python API. We also provide
implementations (based on TensorFlow) of state-of-the-art algorithms to enable
game developers and hobbyists to easily train intelligent agents for 2D, 3D and
VR/AR games. These trained agents can be used for multiple purposes, including
controlling NPC behavior (in a variety of settings such as multi-agent and
adversarial), automated testing of game builds and evaluating different game
design decisions pre-release. The ML-Agents toolkit is mutually beneficial for
both game developers and AI researchers as it provides a central platform where
advances in AI can be evaluated on Unity’s rich environments and then made
accessible to the wider research and game developer communities.

Depending on your background (i.e. researcher, game developer, hobbyist), you
may have very different questions on your mind at the moment. To make your
transition to the ML-Agents toolkit easier, we provide several background pages
that include overviews and helpful resources on the [Unity
Engine](Background-Unity.md), [machine learning](Background-Machine-Learning.md)
and [TensorFlow](Background-TensorFlow.md). We **strongly** recommend browsing
the relevant background pages if you're not familiar with a Unity scene, basic
machine learning concepts or have not previously heard of TensorFlow.

The remainder of this page contains a deep dive into ML-Agents, its key
components, different training modes and scenarios. By the end of it, you should
have a good sense of _what_ the ML-Agents toolkit allows you to do. The
subsequent documentation pages provide examples of _how_ to use ML-Agents. But
first, watch this [demo video of ML-Agents in action](https://www.youtube.com/watch?v=fiQsmdwEGT8&feature=youtu.be).

## Running Example: Training NPC Behaviors

To help explain the material and terminology in this page, we'll use a
hypothetical, running example throughout. We will explore the problem of
training the behavior of a non-playable character (NPC) in a game. (An NPC is a
game character that is never controlled by a human player and its behavior is
pre-defined by the game developer.) More specifically, let's assume we're
building a multi-player, war-themed game in which players control the soldiers.
In this game, we have a single NPC who serves as a medic, finding and reviving
wounded players. Lastly, let us assume that there are two teams, each with five
players and one NPC medic.

The behavior of a medic is quite complex. It first needs to avoid getting
injured, which requires detecting when it is in danger and moving to a safe
location. Second, it needs to be aware of which of its team members are injured
and require assistance. In the case of multiple injuries, it needs to assess the
degree of injury and decide who to help first. Lastly, a good medic will always
place itself in a position where it can quickly help its team members. Factoring
in all of these traits means that at every instance, the medic needs to measure
several attributes of the environment (e.g. position of team members, position
of enemies, which of its team members are injured and to what degree) and then
decide on an action (e.g. hide from enemy fire, move to help one of its
members). Given the large number of settings of the environment and the large
number of actions that the medic can take, defining and implementing such
complex behaviors by hand is challenging and prone to errors.

With ML-Agents, it is possible to _train_ the behaviors of such NPCs (called
**agents**) using a variety of methods. The basic idea is quite simple. We need
to define three entities at every moment of the game (called **environment**):

- **Observations** - what the medic perceives about the environment.
  Observations can be numeric and/or visual. Numeric observations measure
  attributes of the environment from the point of view of the agent. For our
  medic this would be attributes of the battlefield that are visible to it. For
  most interesting environments, an agent will require several continuous
  numeric observations. Visual observations, on the other hand, are images
  generated from the cameras attached to the agent and represent what the agent
  is seeing at that point in time. It is common to confuse an agent's
  observation with the environment (or game) **state**. The environment state
  represents information about the entire scene containing all the game
  characters. The agents observation, however, only contains information that
  the agent is aware of and is typically a subset of the environment state. For
  example, the medic observation cannot include information about an enemy in
  hiding that the medic is unaware of.
- **Actions** - what actions the medic can take. Similar to observations,
  actions can either be continuous or discrete depending on the complexity of
  the environment and agent. In the case of the medic, if the environment is a
  simple grid world where only their location matters, then a discrete action
  taking on one of four values (north, south, east, west) suffices. However, if
  the environment is more complex and the medic can move freely then using two
  continuous actions (one for direction and another for speed) is more
  appropriate.
- **Reward signals** - a scalar value indicating how well the medic is doing.
  Note that the reward signal need not be provided at every moment, but only
  when the medic performs an action that is good or bad. For example, it can
  receive a large negative reward if it dies, a modest positive reward whenever
  it revives a wounded team member, and a modest negative reward when a wounded
  team member dies due to lack of assistance. Note that the reward signal is how
  the objectives of the task are communicated to the agent, so they need to be
  set up in a manner where maximizing reward generates the desired optimal
  behavior.

After defining these three entities (the building blocks of a **reinforcement
learning task**), we can now _train_ the medic's behavior. This is achieved by
simulating the environment for many trials where the medic, over time, learns
what is the optimal action to take for every observation it measures by
maximizing its future reward. The key is that by learning the actions that
maximize its reward, the medic is learning the behaviors that make it a good
medic (i.e. one who saves the most number of lives). In **reinforcement
learning** terminology, the behavior that is learned is called a **policy**,
which is essentially a (optimal) mapping from observations to actions. Note that
the process of learning a policy through running simulations is called the
**training phase**, while playing the game with an NPC that is using its learned
policy is called the **inference phase**.

The ML-Agents toolkit provides all the necessary tools for using Unity as the
simulation engine for learning the policies of different objects in a Unity
environment. In the next few sections, we discuss how the ML-Agents toolkit
achieves this and what features it provides.

## Key Components

The ML-Agents toolkit is a Unity plugin that contains three high-level
components:

- **Learning Environment** - which contains the Unity scene and all the game
  characters.
- **Python API** - which contains all the machine learning algorithms that are
  used for training (learning a behavior or policy). Note that, unlike the
  Learning Environment, the Python API is not part of Unity, but lives outside
  and communicates with Unity through the External Communicator.
- **External Communicator** - which connects the Learning Environment with the
  Python API. It lives within the Learning Environment.

<p align="center">
  <img src="images/learning_environment_basic.png"
       alt="Simplified ML-Agents Scene Block Diagram"
       width="700" border="10" />
</p>

_Simplified block diagram of ML-Agents._

The Learning Environment contains an additional component that help
organize the Unity scene:

- **Agents** - which is attached to a Unity GameObject (any character within a
  scene) and handles generating its observations, performing the actions it
  receives and assigning a reward (positive / negative) when appropriate. Each
  Agent is linked to a Policy.

Every Learning Environment will always have one Agent for
every character in the scene. While each Agent must be linked to a Policy, it is
possible for Agents that have similar observations and actions to have
the same Policy type. In our sample game, we have two teams each with their own medic.
Thus we will have two Agents in our Learning Environment, one for each medic,
but both of these medics can have the same Policy. Note that these two
medics have the same Policy because their _space_ of observations and
actions are similar. This does not mean that at each instance they will have
identical observation and action _values_. In other words, the Policy defines the
space of all possible observations and actions, while the Agents connected to it
(in this case the medics) can each have their own, unique observation and action
values. If we expanded our game to include tank driver NPCs, then the Agent
attached to those characters cannot share a Policy with the Agent linked to the
medics (medics and drivers have different actions).

<p align="center">
  <img src="images/learning_environment_example.png"
       alt="Example ML-Agents Scene Block Diagram"
       border="10" />
</p>

_Example block diagram of ML-Agents toolkit for our sample game._

We have yet to discuss how the ML-Agents toolkit trains behaviors, and what role
the Python API and External Communicator play. Before we dive into those
details, let's summarize the earlier components. Each character is attached to
an Agent, and each Agent has a Policy. The Policy receives observations
and rewards from the Agent and returns actions. The Academy ensures that all the
Agents are in sync in addition to controlling environment-wide
settings.


## Training Modes

Given the flexibility of ML-Agents, there are a few ways in which training and
inference can proceed.

### Built-in Training and Inference

As mentioned previously, the ML-Agents toolkit ships with several
implementations of state-of-the-art algorithms for training intelligent agents.
More specifically, during training, all the medics in the
scene send their observations to the Python API through the External
Communicator. The Python API
processes these observations and sends back actions for each medic to take.
During training these actions are mostly exploratory to help the Python API
learn the best policy for each medic. Once training concludes, the learned
policy for each medic can be exported. Given that all our implementations are
based on TensorFlow, the learned policy is just a TensorFlow model file. Then
during the inference phase, we use the
TensorFlow model generated from the training phase. Now during the inference
phase, the medics still continue to generate their observations, but instead of
being sent to the Python API, they will be fed into their (internal, embedded)
model to generate the _optimal_ action for each medic to take at every point in
time.

To summarize: our built-in implementations are based on TensorFlow, thus, during
training the Python API uses the observations it receives to learn a TensorFlow
model. This model is then embedded within the Agent during inference.

The [Getting Started Guide](Getting-Started.md)
tutorial covers this training mode with the **3D Balance Ball** sample environment.

### Custom Training and Inference

In the previous mode, the Agents were used for training to generate
a TensorFlow model that the Agents can later use. However,
any user of the ML-Agents toolkit can leverage their own algorithms for
training. In this case, the behaviors of all the Agents in the scene
will be controlled within Python.
You can even turn your environment into a [gym.](../gym-unity/README.md)

We do not currently have a tutorial highlighting this mode, but you can
learn more about the Python API [here](Python-API.md).

## Flexible Training Scenarios

While the discussion so-far has mostly focused on training a single agent, with
ML-Agents, several training scenarios are possible. We are excited to see what
kinds of novel and fun environments the community creates. For those new to
training intelligent agents, below are a few examples that can serve as
inspiration:

- Single-Agent. A single agent, with its own reward
  signal. The traditional way of training an agent. An example is any
  single-player game, such as Chicken.
- Simultaneous Single-Agent. Multiple independent agents with independent reward
  signals with same `Behavior Parameters`. A parallelized version of the traditional
  training scenario, which can speed-up and stabilize the training process.
  Helpful when you have multiple versions of the same character in an
  environment who should learn similar behaviors. An example might be training a
  dozen robot-arms to each open a door simultaneously.
- Adversarial Self-Play. Two interacting agents with inverse reward signals.
  In two-player games, adversarial self-play can allow
  an agent to become increasingly more skilled, while always having the
  perfectly matched opponent: itself. This was the strategy employed when
  training AlphaGo, and more recently used by OpenAI to train a human-beating
  1-vs-1 Dota 2 agent.
- Cooperative Multi-Agent. Multiple interacting agents with a shared reward
  signal with same or different `Behavior Parameters`. In this
  scenario, all agents must work together to accomplish a task that cannot be
  done alone. Examples include environments where each agent only has access to
  partial information, which needs to be shared in order to accomplish the task
  or collaboratively solve a puzzle.
- Competitive Multi-Agent. Multiple interacting agents with inverse reward
  signals with same or different `Behavior Parameters`. In this
  scenario, agents must compete with one another to either win a competition, or
  obtain some limited set of resources. All team sports fall into this scenario.
- Ecosystem. Multiple interacting agents with independent reward signals with
  same or different `Behavior Parameters`. This scenario can be thought
  of as creating a small world in which animals with different goals all
  interact, such as a savanna in which there might be zebras, elephants and
  giraffes, or an autonomous driving simulation within an urban environment.

## Training Options

### Deep Reinforcement Learning

ML-Agents provide an implementation of two reinforcement learning algorithms:
* [Proximal Policy Optimization (PPO)](https://blog.openai.com/openai-baselines-ppo/)
* [Soft Actor-Critic (SAC)](https://bair.berkeley.edu/blog/2018/12/14/sac/)

PPO uses a neural network to approximate the ideal function that maps an agent's
observations to the best action an agent can take in a given state. The
ML-Agents PPO algorithm is implemented in TensorFlow and runs in a separate
Python process (communicating with the running Unity application over a socket).

In contrast with PPO, SAC is _off-policy_, which means it can learn from experiences collected
at any time during the past. As experiences are collected, they are placed in an
experience replay buffer and randomly drawn during training. This makes SAC
significantly more sample-efficient, often requiring 5-10 times less samples to learn
the same task as PPO. However, SAC tends to require more model updates. SAC is a
good choice for heavier or slower environments (about 0.1 seconds per step or more).

SAC is also a "maximum entropy" algorithm, and enables exploration in an intrinsic way.
Read more about maximum entropy RL [here](https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/).

### Curriculum Learning

Curriculum learning is a way of training a machine learning model where more
difficult aspects of a problem are gradually introduced in such a way that the
model is always optimally challenged. This idea has been around for a long time,
and it is how we humans typically learn. If you imagine any childhood primary
school education, there is an ordering of classes and topics. Arithmetic is
taught before algebra, for example. Likewise, algebra is taught before calculus.
The skills and knowledge learned in the earlier subjects provide a scaffolding
for later lessons. The same principle can be applied to machine learning, where
training on easier tasks can provide a scaffolding for harder tasks in the
future.

Imagine training the medic to to scale a wall to arrive at a wounded team member.
The starting point when training a medic to accomplish this task will be a random
policy. That starting policy will have the medic running in circles, and will
likely never, or very rarely scale the wall properly to revive their team member
(and achieve the reward). If we start with a simpler task, such as moving toward
an unobstructed team member, then the medic can easily learn to accomplish the task.
From there, we can slowly add to the difficulty of the task by increasing the size of
the wall until the medic can complete the initially near-impossible task of scaling the
wall. We have included an environment to demonstrate this with ML-Agents,
called [Wall Jump](Learning-Environment-Examples.md#wall-jump).

![Wall](images/curriculum.png)

_Demonstration of a curriculum training scenario in which a progressively taller
wall obstructs the path to the goal._

To see curriculum learning in action, observe the two learning curves below. Each
displays the reward over time for an agent trained using PPO with the same set of
training hyperparameters. The difference is that one agent was trained using the
full-height wall version of the task, and the other agent was trained using the
curriculum version of the task. As you can see, without using curriculum
learning the agent has a lot of difficulty. We think that by using well-crafted
curricula, agents trained using reinforcement learning will be able to
accomplish tasks otherwise much more difficult.

![Log](images/curriculum_progress.png)

The ML-Agents toolkit supports setting custom environment parameters within
the Academy. This allows elements of the environment related to difficulty or
complexity to be dynamically adjusted based on training progress.

### Curiosity for Sparse Rewards

### Imitation Learning

It is often more intuitive to simply demonstrate the behavior we want an agent
to perform, rather than attempting to have it learn via trial-and-error methods.
For example, instead of indirectly training a medic with the help
of a reward function, we can give the medic real world examples of observations
from the game and actions from a game controller to guide the medic's behavior.
Imitation Learning uses pairs of observations and actions from
a demonstration to learn a policy. See this
[video demo](https://youtu.be/kpb8ZkMBFYs) of imitation learning .

Imitation learning can either be used alone or in conjunction with reinforcement learning.
Especially in environments with sparse (i.e., infrequent or rare) rewards, the agent may never see
the reward and thus not learn from it. Curiosity (which is available in the toolkit)
helps the agent explore, but in some cases it is easier to show the agent how to
achieve the reward. In these cases, imitation learning combined with reinforcement
learning can dramatically reduce the time the agent takes to solve the environment.
For instance, on the [Pyramids environment](Learning-Environment-Examples.md#pyramids),
using 6 episodes of demonstrations can reduce training steps by more than 4 times.
See Behavioral Cloning + GAIL + Curiosity + RL below.

<p align="center">
  <img src="images/mlagents-ImitationAndRL.png"
       alt="Using Demonstrations with Reinforcement Learning"
       width="700" border="0" />
</p>

The ML-Agents Toolkit provides a way to learn directly from demonstrations, as well as use them
to help speed up reward-based training (RL). We include two algorithms called
Behavioral Cloning (BC) and Generative Adversarial Imitation Learning (GAIL).
In most scenarios, you can combine these two features.

* GAIL (Generative Adversarial Imitation Learning) uses an adversarial approach to
  reward your Agent for behaving similar to a set of demonstrations. To use GAIL, you can add the
  [GAIL reward signal](Reward-Signals.md#gail-reward-signal). GAIL can be
  used with or without environment rewards, and works well when there are a limited
  number of demonstrations.
* Behavioral Cloning (BC) trains the Agent's policy to exactly mimic the actions
  shown in a set of demonstrations.
  The BC feature can be enabled on the PPO or SAC trainers. As BC cannot generalize
  past the examples shown in the demonstrations, BC tends to work best when there exists demonstrations
  for nearly all of the states that the agent can experience, or in conjunction with GAIL and/or an extrinsic reward.

#### Recording Demonstrations

Demonstrations of agent behavior can be recorded from the Unity Editor,
and saved as assets. These demonstrations contain information on the
observations, actions, and rewards for a given agent during the recording session.
They can be managed in the Editor, as well as used for training with BC and GAIL.

### Memory-enhacened Agents using Recurrent Neural Networks

Have you ever entered a room to get something and immediately forgot what you
were looking for? Don't let that happen to your agents.

![Inspector](images/ml-agents-LSTM.png)

In some scenarios, agents must learn to remember the past in order to take the best
decision. When an agent only has partial observability of the environment, keeping
track of past observations can help the agent learn. Deciding what the agents should
remember in order to solve a task is not easy to do by hand, but our training algorithms
can learn to keep track of what is important to remember with
[LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory).

### Environment Parameter Randomization

An agent trained on a specific environment, may be unable to generalize to any
tweaks or variations in the environment (in machine learning this is referred to
as overfitting). This becomes problematic in cases where environments are instantiated
with varying objects or properties. One mechanism to alleviate this and train more
robust agents that can generalize to unseen variations of the environment
is to expose them to these variations during training. Similar to Curriculum Learning,
where environments become more difficult as the agent learns, the ML-Agents Toolkit provides
a way to randomly sample parameters of the environment during training. We refer to
this approach as **Environment Parameter Randomization**. For those familiar with
Reinforcement Learning research, this approach is based on the concept of Domain Randomization
(you can read more about it [here](https://arxiv.org/abs/1703.06907)). By using parameter
randomization during training, the agent can be better suited to adapt (with higher performance)
to future unseen variations of the environment.

_Example of variations of the 3D Ball environment._

Ball scale of 0.5          |  Ball scale of 4
:-------------------------:|:-------------------------:
![](images/3dball_small.png)  |  ![](images/3dball_big.png)


## Additional Features

Beyond the flexible training scenarios available, the ML-Agents Toolkit includes
additional features which improve the flexibility and interpretability of the
training process.

- **Concurrent Unity Instances** - we enable developers to run concurrent, parallel
  instances of the Unity executable during training. For certain scenarios, this
  should speed up training.

- **Monitoring Agent’s Decision Making** - Since communication in ML-Agents is a
  two-way street, we provide an Agent Monitor class in Unity which can display
  aspects of the trained Agent, such as the Agents perception on how well it is
  doing (called **value estimates**) within the Unity environment itself. By
  leveraging Unity as a visualization tool and providing these outputs in
  real-time, researchers and developers can more easily debug an Agent’s
  behavior.

- **Complex Visual Observations** - Unlike other platforms, where the agent’s
  observation might be limited to a single vector or image, the ML-Agents
  toolkit allows multiple cameras to be used for observations per agent. This
  enables agents to learn to integrate information from multiple visual streams.
  This can be helpful in several scenarios such as training a self-driving car
  which requires multiple cameras with different viewpoints, or a navigational
  agent which might need to integrate aerial and first-person visuals. You can
  learn more about adding visual observations to an agent
  [here](Learning-Environment-Design-Agents.md#multiple-visual-observations).

## Summary and Next Steps

To briefly summarize: The ML-Agents toolkit enables games and simulations built
in Unity to serve as the platform for training intelligent agents. It is
designed to enable a large variety of training modes and scenarios and comes
packed with several features to enable researchers and developers to leverage
(and enhance) machine learning within Unity.

To help you use ML-Agents, we've created several in-depth tutorials for
[installing ML-Agents](Installation.md),
[getting started](Getting-Started.md) with the 3D Balance Ball
environment (one of our many
[sample environments](Learning-Environment-Examples.md)) and
[making your own environment](Learning-Environment-Create-New.md).
