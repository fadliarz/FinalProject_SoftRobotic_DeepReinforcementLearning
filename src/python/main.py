import warnings

warnings.filterwarnings("ignore")

from src.python.noise import OUActionNoise
from src.python.buffer import Buffer

from src.python.network.KerasActorNetwork import KerasActorNetwork
from src.python.network.KerasCriticNetwork import KerasCriticNetwork

from src.python.environment.Environment import Environment

from src.python.utility import Utility

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

import numpy as np
import matplotlib.pyplot as plt

"""
    CONFIGURATION
        buffer_capacity     : - 
        actor_lr            : learning rate for actor network
        critic_lr           : learning rate for critic network    
        std_dev             :
        total_episodes      :
        gamma               : discount factor for future rewards
        tau                 : used to update target networks
        state_size          :
        action_size         :
        min_action          :
        max_action          :
        manipulator_length  : soft robotic length in meter
"""
# for buffer
buffer_capacity = 10000
critic_lr = 0.002
actor_lr = 0.001

# for noise
std_dev = 0.2

max_steps = 10
total_episodes = 100
gamma = 0.99
tau = 0.005

state_size = 5
action_size = 2
action_range = (-50, 50)
position_range = [(-0.1662, 0.1662), (-0.3931, 0.3931)]
manipulator_length = 0.3  # m

ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

# randomly initialize critic and actor networks
critic = KerasCriticNetwork(state_size=state_size, action_size=action_size)
actor = KerasActorNetwork(min_action=action_range[0], max_action=action_range[1], state_size=state_size,
                          action_size=action_size)

# initialize target network with weights
target_critic = KerasCriticNetwork(state_size=state_size, action_size=action_size)
target_actor = KerasActorNetwork(min_action=action_range[0], max_action=action_range[1], state_size=state_size,
                                 action_size=action_size)
target_actor.set_weights(actor.get_weights())
target_critic.set_weights(critic.get_weights())

# initialize replay buffer
buffer = Buffer(state_size=state_size, action_size=action_size, critic_lr=critic_lr, actor_lr=actor_lr, gamma=gamma,
                buffer_capacity=buffer_capacity)

ep_reward_list = []
avg_reward_list = []

env = Environment(
    action_size, state_size,
    action_range, position_range,
    max_steps=max_steps, threshold=0.028 * manipulator_length, penalty_term=1, reward_constant=10, verbose=False
)

for ep in range(total_episodes):
    prev_state, _ = env.reset()
    episodic_reward = 0

    while True:
        tf_prev_state = keras.ops.expand_dims(
            keras.ops.convert_to_tensor(prev_state), 0
        )

        action = actor.policy(tf_prev_state, ou_noise)

        state, reward, done, truncated = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn(actor, critic, target_actor, target_critic)

        Utility.update_target(target_actor, actor, tau)
        Utility.update_target(target_critic, critic, tau)

        if done or truncated:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()
