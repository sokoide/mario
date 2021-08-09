import gym
import ppaquette_gym_super_mario
import numpy as np
import signal
import os
import time
import pickle
import copy
import argparse
import tensorflow as tf

from tensorflow import keras

from tf_agents.environments import gym_wrapper, py_environment, tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver


# global

# e.g. action = [0, 0, 0, 1, 1, 0]
# [up, left, down, right, A, B]
# would activate right (4th element), and A (5th element)
actions = [
    [0, 0, 0, 0, 0, 0],  # no input
    [0, 0, 0, 1, 0, 0],  # right
    [0, 0, 0, 1, 1, 0],  # right + jump
    [0, 0, 0, 1, 0, 1],  # right dash
    [0, 0, 0, 1, 1, 1],  # right dash + jump
    [0, 0, 0, 0, 1, 0],  # jump
    [0, 1, 0, 0,  0, 0],  # left
]
episode_distance = 0
max_distance = 0

class EnvironmentSimulator(py_environment.PyEnvironment):
    def __init__(self, args):
        global max_distance

        super(EnvironmentSimulator, self).__init__()
        max_distance = 0
        # state
        # 0: empty, 1: wall, 2: enemy, 3: mario
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(13, 16, 1), dtype=np.float32, minimum=0, maximum=3
        )
        # action
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(actions)-1
        )
        self._screen = np.zeros(shape=(13, 16, 1), dtype=np.float32)
        self._args = args
        self._gym_env = None
        self._reset()

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def _reset(self):
        global max_distance
        global episode_distance
        episode_distance = 0

        if self._gym_env is not None:
            self._gym_env.close()
        clean_fceux()
        self._gym_env = gym.make(self._args.stage)
        self._gym_env.reset()

        return ts.restart(self._screen)

    def _step(self, action):
        global max_distance
        global episode_distance

        aid = actions[action]
        observation, reward, done, info = self._gym_env.step(aid)
        _screen_state = np.array(observation[:, :, np.newaxis], dtype=np.float32)
        episode_distance =  max(episode_distance, int(info['distance']))
        max_distance =  max(max_distance, int(info['distance']))

        if reward==0:
            reward = -1

        if done and info['life'] == 0:
            # killed
            reward = -100
            # print('r:{}'.format(reward))
            return ts.termination(_screen_state, reward=reward)
        elif done:
            # cleared
            reward += 100
            # print('r:{}'.format(reward))
            return ts.termination(_screen_state, reward=reward)
        else:
            # print('r:{},'.format(reward), end='')
            return ts.transition(_screen_state, reward=reward, discount=1)


class QNetwork(network.Network):
    def __init__(self, observation_spec, action_spec, n_hidden_channels=2, name='QNetwork'):
        super(QNetwork, self).__init__(
            input_tensor_spec=observation_spec,
            state_spec=(),
            name=name
        )
        n_action = action_spec.maximum - action_spec.minimum + 1

        self.model = keras.Sequential(
            [
                keras.layers.Conv2D(16, 3, 1, activation='relu', padding='same'),
                keras.layers.MaxPool2D(pool_size=(2, 2)),
                keras.layers.Conv2D(32, 3, 1, activation='relu', padding='same'),
                keras.layers.MaxPool2D(pool_size=(2, 2)),
                keras.layers.Conv2D(64, 3, 1, activation='relu', padding='same'),
                keras.layers.MaxPool2D(pool_size=(2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(n_action),
            ]
        )

    def call(self, observation, step_type=None, network_state=(), training=True):
        actions = self.model(observation, training=training)
        return actions, network_state


def kill_process(path: str):
    for line in os.popen("ps -ef | grep '{}' | grep -v grep".format(path)):
        pid = int(line.split()[1])
        try:
            os.kill(pid, signal.SIGKILL)
        except:
            pass


def clean_fceux():
    hb_prefix = os.getenv('HOMEBREW_PREFIX', '/usr/local')
    kill_process(os.path.join(hb_prefix, 'bin/fceux'))
    kill_process(os.path.join(hb_prefix, 'Cellar/fceux/2.4.0/libexec/fceux'))



def parse_args():
    parser = argparse.ArgumentParser(description='Deep Q lerning  Mario resolver')
    parser.add_argument('--replay', action=argparse.BooleanOptionalAction,
                        help='replay the best one in the generation')
    parser.add_argument('--kill', action=argparse.BooleanOptionalAction,
                        help='kill fceux processes')
    parser.add_argument('--continue', dest='cont', action=argparse.BooleanOptionalAction,
                        help='continue learning from the latest checkpoint')
    parser.add_argument('--stage', dest='stage', type=str, default='ppaquette/SuperMarioBros-1-1-Tiles-v0',
                        help='stage (default: ppaquette/SuperMarioBros-1-1-Tiles-v0 )')
    parser.add_argument('--episodes', dest='episodes', type=int, default=1000,
                        help='number of episodes')
    parser.add_argument('--policy', dest='policy', type=int, default=-1,
                        help='saved policy number to replay')
    args = parser.parse_args()
    return args


def replay(policy, env):
    FRAMES = 4
    # 1000 steps == 400 TIME periods in Mario
    NUM_STEPS = 1000 * FRAMES
    episode_rewards = 0

    time_step = env.reset()
    for t in range(NUM_STEPS):
        policy_step = policy.action(time_step)
        for i in range(FRAMES):
            next_time_step = env.step(policy_step.action)
            if next_time_step.is_last()[0]:
                break
        S = time_step.observation.numpy().tolist()[0]
        A = policy_step.action.numpy().tolist()[0]
        R = next_time_step.reward.numpy().astype('int').tolist()[0]
        print('A:{}, R:{}'.format(A, R))
        episode_rewards += R

        time_step = next_time_step
    print(f'Rewards:{episode_rewards}')
    env.close()


def main():
    FRAMES = 4
    # 1000 steps == 400 TIME periods in Mario
    NUM_STEPS = 1000 * FRAMES

    args = parse_args()

    if args.kill:
        clean_fceux()
        return

    py_env = EnvironmentSimulator(args)
    env = tf_py_environment.TFPyEnvironment(py_env)
    primary_network = QNetwork(env.observation_spec(), env.action_spec())

    # agent
    n_step_update = 1
    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=primary_network,
        optimizer=keras.optimizers.Adam(learning_rate=1e-2, epsilon=1e-2),
        n_step_update=n_step_update,
        epsilon_greedy=1.0,
        target_update_tau=1.0,
        target_update_period=10,
        gamma=0.9,
        td_errors_loss_fn = common.element_wise_squared_loss,
        train_step_counter = tf.Variable(0)
      )
    agent.initialize()
    agent.train = common.function(agent.train)

    # policy
    if args.policy >= 0:
        policy_dir = 'policy%08d' % args.policy
        policy = tf.compat.v2.saved_model.load(policy_dir)
        replay(policy, env)
        return

    policy = agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=agent.collect_data_spec,
      batch_size=env.batch_size,
      max_length=10**6
    )
    dataset = replay_buffer.as_dataset(
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
      sample_batch_size=32,
      num_steps=n_step_update+1
    ).prefetch(tf.data.experimental.AUTOTUNE)
    iterator = iter(dataset)

    env.reset()

    # check pointer
    train_checkpointer = common.Checkpointer(
      ckpt_dir='checkpointer',
      max_to_keep=3,
      agent=agent,
      policy=agent.policy,
      replay_buffer=replay_buffer,
      global_step=agent.train_step_counter
    )

    if args.cont:
        print('resotoring checkopoint...')
        train_checkpointer.initialize_or_restore()
        print('checkopoint restored')
    else:
        # data
        print('* data')
        driver = dynamic_episode_driver.DynamicEpisodeDriver(
          env,
          policy,
          observers=[replay_buffer.add_batch],
          num_episodes = 10,
        )
        print('* driver.run')
        driver.run(maximum_iterations=80)
        print('* driver.run completed')

    num_episodes = args.episodes
    epsilon = np.linspace(start=0.5, stop=0.0, num=num_episodes+1)
    tf_policy_saver = policy_saver.PolicySaver(policy=agent.policy)

    print('* episodes')
    try:
        for episode in range(1, num_episodes+1):
            episode_rewards = 0
            episode_average_loss = []
            policy._epsilon = epsilon[episode]
            time_step = env.reset()

            for t in range(NUM_STEPS):
                policy_step = policy.action(time_step)
                for i in range(FRAMES):
                    next_time_step = env.step(policy_step.action)
                    if next_time_step.is_last()[0]:
                        break

                traj =  trajectory.from_transition(time_step, policy_step, next_time_step)
                replay_buffer.add_batch(traj)

                experience, _ = next(iterator)
                loss_info = agent.train(experience=experience)

                R = next_time_step.reward.numpy().astype('int').tolist()[0]
                episode_average_loss.append(loss_info.loss.numpy())
                episode_rewards += R

                if next_time_step.is_last()[0]:
                    break

                time_step = next_time_step

            print(f'* Episode:{episode:4.0f}/{num_episodes:d}, Epsilon:{policy._epsilon:.2f}, MaxDistance:{max_distance:d}, Distance:{episode_distance:d}, Step:{t:3.0f}, R:{episode_rewards:3.0f}, AL:{np.mean(episode_average_loss):.4f}, PE:{policy._epsilon:.6f}')
            train_checkpointer.save(global_step=agent.train_step_counter)
            if episode > 0 and episode%10 == 0:
                tf_policy_saver.save(export_dir='policy%08d' % episode)
    except KeyboardInterrupt:
        clean_fceux()


if __name__ == '__main__':
    main()
