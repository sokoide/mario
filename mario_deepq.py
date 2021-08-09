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

class EnvironmentSimulator(py_environment.PyEnvironment):
    def __init__(self, args):
        super(EnvironmentSimulator, self).__init__()
        # state
        # 0: empty, 1: wall, 2: enemy, 3: mario
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(16,13,1), dtype=np.float32, minimum=0, maximum=3
        )
        # action
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(actions)-1
        )
        self._screen = np.zeros(shape=(16,13,1), dtype=np.float32)
        self._args = args
        self._gym_env = None
        self._reset()

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def _reset(self):
        if self._gym_env is not None:
            self._gym_env.close()
        clean_fceux()
        self._gym_env = gym.make(self._args.stage)
        self._gym_env.reset()

        self._state = [0,1,2,3,4,5] # not used
        return ts.restart(np.array(self._screen, dtype=np.float32))

    def _step(self, action):
        reward = 0

        # TODO
        aid = actions[action]
        observation, reward, done, info = self._gym_env.step(aid)
        # print('o:{}, r:{},d:{},i:{}'.format(observation, reward, done, info))
        _screen_state = np.reshape(observation, (16,13,1))

        if done and info['life'] == 0:
            # killed
            reward = 0
            return ts.termination(np.array(_screen_state, dtype=np.float32), reward=reward)
        else:
            return ts.transition(np.array(_screen_state, dtype=np.float32), reward=reward, discount=1)



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
                keras.layers.Conv2D(64, 3, 1, activation='relu', padding='same'),
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



def play_mario(env, gene):
    # returns (score, cleared)
    FRAMES = 4
    score = 0
    observation = env.reset()

    for action_id in gene:
        for i in range(FRAMES):
            action = actions[action_id]
            observation, reward, done, info = env.step(action)
            # print('r:{}, d:{}, i:{}'.format(reward, done, info))
            score = max(score, info['distance'])
            if done and info['life'] == 0:
                return (score, False)
            elif done:
                return (score, True)
    return (score, False)


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
    args = parser.parse_args()
    return args


def replay(args):
    pass
    # l = sorted(genes, key=lambda k: k['score'])
    # first = l[-1]
    # print(first)
    # print('len:{}'.format(len(first['gene'])))
    # env = gym.make(args.stage)
    # score, cleared = play_mario(env, first['gene'])
    # print('score: {}, cleared: {}'.format(score, cleared))
    # clean_fceux()
    # env.close()


def main():
    # 1000 steps == 400 TIME periods in Mario
    NUM_STEPS = 1000
    FRAMES = 4

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
        driver.run(maximum_iterations=100)
        print('* driver.run completed')

    num_episodes = 1000
    epsilon = np.linspace(start=0.2, stop=0.0, num=num_episodes+1)#ε-greedy法用
    tf_policy_saver = policy_saver.PolicySaver(policy=agent.policy)#ポリシーの保存設定

    print('* episodes')
    try:
        for episode in range(num_episodes):
            episode_rewards = 0#報酬の計算用
            episode_average_loss = []#lossの計算用
            policy._epsilon = epsilon[episode]#エピソードに合わせたランダム行動の確率
            time_step = env.reset()#環境の初期化

            for t in range(NUM_STEPS):
                policy_step = policy.action(time_step)#状態から行動の決定
                for i in range(FRAMES):
                    next_time_step = env.step(policy_step.action)#行動による状態の遷移

                traj =  trajectory.from_transition(time_step, policy_step, next_time_step)#データの生成
                replay_buffer.add_batch(traj)#データの保存

                experience, _ = next(iterator)#学習用データの呼び出し
                loss_info = agent.train(experience=experience)#学習

                R = next_time_step.reward.numpy().astype('int').tolist()[0]
                episode_average_loss.append(loss_info.loss.numpy())
                episode_rewards += R

                if next_time_step.is_last()[0]:
                    break

                time_step = next_time_step#次の状態を今の状態に設定

            print(f'* Episode:{episode:4.0f}, Step:{t:3.0f}, R:{episode_rewards:3.0f}, AL:{np.mean(episode_average_loss):.4f}, PE:{policy._epsilon:.6f}')
            train_checkpointer.save(global_step=agent.train_step_counter)
            if episode%10 == 0:
                tf_policy_saver.save(export_dir='policy%08d' % episode)
        # if args.replay:
        #     replay(args)
        #     return
        #
        # cleared = False
        # while cleared == False:
        #     pass
            # print('[{}] * generation: {}'.format(time.strftime('%H:%M:%S'), gen))
            # for gene in genes:
            #     if gene['score'] > 0:
            #         print('result: {} (prev)'.format(gene['score']))
            #         continue
            #     # recreate env every episode to avoid 5-7 second dealy at start up after Mario is killed by Kuribo
            #     env = gym.make(args.stage)
            #     gene['score'], cleared = play_mario(env, gene['gene'])
            #     print('score: {}, cleared: {}'.format(gene['score'], cleared))
            #     clean_fceux()
            #     env.close()
            # print_generation_result(gen)
            # save_genes(gen)
            # change_generation()
            # gen += 1
    except KeyboardInterrupt:
        clean_fceux()


if __name__ == '__main__':
    main()
