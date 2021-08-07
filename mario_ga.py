import gym
import ppaquette_gym_super_mario
import numpy as np
import signal
import os
import time
from multiprocessing import Lock


def kill_process(path: str):
    for line in os.popen("ps -ef | grep '{}' | grep -v grep".format(path)):
        pid = int(line.split()[1])
        print(line)
        print('kill -9 {}'.format(pid))
        try:
            os.kill(pid, signal.SIGKILL)
        except:
            pass


def clean_fceux():
    kill_process('/opt/homebrew/bin/fceux')
    kill_process('/opt/homebrew/Cellar/fceux/2.4.0/libexec/fceux')


def get_action():
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
    return actions[np.random.randint(0, len(actions))]


def make_random_gene(steps):
    gene = []
    for _ in range(steps):
        gene.append(get_action())
    return gene


def play_mario(env, gene):
    FRAMES = 4
    score = 0
    observation = env.reset()
    print('reset')

    for action in gene:
        for i in range(FRAMES):
            # action = np.random.randint(0, 1+1, 6)
            # action = [0, 0, 0, 1, 0, 1]
            # action = get_action()
            observation, reward, done, info = env.step(action)
            # print("action=", action)
            # print("observation=", observation)
            # print("reward=", reward)
            # print("done=", done)
            # print("info=", info)
            score = max(score, info['distance'])
            if done:
                print('done')
                return score
    return score


def main():
    for episode in range(5):
        # recreate env every episode to avoid 5-7 second dealy at start up after Mario is killed by Kuribo
        env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
        gene = make_random_gene(100)
        score = play_mario(env, gene)
        print('score: {}'.format(score))
        clean_fceux()
        env.close()


if __name__ == '__main__':
    main()
