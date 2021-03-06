import gym
import ppaquette_gym_super_mario
import numpy as np
import signal
import os
import time
import pickle
import copy
import argparse

# global
genes = []

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

NUM_GENES = 10
NUM_STEPS = 1000  # 1000 steps == 400 TIME periods in Mario


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


def make_random_gene(steps):
    gene = []
    for _ in range(steps):
        gene.append(np.random.randint(0, len(actions)))
    return gene


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


def print_generation_result(gen: int):
    l = sorted(genes, key=lambda k: k['score'])
    first = l[-1]
    print('* best score: {}'.format(first['score']))


def change_generation():
    global genes

    l = sorted(genes, key=lambda k: k['score'])
    first = l[-1]
    second = l[-2]

    new_genes = []

    # DNA 1-2: keep the best 2
    new_genes.append(first)
    new_genes.append(second)

    # DNA 3-4: mutate the best one
    for i in range(2):
        new_gene = copy.deepcopy(first['gene'])
        for i in range(0, len(new_gene)//100):
            r = np.random.randint(0, len(new_gene))
            new_gene[r] = np.random.randint(0, len(actions))
        new_genes.append({'gene': new_gene, 'score': 0})

    # DNA 5-10: crossover
    for i in range(len(genes)-len(new_genes)):
        p = np.random.randint(0, len(first['gene']))
        r = np.random.randint(1, 5)
        r2 = np.random.randint(1, 5)
        s1 = l[-1 * r]
        s2 = l[-1 * r2]
        new_gene = s1['gene'][:p] + s2['gene'][p:]
        new_genes.append({'gene': new_gene, 'score': 0})

    # then mutate
    for i in range(3, len(new_genes)):
        gene = new_genes[i]['gene']
        for j in range(0, len(gene)//100):
            r = np.random.randint(0, len(gene))
            gene[r] = np.random.randint(0, len(actions))
    genes = new_genes


def save_genes(gen):
    filepath = 'gene%04d.pkl' % gen
    with open(filepath, 'wb') as tf:
        pickle.dump(genes, tf)


def load_genes(gen, steps):
    global genes

    filepath = 'gene%04d.pkl' % gen
    with open(filepath, 'rb') as tf:
        new_genes = pickle.load(tf)

    for new_gene in new_genes:
        if len(new_gene['gene']) < steps:
            # add spteps
            for i in range(steps - len(new_gene['gene'])):
                new_gene['gene'].append(np.random.randint(0, len(actions)))
            new_gene['score'] = 0
        elif len(new_gene['gene']) > steps:
            new_gene['gene'] = new_gene['gene'][:steps]
            new_gene['score'] = 0
    genes = new_genes


def parse_args():
    parser = argparse.ArgumentParser(description='GA Mario resolver')
    parser.add_argument('--gen', dest='gen', type=int, default=0,
                        help='start from the generation')
    parser.add_argument('--replay', action=argparse.BooleanOptionalAction,
                        help='replay the best one in the generation')
    parser.add_argument('--kill', action=argparse.BooleanOptionalAction,
                        help='kill fceux processes')
    parser.add_argument('--stage', dest='stage', type=str, default='ppaquette/SuperMarioBros-1-1-Tiles-v0',
                        help='stage (default: ppaquette/SuperMarioBros-1-1-Tiles-v0 )')
    args = parser.parse_args()
    return args


def replay(args):
    l = sorted(genes, key=lambda k: k['score'])
    first = l[-1]
    print(first)
    print('len:{}'.format(len(first['gene'])))
    env = gym.make(args.stage)
    score, cleared = play_mario(env, first['gene'])
    print('score: {}, cleared: {}'.format(score, cleared))
    clean_fceux()
    env.close()


def main():
    global genes

    args = parse_args()

    if args.kill:
        clean_fceux()
        return

    gen = args.gen
    if gen > 0:
        load_genes(gen, NUM_STEPS)
    else:
        for i in range(NUM_GENES):
            genes.append({'gene': make_random_gene(NUM_STEPS), 'score': 0})
    if gen <= 0:
        gen = 1

    try:
        if args.replay:
            replay(args)
            return

        cleared = False
        while cleared == False:
            print('[{}] * generation: {}'.format(time.strftime('%H:%M:%S'), gen))
            for gene in genes:
                if gene['score'] > 0:
                    print('result: {} (prev)'.format(gene['score']))
                    continue
                # recreate env every episode to avoid 5-7 second dealy at start up after Mario is killed by Kuribo
                env = gym.make(args.stage)
                gene['score'], cleared = play_mario(env, gene['gene'])
                print('score: {}, cleared: {}'.format(gene['score'], cleared))
                clean_fceux()
                env.close()
            print_generation_result(gen)
            save_genes(gen)
            change_generation()
            gen += 1
    except KeyboardInterrupt:
        clean_fceux()


if __name__ == '__main__':
    main()
