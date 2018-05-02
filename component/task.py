#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import gym
import os
import numpy as np
from .atari_wrapper import *
import multiprocessing as mp
import sys
from .bench import Monitor
from utils import *
import datetime
import uuid

class BaseTask:
    def set_monitor(self, env, log_dir, worker_id=None):
        if log_dir is None:
            return env
        mkdir(log_dir)
        prefix_str = str(uuid.uuid4()) if worker_id is None else str(worker_id)
        return Monitor(env, os.path.join(log_dir, prefix_str))
        # return Monitor(env, '%s/%s' % (log_dir, uuid.uuid4()))

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def seed(self, random_seed):
        return self.env.seed(random_seed)

class ClassicalControl(BaseTask):
    def __init__(self, name='CartPole-v0', max_steps=200, log_dir=None):
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(self.name)
        self.env._max_episode_steps = max_steps
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

class PixelAtari(BaseTask):
    def __init__(self, name, seed=0, log_dir=None,
                 frame_skip=4, history_length=4, dataset=False, worker_id=None):
        BaseTask.__init__(self)
        env = make_atari(name, frame_skip)
        env.seed(seed)
        if dataset:
            env = DatasetEnv(env)
            self.dataset_env = env
        env = self.set_monitor(env, log_dir, worker_id)
        env = wrap_deepmind(env, history_length=history_length)
        self.env = env
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape
        self.name = name

class RamAtari(BaseTask):
    def __init__(self, name, no_op, frame_skip, log_dir=None):
        BaseTask.__init__(self)
        self.name = name
        env = gym.make(name)
        assert 'NoFrameskip' in env.spec.id
        env = self.set_monitor(env, log_dir)
        env = EpisodicLifeEnv(env)
        env = NoopResetEnv(env, noop_max=no_op)
        env = SkipEnv(env, skip=frame_skip)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        self.env = env
        self.action_dim = self.env.action_space.n
        self.state_dim = 128

class Pendulum(BaseTask):
    def __init__(self, log_dir=None):
        BaseTask.__init__(self)
        self.name = 'Pendulum-v0'
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(2 * action, -2, 2))

class Box2DContinuous(BaseTask):
    def __init__(self, name, log_dir=None):
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(action, -1, 1))

class Roboschool(BaseTask):
    def __init__(self, name, log_dir=None):
        import roboschool
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(action, -1, 1))

class DMControl(BaseTask):
    def __init__(self, domain_name, task_name, log_dir=None):
        from dm_control import suite
        import dm_control2gym
        BaseTask.__init__(self)

        self.name = domain_name + '_' + task_name
        self.env = dm_control2gym.make(domain_name, task_name)

        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

class GymRobotics(BaseTask):
    def __init__(self, name, log_dir=None):
        BaseTask.__init__(self)

        self.name = name
        self.env = gym.make(name)

        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = len(self.flatten_state(self.env.reset()))
        self.env = self.set_monitor(self.env, log_dir)

    def flatten_state(self, state):
        flat = []
        for key, value in state.items():
            flat.append(state[key])
        flat = np.concatenate(flat, axis=0)
        return flat

    def reset(self):
        return self.flatten_state(self.env.reset())

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        return self.flatten_state(next_state), reward, done, _

def sub_task(parent_pipe, pipe, task_fn, rank, log_dir):
    np.random.seed()
    seed = np.random.randint(0, sys.maxsize)
    parent_pipe.close()
    if rank is None:
        task = task_fn(log_dir=log_dir)
    else:
        task = task_fn(log_dir=log_dir, worker_id=rank)
    task.seed(seed)
    while True:
        op, data = pipe.recv()
        if op == 'step':
            ob, reward, done, info = task.step(data)
            if done:
                ob = task.reset()
            pipe.send([ob, reward, done, info])
        elif op == 'reset':
            pipe.send(task.reset())
        elif op == 'exit':
            pipe.close()
            return
        else:
            assert False, 'Unknown Operation'

def stack_lists_of_ndarray(x):
    """
    :param x: a tuple of lists of numpy arrays
    :return: a list of numpy arrays
    """
    assert type(x[0][0]) == np.ndarray
    return [np.stack(x[i][j] for i in range(len(x))) for j in range(len(x[0]))]

class ParallelizedTask:
    def __init__(self, task_fn, num_workers, log_dir=None, worker_ids=None):
        self.task_fn = task_fn
        self.task = task_fn(log_dir=None, worker_id=(worker_ids[0] if worker_ids is not None else None))
        self.name = self.task.name
        if log_dir is not None:
            mkdir(log_dir)
        if worker_ids is None:
            worker_ids = [None] * num_workers
        self.pipes, worker_pipes = zip(*[mp.Pipe() for _ in range(num_workers)])
        args = [(p, wp, task_fn, worker_ids[rank], log_dir)
                for rank, (p, wp) in enumerate(zip(self.pipes, worker_pipes))]
        self.workers = [mp.Process(target=sub_task, args=arg) for arg in args]
        for p in self.workers: p.start()
        for p in worker_pipes: p.close()
        self.state_dim = self.task.state_dim
        self.action_dim = self.task.action_dim

    def step(self, actions):
        for pipe, action in zip(self.pipes, actions):
            pipe.send(('step', action))
        results = [p.recv() for p in self.pipes]
        # results = map(lambda x: np.stack(x), zip(*results))
        results = map(lambda x: (np.stack(x) if type(x[0]) != list else stack_lists_of_ndarray(x)), zip(*results))
        return results

    def reset(self, i=None):
        if i is None:
            for pipe in self.pipes:
                pipe.send(('reset', None))
            results = [p.recv() for p in self.pipes]
        else:
            self.pipes[i].send(('reset', None))
            results = self.pipes[i].recv()
        # return np.stack(results)
        results = np.stack(results) if type(results[0]) != list else stack_lists_of_ndarray(results)
        return results

    def close(self):
        for pipe in self.pipes:
            pipe.send(('exit', None))
        for p in self.workers: p.join()

    def normalize_state(self, state):
        return self.task.normalize_state(state)
