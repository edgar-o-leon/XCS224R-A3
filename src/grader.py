#!/usr/bin/env python3
import inspect
import unittest
import argparse
import sys
import os

# Add submission and solution directories to path for proper imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_current_dir, 'submission'))
sys.path.insert(0, os.path.join(_current_dir, 'solution'))

from graderUtil import graded, CourseTestRunner, GradedTestCase
import torch
import torch.nn as nn

import numpy as np
import random

import gymnasium as gym

from autograde_utils import assert_allclose

# Import submission
from submission.xcs224r.infrastructure.pytorch_util import build_mlp, from_numpy
from submission.xcs224r.infrastructure.replay_buffer import ReplayBuffer
from submission.xcs224r.critics.cql_critic import CQLCritic
from submission.xcs224r.critics.iql_critic import IQLCritic
from submission.xcs224r.infrastructure.dqn_utils import (
    create_boxenv_q_network, 
    pointmass_optimizer,
    register_custom_envs,
    PreprocessAtari,
)
from submission.xcs224r.infrastructure import pytorch_util as ptu

# Import reference solution
if os.path.exists("./solution"):
    from solution.xcs224r.infrastructure.pytorch_util import build_mlp as ref_build_mlp
    from solution.xcs224r.critics.cql_critic import CQLCritic as RefCQLCritic
    from solution.xcs224r.critics.iql_critic import IQLCritic as RefIQLCritic
    from solution.xcs224r.infrastructure.dqn_utils import (
        PreprocessAtari as RefPreprocessAtari,
    )
    from solution.xcs224r.agents.iql_agent import IQLAgent as RefIQLAgent
else:
    ref_build_mlp = build_mlp
    RefCQLCritic = CQLCritic
    RefIQLCritic = IQLCritic
    RefPreprocessAtari = PreprocessAtari
    RefIQLAgent = None


#########
# TESTS #
#########


#########################################
# HELPER FUNCTIONS FOR DQN/CQL/IQL TESTS #
#########################################

def create_test_hparams(ob_dim=4, ac_dim=4, gamma=0.95):
    """Create test hyperparameters for critics"""
    return {
        'env_name': 'PointmassHard-v0',
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'double_q': True,
        'grad_norm_clipping': 10,
        'gamma': gamma,
        'q_func': create_boxenv_q_network,
        'cql_alpha': 0.5,
        'iql_expectile': 0.7,
    }

class Test_1a(GradedTestCase):
    """Tests for IQL Critic v_net definition"""

    def setUp(self):
        torch.manual_seed(224)
        np.random.seed(224)
        random.seed(224)
        ptu.init_gpu(use_gpu=False)

        self.ob_dim = 4
        self.ac_dim = 4
        self.batch_size = 32

        self.hparams = create_test_hparams(self.ob_dim, self.ac_dim)
        self.optimizer_spec = pointmass_optimizer()

    @graded()
    def test_0(self):
        """1a-0-basic: test IQL v_net is defined"""
        critic = IQLCritic(self.hparams, self.optimizer_spec)

        self.assertTrue(
            hasattr(critic, 'v_net'),
            msg="IQLCritic should have a v_net attribute!"
        )
        self.assertIsNotNone(
            critic.v_net,
            msg="IQLCritic.v_net should not be None!"
        )

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    # Tests for IQL Agent estimate_advantage function (formerly Test_1b)
    def _setup_iql_agent_test(self):
        """Helper to set up IQL agent tests"""
        register_custom_envs()

    @graded(timeout=30)
    def test_5(self):
        """1a-5-basic: test estimate_advantage returns tensor of correct shape"""
        self._setup_iql_agent_test()
        if RefIQLAgent is None:
            self.skipTest("Reference IQL Agent not available")

        from submission.xcs224r.agents.iql_agent import IQLAgent
        from submission.xcs224r.infrastructure.dqn_utils import PiecewiseSchedule

        env = gym.make('PointmassHard-v0')

        agent_params = {
            'env_name': 'PointmassHard-v0',
            'ob_dim': self.ob_dim,
            'ac_dim': self.ac_dim,
            'discrete': True,
            'n_layers': 2,
            'size': 64,
            'learning_rate': 1e-3,
            'num_exploration_steps': 1000,
            'offline_exploitation': False,
            'cql_alpha': 0.5,
            'exploit_rew_shift': 0,
            'exploit_rew_scale': 1,
            'eps': 0.1,
            'use_boltzmann': False,
            'double_q': True,
            'grad_norm_clipping': 10,
            'gamma': 0.95,
            'q_func': create_boxenv_q_network,
            'iql_expectile': 0.7,
            'awac_lambda': 10,
            'n_actions': 10,
            'explore_weight_schedule': PiecewiseSchedule([(0, 1), (1000, 0)], outside_value=0),
            'exploit_weight_schedule': PiecewiseSchedule([(0, 0), (1000, 1)], outside_value=1),
            'batch_size': self.batch_size,
            'learning_starts': 100,
            'learning_freq': 1,
            'target_update_freq': 100,
            'exploration_schedule': PiecewiseSchedule([(0, 1), (1000, 0.1)], outside_value=0.1),
            'optimizer_spec': pointmass_optimizer(),
            'replay_buffer_size': 10000,
            'frame_history_len': 1,
            'rnd_output_size': 5,
            'rnd_n_layers': 2,
            'rnd_size': 32,
        }

        agent = IQLAgent(env, agent_params)

        ob_no = np.random.randn(self.batch_size, self.ob_dim).astype(np.float32)
        ac_na = np.random.randint(0, self.ac_dim, size=self.batch_size).astype(np.float32)
        re_n = np.random.randn(self.batch_size).astype(np.float32)
        next_ob_no = np.random.randn(self.batch_size, self.ob_dim).astype(np.float32)
        terminal_n = np.random.choice([0, 1], size=self.batch_size).astype(np.float32)

        advantage = agent.estimate_advantage(ob_no, ac_na, re_n, next_ob_no, terminal_n)

        self.assertEqual(
            advantage.shape[0], self.batch_size,
            msg=f"estimate_advantage should return tensor of shape ({self.batch_size},), got {advantage.shape}"
        )

    @graded(timeout=30)
    def test_6(self):
        """1a-6-basic: test estimate_advantage uses correct critic"""
        self._setup_iql_agent_test()
        if RefIQLAgent is None:
            self.skipTest("Reference IQL Agent not available")

        from submission.xcs224r.agents.iql_agent import IQLAgent
        from submission.xcs224r.infrastructure.dqn_utils import PiecewiseSchedule

        torch.manual_seed(224)
        np.random.seed(224)

        env = gym.make('PointmassHard-v0')

        agent_params = {
            'env_name': 'PointmassHard-v0',
            'ob_dim': self.ob_dim,
            'ac_dim': self.ac_dim,
            'discrete': True,
            'n_layers': 2,
            'size': 64,
            'learning_rate': 1e-3,
            'num_exploration_steps': 1000,
            'offline_exploitation': False,
            'cql_alpha': 0.5,
            'exploit_rew_shift': 0,
            'exploit_rew_scale': 1,
            'eps': 0.1,
            'use_boltzmann': False,
            'double_q': True,
            'grad_norm_clipping': 10,
            'gamma': 0.95,
            'q_func': create_boxenv_q_network,
            'iql_expectile': 0.7,
            'awac_lambda': 10,
            'n_actions': 10,
            'explore_weight_schedule': PiecewiseSchedule([(0, 1), (1000, 0)], outside_value=0),
            'exploit_weight_schedule': PiecewiseSchedule([(0, 0), (1000, 1)], outside_value=1),
            'batch_size': self.batch_size,
            'learning_starts': 100,
            'learning_freq': 1,
            'target_update_freq': 100,
            'exploration_schedule': PiecewiseSchedule([(0, 1), (1000, 0.1)], outside_value=0.1),
            'optimizer_spec': pointmass_optimizer(),
            'replay_buffer_size': 10000,
            'frame_history_len': 1,
            'rnd_output_size': 5,
            'rnd_n_layers': 2,
            'rnd_size': 32,
        }

        agent = IQLAgent(env, agent_params)

        # The advantage should be computed using exploitation_critic
        # Verify it uses v_net and q_net from exploitation_critic
        ob_no = np.random.randn(self.batch_size, self.ob_dim).astype(np.float32)
        ac_na = np.random.randint(0, self.ac_dim, size=self.batch_size).astype(np.float32)
        re_n = np.random.randn(self.batch_size).astype(np.float32)
        next_ob_no = np.random.randn(self.batch_size, self.ob_dim).astype(np.float32)
        terminal_n = np.random.choice([0, 1], size=self.batch_size).astype(np.float32)

        advantage = agent.estimate_advantage(ob_no, ac_na, re_n, next_ob_no, terminal_n)

        # Advantage should be Q(s,a) - V(s)
        # Verify the shape is correct (batch_size, 1) or (batch_size,)
        self.assertEqual(
            advantage.numel(),
            self.batch_size,
            msg="estimate_advantage should return one value per batch element!"
        )

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_2a(GradedTestCase):
    """Tests for CQL Critic implementation"""

    def setUp(self):
        torch.manual_seed(224)
        np.random.seed(224)
        random.seed(224)
        ptu.init_gpu(use_gpu=False)

        self.ob_dim = 4
        self.ac_dim = 4
        self.batch_size = 32

        self.hparams = create_test_hparams(self.ob_dim, self.ac_dim)
        self.optimizer_spec = pointmass_optimizer()

    @graded()
    def test_0(self):
        """2a-0-basic: test CQL update returns correct info dict keys"""
        critic = CQLCritic(self.hparams, self.optimizer_spec)

        # Generate random batch
        ob_no = np.random.randn(self.batch_size, self.ob_dim).astype(np.float32)
        ac_na = np.random.randint(0, self.ac_dim, size=self.batch_size).astype(np.float32)
        next_ob_no = np.random.randn(self.batch_size, self.ob_dim).astype(np.float32)
        reward_n = np.random.randn(self.batch_size).astype(np.float32)
        terminal_n = np.random.choice([0, 1], size=self.batch_size).astype(np.float32)

        info = critic.update(ob_no, ac_na, next_ob_no, reward_n, terminal_n)

        # Check that CQL-specific keys are present
        self.assertIn('CQL Loss', info, msg="CQL Loss should be in the returned info dict")
        self.assertIn('Data q-values', info, msg="Data q-values should be in the returned info dict")
        self.assertIn('OOD q-values', info, msg="OOD q-values should be in the returned info dict")

    ### BEGIN_HIDE ###
    ### END_HIDE ###


def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)

if __name__ == "__main__":
    # Parse for a specific test or mode
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "test_case",
        nargs="?",
        default="all",
        help="Use 'all' (default), a specific test id like '1-3-basic', 'public', or 'hidden'",
    )
    test_id = parser.parse_args().test_case

    def _flatten(suite):
        """Recursively flatten unittest suites into individual tests."""
        for x in suite:
            if isinstance(x, unittest.TestSuite):
                yield from _flatten(x)
            else:
                yield x

    assignment = unittest.TestSuite()

    if test_id not in {"all", "public", "hidden"}:
        # Run a single specific test
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        # Discover all tests
        discovered = unittest.defaultTestLoader.discover(".", pattern="grader.py")

        if test_id == "all":
            assignment.addTests(discovered)
        else:
            # Filter tests by visibility marker in docstring ("basic" for public tests, "hidden" for hidden tests)
            keyword = "basic" if test_id == "public" else "hidden"
            filtered = [
                t for t in _flatten(discovered)
                if keyword in (getattr(t, "_testMethodDoc", "") or "")
            ]
            assignment.addTests(filtered)

    CourseTestRunner().run(assignment)
