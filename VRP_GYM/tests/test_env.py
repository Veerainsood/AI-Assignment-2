import pytest
from agents import IRPAgent, TSPAgent, VRPAgent, RandomAgent
from gym_vrp.envs import IRPEnv, TSPEnv, VRPEnv
import numpy as np
import networkx as nx
import math


@pytest.fixture(scope="function", autouse=True)
def setup():
    seed = 69
    np.random.seed(seed)

    pytest.node_num = 3
    pytest.graph_num = 2
    pytest.env = VRPEnv(pytest.node_num, pytest.graph_num, 2)

    y_coord = math.sqrt(3) / 2
    graph_one_coords = {
        0: np.array([0, 0]),
        1: np.array([1, 0]),
        2: np.array([0.5, y_coord]),
    }

    graph_two_coords = {
        0: np.array([0, 0]),
        1: np.array([4, 0]),
        2: np.array([2, 4 * y_coord]),
    }

    nx.set_node_attributes(
        pytest.env.sampler.graphs[0], graph_one_coords, "coordinates"
    )
    nx.set_node_attributes(
        pytest.env.sampler.graphs[1], graph_two_coords, "coordinates"
    )


def test_init():
    assert len(pytest.env.sampler.graphs) == 2
    assert len(pytest.env.sampler.graphs[0].nodes) == 3


def test_step():
    actions = np.array([2, 2])[:, None]
    _, batched_reward, _, _ = pytest.env.step(actions)

    assert np.allclose(batched_reward, np.array([-1, 0]))


def test_state():
    state = pytest.env.get_state()

    assert state.shape == (pytest.graph_num, pytest.node_num, 4)
    assert np.sum(state[:, :, 2]) == 2

    actions = np.array([2, 2])[:, None]
    state, _, _, _ = pytest.env.step(actions)

    assert state[0, 2, 3] == 1 and state[1, 2, 3] == 1
