# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import random
import math
import itertools
import random
import numpy as np

import gym


# ---------------------------------------------------------------------------- #
#                                   Constants                                  #
# ---------------------------------------------------------------------------- #
DEBUG = False


# ---------------------------------------------------------------------------- #
#                            Monte Carlo Tree Search                           #
# ---------------------------------------------------------------------------- #
class MCTSNode:
    id_iter = itertools.count()

    def __init__(self, params, done, depth):
        self.params = params
        self.children = {}
        self.parent = None
        self.Q = 0
        self.N = 0
        self.id = next(MCTSNode.id_iter)
        self.done = done
        self.depth = depth

    def add_child(self, action, child):
        self.children[action] = child


def UctSearch(params, n_actions, environment, iterations=100, node=None,  C_p=None, lookahead_target=None):
    if C_p == None:
        C_p = 200
    if lookahead_target == None:
        lookahead_target = 200
    if node == None:
        root_node = MCTSNode(params, False, 0)
    else:
        root_node = node

    counter = 0
    max_depth = 0
    ix = 0
    while True:
        v = TreePolicy(root_node, C_p, n_actions, environment)
        max_depth = max(v.depth - root_node.depth, max_depth)
        Delta = DefaultPolicy(v, environment)
        Backup(v, Delta, root_node)
        counter += 1
        ix += 1
        if ix > iterations:
            break
    if max_depth < lookahead_target:
        C_p = C_p - 1
    else:
        C_p = C_p + 1
    print(
        f"### max_depth: {max_depth:03}, lookahead_target: {lookahead_target:03} ")
    print(f"### C_p: {C_p} ")
    print("### Maximal depth considered: ", max_depth)
    for action, child in sorted(root_node.children.items()):
        print(
            f"### action: {action}, Q: {int(child.Q):08}, N: {child.N:08}, Q/N: {child.Q/child.N:07.2f}")

    best_child = max(root_node.children.values(), key=lambda x: x.N)
    best_child_action = best_child.action
    print(f"### predicted state: {best_child.params}")
    print(f"### chosen action: {best_child_action}")

    best_child_node = max(root_node.children.values(), key=lambda x: x.N)
    return (best_child_node.action, best_child_node, C_p)


def TreePolicy(node, C_p, n_actions, environment):
    while not node.done:
        if len(node.children) < n_actions:
            return Expand(node, n_actions, environment)
        else:
            node = BestChild(node, C_p)
    return node


def Expand(node, n_actions, environment):
    exp_env = gym.make(environment)
    exp_env.reset()
    exp_env.unwrapped.state = np.array(node.params)

    unchosen_actions = list(
        filter(lambda action: not action in node.children.keys(), range(n_actions)))
    a = random.choice(unchosen_actions)
    params, _, done, _ = exp_env.step(a)
    child_node = MCTSNode(params, done, node.depth+1)
    child_node.parent = node
    node.children[a] = child_node
    child_node.action = a
    return child_node


def BestChild(node, c, random=False):
    if random:
        child_values = {child: child.Q/child.N + c *
                        math.sqrt(2*math.log(node.N)/child.N) for child in node.children}
        mv = max(child_values.values())
        am = random.choice([k for (k, v) in child_values.items() if v == mv])
    else:
        am = max(node.children.values(), key=lambda v_prime: v_prime.Q /
                 v_prime.N + c*math.sqrt(2*math.log(node.N)/v_prime.N))
    return am


def DefaultPolicy(node, environment):
    exp_env = gym.make(environment)
    exp_env.reset()
    exp_env.unwrapped.state = np.array(node.params)

    done = node.done
    reward = node.depth

    while not done:
        random_action = random.choice([0, 1])
        _, step_reward, done, _ = exp_env.step(random_action)
        reward += step_reward

    return reward


def Backup(node, Delta, root_node):
    while not node is root_node.parent:
        node.N += 1
        node.Q = node.Q + Delta
        node = node.parent


class MCTSAgent(object):
    def __init__(self, time_budget, environment):
        self.agenttype = 'mcts'
        self.time_budget = time_budget
        self.environment = environment

    def act(self, params, n_actions, node, C_p, lookahead_target):
        return UctSearch(params, n_actions, self.environment, self.time_budget, node=node, C_p=C_p, lookahead_target=lookahead_target)
