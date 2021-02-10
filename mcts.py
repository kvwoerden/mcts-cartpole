import time
import random
import copy
import math
import itertools
import random
import numpy as np

import gym
from gym import wrappers, logger

# C_p = 0.7071067811865475 # 1/math.sqrt(2)
# C_p = 30
DEBUG = False
ENVIRONMENT = 'CartPole-v0'


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


def UctSearch(params, iterations=100, debug=False, node=None, all_nodes=None, C_p=None, lookahead_target=None):
    # if all_nodes == None:
    #     all_nodes = []
    # else:
    #     for node in all_nodes:
    #         node.Q -= node.depth*node.N
    #         node.depth -= 1

    if C_p == None:
        C_p = 200
    if lookahead_target == None:
        lookahead_target = 200
    if node == None:
        root_node = MCTSNode(params, False, 0)
    else:
        root_node = node
    if debug:
        print("### 000", root_node.id)
    start_time = time.time()

    counter = 0
    max_depth = 0
    # for ix in range(iterations):
    ix = 0
    while True:
        if debug:
            print("### 001")
        v = TreePolicy(root_node, all_nodes, C_p)
        # print("### 201 id: ", v.id)
        max_depth = max(v.depth - root_node.depth, max_depth)
        # if max_depth < 200:
        #     C_p -= 1
        # else:
        #     C_p += 1
        Delta = DefaultPolicy(v, root_node.depth)
        Backup(v, Delta, root_node)
        if debug:
            print("### 006", "id: ", root_node.id,
                  "N: ", root_node.N, "Q: ", root_node.Q)
            print("### 007", sorted(list(map(lambda x: (x.id, x.N, x.Q,
                                                        x.action), root_node.children)), key=lambda x: x[3]))
        if debug and counter % 3000 == 0:
            for child in root_node.children:
                pass
                # print("### 101", child.action)
                # child.state.render()
            inp = input()
            if inp == 'c':
                break
        counter += 1
        ix += 1
        if ix > iterations:
            break
    if max_depth < lookahead_target:
        print("### 301 ", max_depth)
        C_p = C_p - 1
        print("### 303 ", C_p)
    else:
        print("### 302 ", max_depth)
        C_p = C_p + 1
        print("### 304 ", C_p)
    # C_p = C_p + 0.1*(max_depth - lookahead_target)

    if debug:
        print("### 008", list(map(lambda v: (v.action, v.Q/v.N), root_node.children)))
    # return BestChild(root_node, 0).action
    print("### 101 Maximal depth considered: ", max_depth)
    # pred_life = max(list(map(lambda x: x.N, root_node.children)))
    child0 = root_node.children[0]
    child1 = root_node.children[1]
    print("### 103 child0 Q/N: ", child0.Q /
          child0.N, "Q: ", child0.Q, "N: ", child0.N)
    print("### 103 child1 Q/N: ", child1.Q /
          child1.N, "Q: ", child1.Q, "N: ", child1.N)
    # print("### 102 Predicted remaining life: ", pred_life)
    best_child = max(root_node.children.values(), key=lambda x: x.N)
    best_child_action = best_child.action
    print("### 102 predicted state: ", best_child.params)
    print("### 103 chosen action: ", best_child_action)
    # for ix in range(7):
    #     best_child = BestChild(best_child, 0)
    #     print("### 202 BestChild ix: ", ix, " action: ", best_child.action, " state: ", best_child.params)

    best_child_node = max(root_node.children.values(), key=lambda x: x.N)
    return (best_child_node.action, best_child_node, all_nodes, C_p)


def TreePolicy(node, all_nodes, C_p):
    depth = node.depth
    # if node.id == 813:
    # print("### 401 ", node.children, node.id)
    while not node.done:
        if len(node.children) < 2:
            return Expand(node, all_nodes)
        else:
            node = BestChild(node, C_p)
    return node


def Expand(node, all_nodes):
    # print("### 402 Expanding")
    # exp_env = copy.deepcopy(node.env)
    exp_env = gym.make(ENVIRONMENT)
    exp_env.reset()
    # breakpoint()
    exp_env.unwrapped.state = np.array(node.params)
    # exp_env.set_state(node.params)
    # exp_env.state = tuple(node.params)
    unchosen_actions = list(
        filter(lambda action: not action in node.children.keys(), [0, 1]))
    a = random.choice(unchosen_actions)
    params, reward, done, _ = exp_env.step(a)
    child_node = MCTSNode(params, done, node.depth+1)
    child_node.parent = node
    node.children[a] = child_node
    child_node.action = a
    # all_nodes.append(child_node)
    return child_node
    # av_act = node.state.available_actions()
    # chosen_actions = list(map(lambda child: child.action, node.children))
    # unchosen_actions = list(filter(lambda action: not action in chosen_actions, av_act))
    # a = random.choice(unchosen_actions)
    # child_state = copy.deepcopy(node.state)
    # # print("### 102", child_state.render())
    # child_state.step(a)
    # # print("### 103", child_state.render())
    # child_node = MCTSNode(child_state)
    # child_node.parent = node
    # node.add_child(child_node)
    # child_node.action = a
    # return child_node


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


def DefaultPolicy(node, root_depth):
    # exp_env = copy.deepcopy(node.env)
    exp_env = gym.make(ENVIRONMENT)
    exp_env.reset()
    exp_env.unwrapped.state = np.array(node.params)
    # # exp_env.set_state(node.params)
    # exp_env.state = tuple(node.params)
    done = node.done
    reward = node.depth
    if DEBUG:
        print("### 602 reward, done:", reward, done)
        # rollout_state.render()
    while not done:
        random_action = random.choice([0, 1])
        params, step_reward, done, _ = exp_env.step(random_action)
        reward += step_reward
    if DEBUG:
        print("### 501")
        rollout_state.render()
    if DEBUG:
        print("### 502 Reward:", reward)
    return reward


def Backup(node, Delta, root_node):
    while not node is root_node.parent:
        if DEBUG:
            node.state.render()
            print("### 601 Delta", Delta)
        # input()
        # print("301", Delta)
        # input()
        node.N += 1
        node.Q = node.Q + Delta
        node = node.parent
