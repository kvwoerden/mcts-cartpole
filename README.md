# Cartpole with Monte Carlo Tree Search agent

This implements an agent for the Cartpole environment using Monte Carlo Tree Search. The agent is implemented in `mcts.py`.

# How to run

To run, first install the dependencies in `requirements.txt`, for example using a pip in a virtual environment:

```bash
	$ python3 -m venv env
	$ source env/bin/activate
	$ pip install -r requirements.txt
```

Next, start run the agent as follows:

```bash
	$ python cartpole.py
```

# Example

![Example run](img/example.gif)

# Options

## Iteration budget

To specify the number of iterations to use at each search step, use the `--iteration_budget` flag:

```bash
$ python cartpole.py --iteration_budget 200
```

Increasing this makes the agent perform better in general, and also slower.

## Look ahead target

To specify the target number of steps the agent should look ahead, use the `--lookahead_target` flag:

```bash
$ python cartpole.py --lookahead_target 200
```

## Start value of C_p

To specify the starting value of `C_p`, the number the agent modifies to achieve the lookahead target, use the `--start_cp` flag:

```bash
$ python cartpole.py --start_cp 20
```

## More options

To see a description of all options, use the `--help` flag:

```bash
$ python cartpole.py --help
usage: cartpole.py [-h] [--env_id [ENV_ID]] [--episodes [EPISODES]]
                   [--iteration_budget [ITERATION_BUDGET]]
                   [--lookahead_target [LOOKAHEAD_TARGET]]
                   [--max_episode_steps [MAX_EPISODE_STEPS]]
                   [--video_basepath [VIDEO_BASEPATH]] [--start_cp [START_CP]]
                   [--seed [SEED]]

Run a Monte Carlo Tree Search agent on the Cartpole environment

optional arguments:
  -h, --help            show this help message and exit
  --env_id [ENV_ID]     The environment to run (only CartPole-v0 is supperted)
                        (default: CartPole-v0)
  --episodes [EPISODES]
                        The number of episodes to run. (default: 1)
  --iteration_budget [ITERATION_BUDGET]
                        The number of iterations for each search step.
                        Increasing this should lead to better performance.
                        (default: 80)
  --lookahead_target [LOOKAHEAD_TARGET]
                        The target number of steps the agent aims to look
                        forward. (default: 100)
  --max_episode_steps [MAX_EPISODE_STEPS]
                        The maximum number of steps to play. (default: 1500)
  --video_basepath [VIDEO_BASEPATH]
                        The basepath where the videos will be stored.
                        (default: ./video)
  --start_cp [START_CP]
                        The start value of C_p, the value that the agent
                        changes to try to achieve the lookahead target.
                        Decreasing this makes the search tree deeper,
                        increasing this makes the search tree wider. (default:
                        20)
  --seed [SEED]         The random seed. (default: 28)
```
