[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# AlphaDev
AlphaDev is an AI model based on the AlphaZero/MuZero Reinforcement Learning architecture. It's designed to optimize assembly code using a set of assembly instructions and a cost function which takes into account both correctness and performance.

## Usage
`pip install alphadev`

## Architecture

AlphaDev consists of:

1. Representation Network: `f_rep` that outputs a latent representation `ht` of the state `St`.

2. Prediction Network: `f_pred` that predicts the expected return (the value) `vˆt` and a policy `πˆt` from a given latent state.

3. Dynamics Network: `f_dyn` that predicts the next latent state `htk+1` and reward `rˆtk+1` resulting from a transition.

## How AlphaDev Works

On reaching a new state, AlphaDev encodes the state into a latent representation using the representation network. The dynamics and prediction networks are used to simulate several trajectories that fill out a search tree by sampling state transitions.

The actions are selected using a strategy that balances exploration (trying new actions) and exploitation (progressing further down the subtree of the current best action).

Finally, the predicted policy is trained to match the visit counts of the MCTS policy in an attempt to distil the search procedure into a policy that will disregard nodes that are not promising.

## Potential Use Cases

AlphaDev, due to its general architecture, could potentially be adapted to solve a wide variety of optimization problems. Here are a few examples:

1. **Route Optimization**: For logistics companies, optimizing the routes of their fleet can result in significant cost savings. AlphaDev could be used to learn the optimal routes based on a variety of factors such as traffic, distance, and number of stops.

2. **Job Scheduling**: In computing, job scheduling is a key issue. AlphaDev could be used to learn the optimal schedule that maximizes the usage of computational resources and minimizes job completion time.

3. **Stock Portfolio Optimization**: AlphaDev could be used to learn the optimal mix of stocks to maximize return and minimize risk, given the current market conditions.

4. **Game Playing**: Similar to its ancestor AlphaZero, AlphaDev could potentially be used to master a wide variety of games, by learning the optimal strategies.

5. **Drug Discovery**: AlphaDev could be used to find the optimal chemical structure for a new drug that maximizes efficacy and minimizes side effects.


## Usage

* AssemblyGame This represents the Assembly Game RL environment. The state of the RL environment contains the current program and the state of memory and registers. Doing a step in this environment is equivalent to adding a new assembly instruction to the program (see the step method). The reward is a combination of correctness and latency reward after executing the assembly program over an input distribution. For simplicity of the overall algorithm we are not including the assembly runner, but assembly execution can be delegated to an external library (e.g. AsmJit).

* AlphaDevConfig contains the main hyperparameters used for the AlphaDev agent. This includes configuration of AlphaZero, MCTS, and underlying networks.

* play_game contains the logic to run an AlphaDev game. This include the MCTS procedure and the storage of the game.

* RepresentationNet and PredictionNet contain the implementation the networks used in the AlphaZero algorithm. It uses a MultiQuery Transformer to represent assembly instruction

## Future Work

Future adaptations of AlphaDev could implement different learning algorithms or optimization techniques for specific domains or problem areas.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

We appreciate the efforts of the researchers and developers who contributed to the development of the AlphaZero/MuZero architectures on which AlphaDev is based.

## Roadmap

* Add jax-based multi query attention: `MultiQueryAttentionBlock`

* add `ResBlockV2`

* add utils, terminal, is_correct, legal_actions
