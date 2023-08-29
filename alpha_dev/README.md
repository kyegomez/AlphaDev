
**Reinforcement Learning Algorithms**

The key RL algorithms used in this code are:

- Monte Carlo Tree Search (MCTS): This is used to select actions during self-play. It involves recursively building out a tree of future states and actions to find the optimal path that maximizes long-term rewards. At each step, the tree is traversed according to the UCB formula which balances exploration and exploitation. After a specified number of simulations, the most visited child node is selected as the next action.

- n-step bootstrapping: This is used to create target values for training the neural network. Rather than updating the network towards the 1-step bootstrapped value, n-step returns are used which leads to more stable learning. The n-step return sums rewards up to n steps into the future and then bootsraps off the value estimate at that point.

- Prioritized experience replay: The replay buffer stores full game trajectories. During training, games are sampled proportional to some priority measure. This focuses learning on important transitions rather than uniformly sampling experience.

**System Overview**

The key components are:

- `Network` module - Contains the representation and prediction networks implemented in Haiku. This would be re-implemented in PyTorch using Sequential modules and custom Module classes.

- `Node` class - Represents a node in the MCTS search tree. Stores visit counts, prior probabilities, values, etc. This is algorithm logic independent of framework.

- `AssemblyGame` class - The environment simulator. This encapsulates the task-specific logic and dynamics. It would remain unchanged. 

- `ReplayBuffer` class - Stores game trajectories and samples minibatches for training. A PyTorch equivalent would sample tensors rather than Python dicts.

- `SharedStorage` class - Shares latest network checkpoints between self-play and training. Could use TensorBoard or a PyTorch `Event` hook to transfer new weights.

- `run_selfplay` - Runs MCTS using latest network to generate games. Wrap in PyTorch train loop with opted-out loss.

- `train_network` - Train loop that samples games from buffer and updates network. Implement in PyTorch using standard training loop over batches.

The main change is re-implementing the `Network` in PyTorch using Sequential and custom Network modules and training that from scratch. The algorithmic logic would remain largely the same implemented in Python. Hoist performance critical parts like MCTS into TorchScript to optimize. Use PyTorch tools like distributed training/self-play, TensorBoard integration, and ONNX export.