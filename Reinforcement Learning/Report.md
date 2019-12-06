# Deep Q-Learning (DQN)
**DQN** is implemented to train the agent (banana brain) to collect the banana.

# Learning Algorithm
The DQN algorithm is composed of `dqn_agent.py` and `model.py`. This python file includes DQNAgent and ReplayBuffer. A Q value-based Netwrok is used with replay memory. The optimal algorithm is Adam.

This Q Netwrok is a three layer nerual network. The hidden layers are ``State -> fc1 (64) -> Relu -> fc2 (64) -> Relu -> fc3 (64, output) ``. 

Some other hypterparameters are minibatch size is 64, discount factor is 0.99 and learning rate is 5e-4. 


# Rewards Result
This is plot of rewards when training.
At Episode 524, agent performance met the criteria and stopped training.
(mean scores of last 100 episodes is above +13)

![plot of rewards](/DQN_result.png)

# Ideas for Future Work
- [Prioritized Experienced Replay](https://arxiv.org/abs/1511.05952)
- [Double DQN](https://arxiv.org/abs/1509.06461)
- [Duel DQN](https://arxiv.org/abs/1511.06581)
- Learning from Pixels

# Trained model
[Trained model (DQN)](./checkpoint.pth)
