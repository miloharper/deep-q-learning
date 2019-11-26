# Deep Q Learning
A demonstration of how Deep Q Learning can solve the NChain-v0 game:

https://gym.openai.com/envs/NChain-v0/

The NChain game has 5 possible states: 0, 1, 2, 3, 4. The agent can play 'Action 0' and take 1 step forward for which it receives a reward of 0, or it can play 'Action 1' and go back to position 0 and receive an instant reward of 2. However, if it makes it all the way to state 4 it can receive a reward of 10. To make it more challenging 20% of the time the agent slips, and the intended action is reversed.

A low intelligence agent which exhibits short-term thinking will learn to play Action 1 repeatedly to get the reward of 2. However, a truly intelligent agent will accept delayed gratification and will learn to keep moving forward to claim the reward of 10, even though there is no short-term benefit for doing so.

To solve the game, we create a neural network with 5 input neurons, 10 neurons in the hidden layer and 2 output neurons. The 5 input neurons take a one hot vector, representing the state. The 2 output neurons represent the expected reward for the 2 actions.
