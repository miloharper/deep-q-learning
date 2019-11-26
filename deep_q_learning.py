import gym
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras import optimizers


class Agent():
    def __init__(self):
        self.learning_rate = 0.05
        self.neural_network = NeuralNetwork(self.learning_rate)
        self.discount_factor = 0.95
        self.epsilon = 0.5
        self.decay_factor = 0.999
        self.average_reward_in_each_game = []

    def play(self, env, number_of_games=100):
        for game in range(number_of_games):

            # Print the current game number to the Terminal
            print("Game {} of {}".format(game + 1, number_of_games))

            # Reset the environment before play begins
            state = env.reset()
            total_reward = 0

            # Reduce the exploration rate each game
            self.epsilon *= self.decay_factor

            # Play the game until it ends after 1000 steps
            end_game = False
            while not end_game:

                # Choose randomly with probability epsilon, otherwise choose the action with the highest expected reward
                if self.__with_probability(self.epsilon):
                    action = self.__get_action_by_choosing_randomly(env)
                else:
                    action = self.__get_action_with_highest_expected_reward(state)

                # Perform the action
                new_state, reward, end_game, _ = env.step(action)
                total_reward += reward

                # Train the neural network
                target_output = self.neural_network.predict_expected_rewards_for_each_action(state)
                target_output[action] = reward + self.discount_factor * self.__get_expected_reward_in_next_state(new_state)    
                self.neural_network.train(state, target_output)
                
                # Update the state
                state = new_state

            # Store the average reward after the game has finished
            self.average_reward_in_each_game.append(total_reward / 1000)

            # Print the latest neural network results to the Terminal
            print(tabulate(self.neural_network.results(), showindex="always", headers=["State", "Action 0 (Forward 1 step)", "Action 1 (Back to 0)"]))

    def __with_probability(self, probability):
        return np.random.random() < probability

    def __get_action_by_choosing_randomly(self, env):
        return env.action_space.sample()

    def __get_action_with_highest_expected_reward(self, state):
        return np.argmax(self.neural_network.predict_expected_rewards_for_each_action(state))

    def __get_expected_reward_in_next_state(self, next_state):
        return np.max(self.neural_network.predict_expected_rewards_for_each_action(next_state))

# Create a neural network to predict the Q values. There are 5 inputs corresponding to the 5 states. And 2 outputs corresponding to the 2 actions.
# There are 10 neurons in the hidden layer.
class NeuralNetwork(Sequential):
    def __init__(self, learning_rate=0.05):
        super().__init__()
        self.add(InputLayer(batch_input_shape=(1, 5)))
        self.add(Dense(10, activation='sigmoid'))
        self.add(Dense(2, activation='linear'))
        self.compile(loss='mse', optimizer=optimizers.Adam(lr=learning_rate))

    def train(self, state, target_output):
        input_signal = self.__convert_state_to_neural_network_input(state)
        target_output = target_output.reshape(-1, 2)
        self.fit(input_signal, target_output, epochs=1, verbose=0)

    def predict_expected_rewards_for_each_action(self, state):
        input_signal = self.__convert_state_to_neural_network_input(state)
        return self.predict(input_signal)[0]

    def results(self):
        results = []
        for state in range(0, 5):
            results.append(self.predict_expected_rewards_for_each_action(state))
        return results

    # Best explained with an example: will convert 3 to [[0, 0, 0, 1, 0]
    def __convert_state_to_neural_network_input(self, state):
        input_signal = np.zeros((1, 5))
        input_signal[0, state] = 1
        return input_signal

def graph_average_reward(average_reward):
    plt.plot(average_reward)
    plt.title("Performance over time")
    plt.ylabel("Average reward")
    plt.xlabel("Games")
    plt.show()

# Create the Nchain-v0 environment
env = gym.make('NChain-v0')

# Create an intelligent agent
agent = Agent()

# Play the game
agent.play(env)

# Graph the average reward
graph_average_reward(agent.average_reward_in_each_game)
