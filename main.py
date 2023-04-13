import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import gymnasium as gym

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
memory_size = 10000
n_episodes = 500

class DQNAgent:
    def __init__(self, input_shape, n_actions):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.memory = []
        
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential([
            Dense(128, activation='relu', input_shape=self.input_shape),
            Dense(64, activation='relu'),
            Dense(self.n_actions, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > memory_size:
            self.memory.pop(0)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.n_actions)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states, q_values = [], []

        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            q_update = reward if done else reward + gamma * np.amax(self.target_model.predict(next_state)[0])
            q_values_current = self.model.predict(state)
            q_values_current[0][action] = q_update

            states.append(state[0])
            q_values.append(q_values_current[0])

        self.model.train_on_batch(np.array(states), np.array(q_values))

        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    input_shape = (env.observation_space.shape[0], )
    n_actions = env.action_space.n
    agent = DQNAgent(input_shape, n_actions)

    for episode in range(n_episodes):
        state = env.reset()[0].reshape(1, -1)
        done = False
        step = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = next_state.reshape(1, -1)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            step += 1

            if done:
                print(f"Episode: {episode+1}/{n_episodes}, Score: {step}, Epsilon: {agent.epsilon:.4f}")
                break

        agent.learn(batch_size)
        if (episode + 1) % 10 == 0:
            agent.update_target_model()
