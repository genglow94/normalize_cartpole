import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from MMP import general_01

# ======== ハイパーパラメータ ========
num_episodes = 500
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
lr = 1e-3
batch_size = 64
memory_size = 10000

# ======== 経験再生バッファ ========
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

# ======== 学習関数 ========
def train_variant(mode):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = general_01(state_dim, action_dim, mode=mode)
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(memory_size)

    rewards = []
    global epsilon

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(torch.FloatTensor(state).unsqueeze(0))
                    action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                states, actions, rewards_b, next_states, dones = buffer.sample(batch_size)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards_b = torch.FloatTensor(rewards_b)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = q_net(next_states).max(1)[0]
                    target_q = rewards_b + gamma * max_next_q * (1 - dones)

                loss = F.mse_loss(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)

        # === 正規化テスト ===
        if episode % 100 == 0 and len(buffer) >= batch_size:
            states_batch, _, _, _, _ = buffer.sample(batch_size)
            D = torch.FloatTensor(states_batch)
            f_func = lambda x: q_net.fc_in(x)
            g_func = lambda x: q_net.fc_in(x)
            q_net.mmp.restricted_normalize(D, f_func, g_func)

        if episode % 100 == 0:
            print(f"Episode {episode}: Avg100={avg100:.2f}, Eps={epsilon:.3f}")

    env.close()
    return rewards

# ======== メイン処理 ========
modes = ["none", "min", "max", "both"]
results = {}

for mode in modes:
    print(f"=== Training mode: {mode} ===")
    results[mode] = train_variant(mode)

# ======== グラフ描画 ========
plt.figure()
for mode, r in results.items():
    plt.plot(r, label=mode)
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("CartPole - MMP Normalization Comparison")
plt.show()
