import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from MMP import MaxMinQNet
from prioritized_replay_buffer import PrioritizedReplayBuffer
import random
import matplotlib.pyplot as plt
import time

# 時間計測開始
start_time = time.time()

# ======== 環境設定 ========
env = gym.make("CartPole-v1", render_mode="None")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# ======== Qネットワーク ========
q_net = MaxMinQNet(obs_dim, n_actions)
target_net = MaxMinQNet(obs_dim, n_actions)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=5e-4)
buffer = PrioritizedReplayBuffer(20000)

# ======== ハイパーパラメータ ========
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
target_update_freq = 5
num_episodes = 2000
alpha = 0.6
beta = 0.4
beta_increment_per_episode = 0.001

# ======== ログ用 ========
episode_rewards = []
avg100_list = []

# ======== 学習ループ ========
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # ε-greedy 行動選択
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_net(state_tensor)
            action = q_values.argmax().item()

        # 1ステップ実行
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # CartPole報酬シェイピング（位置・角度ペナルティ）
        x, x_dot, theta, theta_dot = next_state
        shaped_reward = reward - 0.1 * (abs(x) + abs(theta))

        # バッファに保存
        buffer.push(state, action, shaped_reward, next_state, done)
        state = next_state
        total_reward += reward

        # Qネット更新
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones, indices, weights = buffer.sample(batch_size, beta)

            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions).unsqueeze(1)
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
            next_states_tensor = torch.FloatTensor(next_states)
            dones_tensor = torch.FloatTensor(dones).unsqueeze(1)
            weights_tensor = torch.FloatTensor(weights).unsqueeze(1)

            q_values = q_net(states_tensor).gather(1, actions_tensor)

            with torch.no_grad():
                next_actions = q_net(next_states_tensor).argmax(1, keepdim=True)
                next_q_values = target_net(next_states_tensor).gather(1, next_actions)
                target = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

            td_errors = (q_values - target).detach().cpu().numpy().squeeze()
            buffer.update_priorities(indices, td_errors)

            loss = F.smooth_l1_loss(q_values, target, reduction='none')
            loss = (weights_tensor * loss).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ======== Min-Plus正規化（50エピソードごと） ========
    if episode % 50 == 0 and len(buffer) >= batch_size:
        states_batch, _, _, _, _, _, _ = buffer.sample(batch_size, beta)
        D = torch.FloatTensor(states_batch)
        f_func = lambda x: q_net.max1(q_net.fc_in(x))
        g_func = lambda x: q_net.min1(f_func(x))
        q_net.min1.restricted_normalize(D, f_func, g_func)

    # ======== エピソード終了処理 ========
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    if episode % target_update_freq == 0:
        target_net.load_state_dict(q_net.state_dict())

    episode_rewards.append(total_reward)
    avg100 = np.mean(episode_rewards[-100:])  # ← 直近100エピソード平均
    avg100_list.append(avg100)

    print(f"Episode {episode+1}: Total = {total_reward:.1f}, Avg100 = {avg100:.1f}, Epsilon = {epsilon:.3f}")

env.close()

# ======== 学習時間 ========
end_time = time.time()
print(f"\n総学習時間: {end_time - start_time:.2f} 秒")

# ======== 学習曲線 ========
plt.figure(figsize=(10,5))
plt.plot(episode_rewards, label="Episode reward", alpha=0.6)
plt.plot(avg100_list, label="Avg100 reward", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("MMP Learning Curve on CartPole-v1")
plt.legend()
plt.grid()
plt.show()
