import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from prioritized_replay_buffer import PrioritizedReplayBuffer
import random
import matplotlib.pyplot as plt
import time
from MMP import Type2_01   # ← MMPネット
from MMP import Type2_02  # ← MMPネット

# ====== 共通ハイパーパラメータ ======
num_seeds = 5
num_episodes = 2000
batch_size = 64
gamma = 0.99
epsilon_decay = 0.998
min_epsilon = 0.01
target_update_freq = 5
threshold = 475
stable_length = 20
lr = 5e-4
beta_start = 0.4
buffer_size = 20000

# ====== 比較対象モデルの設定 ======
models = {
    "Type2_01": Type2_01,
    "Type2_02": Type2_02
}

results = {}  # 結果格納 {name: {"rewards":[], "converge":[]}}

for name, NetClass in models.items():
    print(f"\n========== {name} モデル開始 ==========")
    final_avg_rewards = []
    converge_episodes = []
    avg100_list_all = []

    for seed in range(num_seeds):
        print(f"\n=== {name} Run {seed+1} ===")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = gym.make("CartPole-v1", render_mode="None")
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        q_net = NetClass(obs_dim, n_actions)
        target_net = NetClass(obs_dim, n_actions)
        target_net.load_state_dict(q_net.state_dict())

        optimizer = optim.Adam(q_net.parameters(), lr=lr)
        buffer = PrioritizedReplayBuffer(buffer_size)

        epsilon = 1.0
        episode_rewards = []
        avg100_list = []
        stable_count = 0
        converge_episode = None

        start_time = time.time()

        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            total_reward = 0
            beta = beta_start + (1.0 - beta_start) * episode / num_episodes

            while not done:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = q_net(state_tensor)
                    action = q_values.argmax().item()

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

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

                    loss = (weights_tensor * (q_values - target).pow(2)).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # ===== 正規化（MMPのみ） =====
            # if name == "MMP" and episode % 100 == 0 and len(buffer) >= batch_size:
            #     states_batch, _, _, _, _, _, _ = buffer.sample(batch_size, beta)
            #     D = torch.FloatTensor(states_batch)
            #     f_func = lambda x: q_net.max1(q_net.fc_in(x))
            #     g_func = lambda x: q_net.min1(f_func(x))
            #     q_net.min1.restricted_normalize(D, f_func, g_func)

            # ===== エピソード後処理 =====
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            if episode % target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())

            episode_rewards.append(total_reward)
            avg100 = np.mean(episode_rewards[-100:])
            avg100_list.append(avg100)

            if episode % 50 == 0:
                print(f"Episode {episode}: Avg100={avg100:.2f}, Eps={epsilon:.3f}")

        # ===== 収束判定 =====
        for i, avg in enumerate(avg100_list):
            if avg >= threshold:
                stable_count += 1
                if stable_count >= stable_length:
                    converge_episode = i - stable_length + 1
                    break
            else:
                stable_count = 0

        final_avg_rewards.append(avg100_list[-1])
        converge_episodes.append(converge_episode)
        avg100_list_all.append(avg100_list)
        env.close()

        end_time = time.time()
        print(f"{name} Run {seed+1} 完了: 最終報酬={avg100_list[-1]:.1f}, 収束={converge_episode}, 時間={end_time-start_time:.1f}s")

    # ===== 各モデルの結果保存 =====
    results[name] = {
        "final": final_avg_rewards,
        "converge": converge_episodes,
        "curves": avg100_list_all
    }

# ===== 結果出力 =====
print("\n========== 結果まとめ ==========")
for name, res in results.items():
    print(f"\n{name} モデル:")

    # ---- 最終平均報酬 ----
    final_rewards = np.array(res['final'])
    mean_final = np.mean(final_rewards)
    std_final = np.std(final_rewards)
    print(f"最終平均報酬: 平均={mean_final:.2f}, 分散={np.var(final_rewards):.2f}, σ={std_final:.2f}")

    # ---- 収束エピソード ----
    valid_converge = [c for c in res["converge"] if c is not None]
    if valid_converge:
        mean_converge = np.mean(valid_converge)
        std_converge = np.std(valid_converge)
        print(f"収束エピソード: 平均={mean_converge:.1f}, 分散={np.var(valid_converge):.1f}, σ={std_converge:.1f}")
    else:
        print("収束なし")

# ===== 学習曲線比較 =====
plt.figure(figsize=(10,6))
for name, res in results.items():
    avg_curve = np.mean(np.array([np.pad(c, (0, num_episodes - len(c)), 'edge') for c in res["curves"]]), axis=0)
    plt.plot(avg_curve, linewidth=2, label=f"{name}")
plt.title("Type2 MMP-NN")######
plt.xlabel("Episode")
plt.ylabel("Avg100 reward")
plt.grid()
plt.legend()
plt.show()
