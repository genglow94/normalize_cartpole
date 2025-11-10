import numpy as np
import random

class PrioritizedReplayBuffer:
    def __init__(self,capacity,alpha=0.6):
        self.capacity = capacity #最大の経験数
        self.alpha = alpha #優先度の強さ
        self.buffer = [] #経験を保存するリスト
        self.pos = 0 #どの経験を指すかの現在地
        self.priorities = np.zeros((capacity,),dtype=np.float32)#各経験に対しての優先度を保存する配列

    def push(self,state,action,reward,next_state,done):#経験バッファに追加
        max_prio = self.priorities.max() if self.buffer else 1.0 #新しい経験に対しての優先度

        if len(self.buffer) < self.capacity:
            self.buffer.append((state,action,reward,next_state,done))#上書き
        else:
            self.buffer[self.pos] = (state,action,reward,next_state,done)

        self.priorities[self.pos] = max_prio#追加した経験に優先度を設定
        self.pos = (self.pos +1)% self.capacity#上書きする位置を更新

    def sample(self,batch_size,beta=0.4):
        if len(self.buffer) == self.capacity:#有効な優先度を取り出す
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha#優先度を調整
        probs /= probs.sum()#正規化

        indices = np.random.choice(len(self.buffer),batch_size,p=probs)#確率probsに従い経験をバッチサイズ分サンプリング
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)#経験の総数
        weights = (total*probs[indices]**(-beta))#重要度サンプリングの重みを計算
        weights /= weights.max()
        weights = np.array(weights,dtype=np.float32)#正規化

        states,actions,rewards,next_states,dones = zip(*samples)#列ごと分解
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )
    
    def update_priorities(self,indices,td_errors):
        for idx,td_errors in zip(indices,td_errors):
            self.priorities[idx] = abs(td_errors) + 1e-5

    def __len__(self):
        return len(self.buffer)