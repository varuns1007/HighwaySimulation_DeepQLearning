## 📄 Tabular Q-Learning & Deep Q-Learning

This project explores highway driving using **Tabular Q-Learning** and **Deep Q-Learning (DQN)** in both discrete and continuous state environments. The goal is to learn safe and efficient driving policies through reinforcement learning techniques.

---

### 🔹 Part I: Tabular Q-Learning

#### 🚗 Learning the Policy
- **Discounted Return**: Rapid improvement in first 20k iterations, then plateau.
- **Max Distance Traveled**: Steady rise with some variance, improving with training.
- **Insight**: Stable convergence; agent learns to drive longer with fewer collisions.

#### 🛣️ Lane Preference Visualization
- Middle lanes are preferred due to lower collision risk.
- Lane values adjust based on obstacle distribution.
- Smooth transitions indicate cautious lane switching.

#### 🕹️ Speed Preference Visualization
- Agent learns to maintain moderate speed in most cases.
- One outlier showed high-reward speed state — potentially underexplored.
- Speed control is dynamic and lane-aware.

---

### 🔁 Hyperparameter Experiments

#### 🔸 Discount Factor (γ)
| γ     | Avg. Distance | Return at Start | Insight                                   |
|-------|---------------|------------------|--------------------------------------------|
| 0.8   | ~470          | ~0.55            | Short-term rewards prioritized              |
| 0.9   | ~462          | ~1.11            | Balanced short- and long-term performance   |
| 0.99  | ~551          | ~7.99            | Long-term focused, best distance covered    |

#### 🔸 Learning Rate (α)
| α     | Avg. Distance | Return at Start | Insight                              |
|-------|---------------|------------------|---------------------------------------|
| 0.1   | ~462          | ~1.11            | Stable convergence                    |
| 0.3   | ~171          | ~1.09            | Faster learning, slightly unstable    |
| 0.5   | ~92           | ~0.99            | High variance, overshooting Q-values |

#### 🔸 Exploration Strategy (ε)
- **Constant ε**:
  - Best at ε = 0.75 (balance between exploration and exploitation).
- **Variable ε (Decay)**:
  - **Exponential decay** yields better returns and distance than linear decay.
  - Encourages exploration early and stability later.

---

### 🧪 Reward Modifications

#### ✅ Overtake-Based Rewards:
- Agent becomes aggressive, leading to **lower returns** and **high variance**.
- Encourages risky behavior — not ideal for safety.

#### ✅ 3-Level Quantization:
- Safer and more consistent policies.
- Promotes **conservative driving** and smooth lane/speed management.

---

### ✅ Best Hyperparameters (Tabular Q-Learning)

| Parameter        | Value                    |
|------------------|--------------------------|
| Learning Rate α  | 0.1                      |
| Discount Factor γ| 0.9                      |
| Exploration ε    | 0.75 (Exponential Decay) |

---

## 🤖 Part II: Deep Q-Learning (DQN)

### 🧱 Implementation Variants

#### 🧊 Hard Model Update
- Update target network every 100 episodes.
- Challenges with over-exploration and **catastrophic forgetting**.

#### 🌊 Soft Model Update
- Uses a small `τ` to update target network continuously.
- Better convergence than hard update in discrete environments.

#### 🧠 Prioritized Experience Replay (PER)
- Focuses on learning from high-priority experiences.
- Reduces variance and improves return significantly.

#### 📉 Epsilon Decay
- Applying `ε-decay` results in **stable convergence**.
- Prevents degradation after initial learning.

---

### 🧪 DQN on Continuous State Space

- **Observation**: Less improvement compared to discrete due to difficulty in generalization.
- **PER** didn’t help as much — further tuning needed.

---

### 🏁 Final Observations

| Variant                       | Avg. Max Distance | Start State Return |
|-------------------------------|-------------------|--------------------|
| Tabular Q-Learning (γ=0.99)   | ~551              | ~7.99              |
| DQN with Soft Update + PER    | ~106              | ~2.23              |
| DQN with Epsilon Decay        | ~139              | ~2.94              |
| DQN on Continuous Space       | ~82               | ~2.00              |

---

> 📌 This project demonstrates how tuning hyperparameters and updating strategies significantly influence learning in reinforcement learning agents. Both Q-Learning and DQN showed strengths in different areas. Combining best practices from both yielded improved and stable driving behaviors.
