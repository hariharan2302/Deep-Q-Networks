# 🎮 Deep Reinforcement Learning with Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-red.svg)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](#)

## 📋 Overview

This project explores **Deep Reinforcement Learning** through the implementation of **Deep Q-Networks (DQN)** and **Prioritized Experience Replay (PER)**. The assignment demonstrates value function approximation algorithms across multiple environments, showcasing the power of neural networks in solving complex sequential decision-making problems.

### 🎯 Key Objectives
- Implement Deep Q-learning based on DeepMind's groundbreaking paper
- Explore various neural network architectures for different observation types
- Compare standard DQN with Prioritized Experience Replay improvements
- Evaluate performance across multiple OpenAI Gym environments

---

## 🏗️ Project Structure

```
📦 Deep-RL-Project
├── 📓 roopeshv_hvenkatr_assignment2_part1.ipynb    # Neural Network Architectures
├── 📓 roopeshv_hvenkatr_assignment2_part2.ipynb    # DQN Implementation & Training
├── 🎯 roopeshv_hvenkatr_assignment2_part2_dqn_*.pickle    # Trained DQN Models
├── 🎯 roopeshv_hvenkatr_assignment2_part3_per_*.pickle    # Trained PER Models
├── 📄 README.md                                    # Project Documentation
└── 📄 LICENSE                                      # License Information
```

---

## 🧠 Part 1: Neural Network Architectures

### 🌍 Wumpus World Environment

The first part implements various neural network architectures for the **Wumpus World** environment - a classic AI problem from Russell & Norvig's textbook.

#### Environment Details:
- **Grid Size**: 6×6 (36 total blocks)
- **Objective**: Collect gold while avoiding the Wumpus and pits
- **Sensors**: Breeze (near pits) and Stench (near Wumpus)

#### 🔧 Implemented Network Architectures:

| Observation Type | Input Dimension | Architecture | Output |
|------------------|----------------|--------------|---------|
| **Integer** | 1 | Dense(1→64→4) | Q-values |
| **Vector (2D)** | 2 | Dense(2→64→4) | Q-values |
| **Vector (One-hot)** | 36 | Dense(36→64→4) | Q-values |
| **Image** | 84×84 | Conv2D(128)→Dense(64→4) | Q-values |
| **Float** | 1 | Dense(1→64→4) | Q-values |
| **Multi-Discrete** | 36 | Dense(36→64→4)→Sigmoid | Action probabilities |
| **Continuous** | 36 | Dense(36→64→1)→Tanh | Continuous action |

---

## 🚀 Part 2: Deep Q-Network Implementation

### 🎮 Environments Tested

#### 1. 🏁 GridWorld (Custom Mario Environment)
- **Theme**: Super Mario-inspired grid navigation
- **Goal**: Reach Princess Peach while avoiding enemies
- **Challenges**: Negative rewards for hitting obstacles
- **State Space**: 6×6 grid positions
- **Action Space**: 4 discrete actions (Up, Down, Left, Right)

#### 2. 🏔️ MountainCar-v0
- **Objective**: Drive an underpowered car up a steep hill
- **Challenge**: Car must build momentum by oscillating
- **State Space**: Position and velocity (continuous)
- **Action Space**: 3 discrete actions (Left, Nothing, Right)

#### 3. 🎯 CartPole-v1
- **Objective**: Balance a pole on a moving cart
- **Challenge**: Prevent pole from falling over
- **State Space**: Cart position, velocity, pole angle, pole velocity
- **Action Space**: 2 discrete actions (Left, Right)

### 🧮 DQN Algorithm Features

#### Core Components:
- **Experience Replay Buffer**: Stores transitions for stable learning
- **Target Network**: Separate network for stable Q-value targets
- **ε-Greedy Exploration**: Balances exploration vs exploitation
- **Soft Target Updates**: Gradual target network updates (τ = 0.005)

#### Network Architecture:
```python
class Net(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

#### Hyperparameters:
- **Learning Rate**: 1e-3
- **Batch Size**: 64
- **Gamma (Discount Factor)**: 0.99
- **Epsilon Decay**: 0.009
- **Buffer Size**: 10,000
- **Target Update Frequency**: Soft updates every step

---

## ⭐ Part 3: Prioritized Experience Replay (PER)

### 🎯 Enhancement Overview

PER improves upon standard DQN by:
- **Prioritizing Important Transitions**: Samples experiences based on TD-error magnitude
- **Importance Sampling**: Corrects bias introduced by non-uniform sampling
- **Faster Learning**: Focuses on transitions where the agent can learn the most

### 📊 Performance Comparison

| Environment | Standard DQN | DQN + PER | Improvement |
|-------------|--------------|-----------|-------------|
| **GridWorld** | Baseline | Enhanced convergence | ⬆️ Faster learning |
| **MountainCar** | Baseline | Better sample efficiency | ⬆️ Reduced episodes |
| **CartPole** | Baseline | More stable training | ⬆️ Consistent performance |

---

## 📈 Results & Analysis

### 🏆 Training Performance

The implementation demonstrates successful learning across all environments:

1. **GridWorld**: Agent learns optimal path to Princess Peach
2. **MountainCar**: Solves the momentum-building challenge
3. **CartPole**: Achieves stable pole balancing

### 📊 Key Metrics Tracked:
- **Episode Rewards**: Total reward per episode
- **Epsilon Decay**: Exploration rate over time
- **Loss Values**: Training stability indicators
- **Success Rate**: Task completion percentage

### 🔍 Observations:
- PER shows improved sample efficiency in sparse reward environments
- Convolutional networks effectively process visual observations
- Proper hyperparameter tuning crucial for stable learning

---

## 🛠️ Installation & Usage

### Prerequisites:
```bash
pip install torch torchvision
pip install gymnasium
pip install matplotlib numpy
pip install jupyter notebook
```

### Running the Code:
1. **Part 1 - Neural Architectures**:
   ```bash
   jupyter notebook roopeshv_hvenkatr_assignment2_part1.ipynb
   ```

2. **Part 2 - DQN Training**:
   ```bash
   jupyter notebook roopeshv_hvenkatr_assignment2_part2.ipynb
   ```

3. **Load Pre-trained Models**:
   ```python
   import torch
   model = torch.load('roopeshv_hvenkatr_assignment2_part2_dqn_cartpole.pickle')
   ```

---

## 🔬 Technical Implementation Details

### 🧪 Experience Replay Buffer
```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

### 🎯 Prioritized Experience Replay
- **Priority Calculation**: |TD-error| + ε
- **Sampling Probability**: P(i) = p_i^α / Σ p_k^α
- **Importance Sampling Weight**: w_i = (1/N * 1/P(i))^β

### 🔄 Training Loop
1. **Environment Interaction**: Agent takes action, observes reward
2. **Experience Storage**: Store transition in replay buffer
3. **Batch Sampling**: Sample mini-batch from buffer
4. **Q-Learning Update**: Update policy network
5. **Target Update**: Soft update of target network

---

## 📚 Key Concepts Demonstrated

### 🎓 Reinforcement Learning Fundamentals:
- **Markov Decision Processes (MDPs)**
- **Value Function Approximation**
- **Temporal Difference Learning**
- **Exploration vs Exploitation Trade-off**

### 🤖 Deep Learning Integration:
- **Function Approximation with Neural Networks**
- **Convolutional Networks for Visual Input**
- **Experience Replay for Stable Learning**
- **Target Networks for Training Stability**

### 🔧 Advanced Techniques:
- **Prioritized Experience Replay**
- **Double DQN (implicit through target networks)**
- **Epsilon-Greedy with Exponential Decay**
- **Gradient Clipping for Training Stability**

---

## 🎯 Future Enhancements

### Potential Improvements:
- **Double DQN**: Reduce overestimation bias
- **Dueling DQN**: Separate value and advantage estimation
- **Rainbow DQN**: Combine multiple improvements
- **Distributional RL**: Model full return distribution

### Additional Environments:
- **Atari Games**: Visual complexity
- **Continuous Control**: DDPG/TD3 algorithms
- **Multi-Agent**: Competitive/Cooperative scenarios

---

## 📖 References

1. **Mnih, V. et al.** (2015). Human-level control through deep reinforcement learning. *Nature*
2. **Schaul, T. et al.** (2016). Prioritized Experience Replay. *ICLR*
3. **Russell, S. & Norvig, P.** Artificial Intelligence: A Modern Approach
4. **Sutton, R. & Barto, A.** Reinforcement Learning: An Introduction

---

## 🏫 Academic Context

**This project is for educational purposes as part of CSE 446/546 coursework.**

### Course Learning Outcomes:
- ✅ Understanding of value function approximation
- ✅ Implementation of deep reinforcement learning algorithms
- ✅ Analysis of different neural network architectures
- ✅ Evaluation of algorithm improvements and variations
- ✅ Practical experience with OpenAI Gym environments

---

## 👥 Contributors

**Authors**: Roopesh V, Hvenkatr  
**Course**: CSE 446/546 - Reinforcement Learning  
**Institution**: University of Washington  

---

## 📄 License

This project is licensed under the Educational Use License - see the [LICENSE](LICENSE) file for details.

---

*Made with ❤️ for advancing the understanding of Deep Reinforcement Learning*