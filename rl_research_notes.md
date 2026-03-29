# Member 2: AI Engineer - RL/PPO Research Notes

## Overview of PPO (Proximal Policy Optimization)
PPO is a popular Reinforcement Learning algorithm developed by OpenAI. It is categorized as a **Policy Gradient** method and is widely used for its stability and ease of tuning.

### Key Concepts
1. **Actor-Critic Architecture**:
   - **Actor**: Decides which action to take (Policy).
   - **Critic**: Evaluates the action by predicting the value of the current state.
2. **Clipping Objective**: 
   - PPO prevents the policy from changing too drastically in a single update, which ensures stable training even with noisy data.
3. **Exploration vs. Exploitation**:
   - Uses an epsilon-greedy or entropy-based approach to ensure the robot tries new paths while gradually favoring the most efficient ones.

## Application in Warehouse AMR
- **State Space**: [Robot X, Robot Y, Target X, Target Y, Orientation].
- **Action Space**: [Linear Velocity, Angular Velocity].
- **Reward Function**: 
  - `+100` for reaching the target.
  - `-1` per step to encourage speed.
  - `-10` for collisions with shelves.

## Day 1 Conclusion
Successfully understood the core PPO logic. Ready to implement the training loop using the environment established by Member 1.
