import numpy as np

def train_agent(agent, env, num_episodes=1000, alpha=0.1):
    """
    Train the RL agent using TD(0) learning.

    Args:
        agent (RLAligner): The RL agent to be trained.
        env (SequenceAlignmentEnvironment): The environment for sequence alignment.
        num_episodes (int): Number of episodes to train the agent.
        alpha (float): Learning rate for TD(0) updates.
    """
    for episode in range(num_episodes):
        # Reset environment for a new episode
        env.reset()

        done = False
        while not done:
            # Select action
            action, pos = agent.select_action()

            # Take action and observe next state and reward
            next_state, reward, done = env.step(action, pos)

            # Update value function using TD(0) update rule
            agent.update_value_function(action, reward, next_state, alpha)

            # Move to next state
            action = next_state

        # Print progress or metrics if needed
        if (episode + 1) % 100 == 0:
            print("Episode:", episode + 1)

    print("Training completed.")

