import gym
import numpy as np
from ddpg_torch import Agent
from ddpg_utils import plot_learning_curve
from DAHP_env_v0 import DAHP
import traceback
import matplotlib.pyplot as plt
from IPython.display import clear_output

if __name__ == '__main__':
    env = DAHP(building_path=r'D:\DOE\pythonProject\fmu\dahp_system_env_v3.fmu',
               start_day=201,
               sim_days=90,
               step_size=900,
               sim_year=2019,
               tz_name='UTC',
               occupied_hour=(6, 20),
               t_set=24,
               pump_energy_max=26.809,
               weight_reward=(0.5, 0.5, 0),
               fmu_on=True)

    if env is None:
        print('Error: Failed to load the fmu')
        quit()

    agent = Agent(alpha=0.0001, beta=0.001,
                  input_dims=(1, 8), tau=0.001,
                  batch_size=3, fc1_dims=128, fc2_dims=256,
                  n_actions=4)  # np.prod(env.action_space.nvec))
    epsds = 1
    best_score = 0  # env.reward_range[0]
    episode_rewards = []

    # Initialize plot
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='Episode Reward')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Reward')
    ax.legend()

    for i in range(epsds):
        print(f"Starting episode {i + 1}")
        try:
            observation = env.reset()
            print(f"Environment reset for episode {i + 1}, initial observation: {observation}")
        except Exception as e:
            print(f"An error occurred while resetting the environment for episode {i + 1}: {e}")
            break  # Break the loop if reset fails

        done = False
        episode_reward = []
        counter= 0
        with open(f'episode_{i + 1}_time_step_{counter}.txt', 'a') as file:
            while not done:
                action = agent.choose_action(observation)
                observation_, done, total_power, action_, reward = env.step(action)
                print(reward[0][0])

                # Store the reward for each time step within the episode
                episode_reward.append(np.sum(reward[0][0]))

                # Save action, observation, and reward to a text file
                file.write(
                    f"Time Step {counter} - Action: {action}, Observation: {observation_}, Reward: {reward[0][0]}\n")

                agent.remember(observation, action_, reward, observation_, done)
                agent.learn()
                observation = observation_
                counter += 1
        episode_rewards.append(episode_reward)

        # Update and display the plot for the current episode
        line.set_xdata(range(1, len(episode_reward) + 1))
        line.set_ydata(episode_reward)
        ax.relim()
        ax.autoscale_view()
        clear_output(wait=True)
        plt.pause(0.1)

    # Keep the plot open after completion
    plt.ioff()
    plt.show()



