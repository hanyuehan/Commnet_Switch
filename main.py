import torch
import gym
import ma_gym
from Memory import Memory
from PPO import PPO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    ############## Hyperparameters ##############
    env_name = "Switch2-v0"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    n_agents = env.n_agents

    render = False
    solved_length = 40  # stop training if avg_length > solved_length
    solved_reward = 90  # stop training if avg_length > solved_length
    log_interval = 20  # print avg reward in the interval
    max_episodes = 500000  # max training episodes
    max_timesteps = 100  # max timesteps in one episode
    update_timestep = 2000  # update policy every n timesteps
    lr = 0.0002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 5  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_agents, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            pos, state, reward, done, _ = env.step(action)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                loss = ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += sum(reward)
            if render:
                env.render()
            if all(done):
                break

        avg_length += t

        # logging
        if i_episode % log_interval == 0:
            avg_length = avg_length / log_interval
            running_reward = running_reward / log_interval

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))

            # stop training if avg_reward > solved_reward
            if avg_length < solved_length and running_reward > solved_reward:
                print("########## Solved! ##########")
                torch.save(ppo.policy.state_dict(), './Solved_length_40_PPO_{}.pth'.format(env_name))
                break

            running_reward = 0
            avg_length = 0

        if i_episode % 5000 == 0 :
            torch.save(ppo.policy.state_dict(), './PPO_{}_{}_hy.pth'.format(env_name, i_episode))


if __name__ == '__main__':
    main()
