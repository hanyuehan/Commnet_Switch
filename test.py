from PIL import Image
import torch
import gym
import ma_gym
from Memory import Memory
from PPO import PPO
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():
    ############## Hyperparameters ##############
    env_name = "Switch2-v0"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    n_agents = env.n_agents

    lr = 0.0001
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 5  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    #############################################

    n_episodes = 20
    max_timesteps = 100
    render = True
    save_gif = False

    filename = "PPO_{}.pth".format(env_name)
    directory = "./preTrained/"
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_agents, lr, betas, gamma, K_epochs, eps_clip)
    #ppo.policy_old.load_state_dict(torch.load(directory + filename))
    ppo.policy_old.load_state_dict(torch.load(directory + "PPO_Switch2-v0_hy02.pth"))


    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = ppo.policy_old.act(state, memory)
            pos, state, reward, done, _ = env.step(action)
            ep_reward += sum(reward)
            if render:
                env.render()
                time.sleep(0.05)
            if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))  
            if all(done):
                time.sleep(1)
                break
            
        print('Episode: {}\tReward: {}\t Step:{}'.format(ep, int(ep_reward), t))
        ep_reward = 0
    env.close()
    
if __name__ == '__main__':
    test()
    
    
