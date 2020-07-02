import tensorflow as tf

from TD3_BipedalWalker import TD3
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import gym


def prob2action(action_array):
    max_i = 0
    max_value = action_array[0]
    for i in range(1, len(action_array)):
        if action_array[i] > max_value:
            max_i = i
            max_value = action_array[i]
    return max_i


EPISODES = 1000
max_episode_length = 2000
train_interval = 1
policy_delay = 2
load_model = False
random.seed(0)
# load scenario from script


env = gym.make('BipedalWalker-v3')

obs_shape = env.observation_space.shape
n_actions = 4
action_shape = (n_actions, )
agent = TD3(observation_shape=obs_shape, action_shape=action_shape, param_noise=None, action_noise=None,
             gamma=0.99, tau=0.005, batch_size=100, memory_max=int(800000), actor_lr=1e-3, critic_lr=1e-3, continuous_action=True)

sess = tf.Session()
agent.initialize(sess)
writer = tf.summary.FileWriter("E:/Research/Program/OCBARL/logs/", sess.graph)
writer.add_graph(sess.graph)
saver = tf.train.Saver()
ckpt_path = 'trained_model'
Rewards = []
if load_model == True:
    saver.restore(sess, ckpt_path)
obs_ = None
t = 0
train_step = 0
avg = 0
avg_train_time = 0
avg_update_time = 0
noise = False
log_f = open("log.txt","w+")
for episode in range(1, EPISODES + 1):
    episode_reward = 0
    episode_train_time = 0
    episode_update_time = 0
    start = True
    t_1 = t
    for _ in range(max_episode_length):
        t += 1

        if start == True:
            obs = env.reset()
            start = False
        else:
            obs = obs_
        # time1 = time.time()
        action = agent.step(obs).reshape(n_actions, )
        action = action + np.random.normal(0, 0.1, size=env.action_space.shape[0])
        action = np.clip(action, -1, 1)
        # time2 = time.time()
        # rand = random.uniform(0, 1)
        # if rand < noise:
        #     action = env.action_space.sample()
        obs_, reward, terminal, info = env.step(action)
        # print(time2 - time1)
        episode_reward += reward
        agent.store(obs, action, reward, obs_, terminal)



        if terminal == True:
            break
    # print(t - t_1)
    if len(agent.memory.obs) >= agent.batch_size:
        for train_i in range(t - t_1):
            if train_i % (policy_delay * train_interval) == 0:
                time3 = time.time()
                agent.train_batch(update_policy=True)
                time4 = time.time()
                agent.update_target_net()
                time5 = time.time()
                episode_train_time += time4 - time3
                episode_update_time += time5 - time4

                # print(time4 - time3)
            elif train_i % train_interval == 0:
                agent.train_batch(update_policy=False)
        train_step += 1
    log_f.write('{},{}\n'.format(episode, episode_reward))
    log_f.flush()
    avg += episode_reward / 10
    avg_train_time += episode_train_time / 10 / (t - t_1) * train_interval
    avg_update_time += episode_update_time / 10 / (t - t_1) * train_interval
    Rewards.append(episode_reward)
    if episode % 10 == 0:
        # noise *= 0.9

        print("Episode %d average reward: %.3f  average training time: %.4f  %.4f" % (episode, avg, avg_train_time, avg_update_time))
        avg = 0
        avg_train_time = 0
        avg_update_time = 0

    if episode % 100 == 0:
        checkpoint = 'check_point_episode_%d' % episode
        saver.save(sess, checkpoint)

Rewards = np.array(Rewards)
divide = 200
for i in range(len(Rewards) // divide):
    start_i = i * divide
    end_i = (i + 1) * divide
    print('Average Reward of Episode %d to %d : %.3f' % (start_i, end_i, np.mean(Rewards[start_i : end_i])))
plt.plot(Rewards)
#plt.show()
for i in range(1):
    start = True
    for _ in range(max_episode_length):
        t += 1

        if start == True:
            obs = env.reset()
            start = False
        else:
            obs = obs_
        # time1 = time.time()
        action = agent.step(obs).reshape(n_actions, )
        print(action)
        # time2 = time.time()
        obs_, reward, terminal, info = env.step(action)
        if terminal:
            break
        env.render()
        time.sleep(0.02)


saver.save(sess, ckpt_path)
