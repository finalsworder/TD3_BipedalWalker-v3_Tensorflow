import tensorflow as tf
import numpy as np
from copy import copy
import tensorflow.contrib.layers as layers
import random
import time

# class Feature():
#     def __init__(self, n_features, name='feature'):
#         self.n_features = n_features
#         self.name = name
#
#     def __call__(self, image):
#         with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
#             x = image
#             x = layers.conv2d(inputs=x, filters=32, kernel_size=[3,3], strides=1, padding='same',
#                               activation=tf.nn.relu)
#             x = layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3], strides=1, padding='same',
#                               activation=tf.nn.relu)
#             x = layers.max_pool2d(inputs=x, pool_size=[2,2], strides=2)
#             x = layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3], strides=1, padding='same',
#                               activation=tf.nn.relu)
#             x = layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3], strides=1, padding='same',
#                               activation=tf.nn.relu)
#             x = layers.max_pool2d(inputs=x, pool_size=[2, 2], strides=2)
#             x = tf.reshape(x, x.shape[0] * x.shape[1] * x.shape[2])
#             return x
#
#     def trainable_vars(self):
#         return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
#
#     def vars(self):
#         return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


class Actor():
    def __init__(self, n_actions, name='actor'):
        self.n_actions = n_actions
        self.name = name

    def __call__(self, obs, continuous_action, image=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = obs
            if image == True:
                x = layers.conv2d(inputs=x, num_outputs=32, kernel_size=[3, 3], stride=1, padding='same',
                                  activation_fn=tf.nn.relu)
                x = layers.conv2d(inputs=x, num_outputs=32, kernel_size=[3, 3], stride=1, padding='same',
                                  activation_fn=tf.nn.relu)
                x = layers.max_pool2d(inputs=x, kernel_size=[2, 2], stride=2)
                x = layers.conv2d(inputs=x, num_outputs=32, kernel_size=[3, 3], stride=1, padding='same',
                                  activation_fn=tf.nn.relu)
                x = layers.conv2d(inputs=x, num_outputs=32, kernel_size=[3, 3], stride=1, padding='same',
                                  activation_fn=tf.nn.relu)
                x = layers.max_pool2d(inputs=x, kernel_size=[2, 2], stride=2)
                x = layers.flatten(x)
            x = layers.fully_connected(x, num_outputs=400, activation_fn=tf.nn.relu)
            x = layers.fully_connected(x, num_outputs=300, activation_fn=tf.nn.relu)
            x = layers.fully_connected(x, num_outputs=self.n_actions, activation_fn=tf.nn.tanh)
            # if continuous_action:
            # x = tf.tanh(x)
            # x = tf.clip_by_value(x, 0, 1)
            return x

    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

class Critic():
    def __init__(self, name='critic'):
        self.name = name

    def __call__(self, obs, action, image=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # x = obs
            # if image == True:
            #     x = layers.conv2d(inputs=x, num_outputs=32, kernel_size=[3, 3], stride=1, padding='same',
            #                       activation_fn=tf.nn.relu)
            #     x = layers.conv2d(inputs=x, num_outputs=32, kernel_size=[3, 3], stride=1, padding='same',
            #                       activation_fn=tf.nn.relu)
            #     x = layers.max_pool2d(inputs=x, kernel_size=[2, 2], stride=2)
            #     x = layers.conv2d(inputs=x, num_outputs=32, kernel_size=[3, 3], stride=1, padding='same',
            #                       activation_fn=tf.nn.relu)
            #     x = layers.conv2d(inputs=x, num_outputs=32, kernel_size=[3, 3], stride=1, padding='same',
            #                       activation_fn=tf.nn.relu)
            #     x = layers.max_pool2d(inputs=x, kernel_size=[2, 2], stride=2)
            #     x = layers.flatten(x)
            x = tf.concat([obs, action], axis=-1)

            x = layers.fully_connected(x, num_outputs=400, activation_fn=tf.nn.relu)
            x = layers.fully_connected(x, num_outputs=300, activation_fn=tf.nn.relu)
            x = layers.fully_connected(x, num_outputs=1, activation_fn=None)
            return x

    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

class OUNoise():
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed=0, mu=0., theta=0.4, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state

def sample(logits):
    u = tf.random_uniform(tf.shape(logits))
    x = tf.nn.softmax(logits - tf.log(-tf.log(u)), axis=-1)

    return x


class Memory():
    def __init__(self, size):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.obs_ = []
        self.terminals = []
        self.full = False
        self.node = 0
        self.size = size

    def append(self, obs, action, reward, obs_, terminal):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.obs_.append(obs_)
        self.terminals.append(terminal)

    def replace(self, obs, action, reward, obs_, terminal, index):
        self.obs[index] = obs
        self.actions[index] = action
        self.rewards[index] = reward
        self.obs_[index] = obs_
        self.terminals[index] = terminal

    def store(self, obs, action, reward, obs_, terminal):
        if self.full == False:
            self.append(obs, action, reward, obs_, terminal)
            if len(self.obs) == self.size:
                self.full = True
        else:
            self.replace(obs, action, reward, obs_, terminal, self.node)
            self.node += 1
            if self.node >= self.size:
                self.node = 0

    def sample(self, batch_size):
        # ids = list(range(0, len(self.obs)))
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_obs_ = []
        batch_terminals = []
        batch_ids = np.random.randint(0, len(self.obs), size=batch_size)
        # batch_ids = random.sample(ids, batch_size)
        for i in batch_ids:
            batch_obs.append(self.obs[i])
            batch_actions.append(self.actions[i])
            batch_rewards.append(self.rewards[i])
            batch_obs_ .append(self.obs_[i])
            batch_terminals.append(self.terminals[i])
        return np.array(batch_obs), np.array(batch_actions), np.array(batch_rewards), np.array(batch_obs_), np.array(batch_terminals)




def minimize_and_clip(optimizer, objective, var_list, clip_val=0.5):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    if clip_val is None:
        return optimizer.minimize(objective, var_list=var_list)
    else:
        gradients = optimizer.compute_gradients(objective, var_list=var_list)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
        return optimizer.apply_gradients(gradients)

class TD3():
    def __init__(self, observation_shape, action_shape, param_noise=None, action_noise=None, clip_value=None,
                 gamma=0.99, tau=0.001, batch_size=100, memory_max=10000, actor_lr=1e-2, critic_lr=1e-2,
                 image=False, continuous_action=False):
        # Inputs.
        self.obs = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs')
        self.obs_ = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs_')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
        self.critic_target1 = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target1')
        self.critic_target2 = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target2')
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')


        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.continous_action = continuous_action
        self.critic1 = Critic(name='critic1')
        self.critic2 = Critic(name='critic2')
        self.actor = Actor(action_shape[-1])
        self.action = self.actor(self.obs, continuous_action=self.continous_action, image=image)
        self.noise = tf.random.normal(mean=0, stddev=0.2, shape=action_shape)
        self.noise = tf.clip_by_value(self.noise, -0.5, 0.5)
        # if self.continous_action == False:
        #     self.action = sample(self.action_raw)
        # else:
        #     self.action = tf.nn.tanh(self.action_raw)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.stats_sample = None
        self.image = image

        # Outputs
        # self.feature = self.feature
        self.critic_with_actor1 = self.critic1(self.obs, self.action, image=image)
        # self.critic_with_actor2 = self.critic2(self.obs, self.action, image=image)
        self.target_actor = copy(self.actor)
        self.target_actor.name = 'target_actor'
        self.target_critic1 = copy(self.critic1)
        self.target_critic1.name = 'target_critic1'
        self.target_critic2 = copy(self.critic1)
        self.target_critic2.name = 'target_critic2'
        self.target_action = self.target_actor(self.obs_, continuous_action=self.continous_action, image=image)
        self.target_action = self.target_action + self.noise
        self.target_action = tf.clip_by_value(self.target_action, -1., 1.)
        # if self.continous_action == False:
        #     self.target_action = sample(self.target_action_raw)
        # else:
        #     self.target_action = tf.nn.tanh(self.target_action_raw)

        self.q_obs1 = self.target_critic1(self.obs_, self.target_action, image=image)
        self.q_obs2 = self.target_critic2(self.obs_, self.target_action, image=image)
        self.target_Q = self.rewards + (1 - self.terminals1) * gamma * tf.minimum(self.q_obs1, self.q_obs2)

        # Memory
        self.memory = Memory(size=memory_max)

        # Loss
        self.Q_loss1 = tf.reduce_mean(tf.square(self.critic1(self.obs, self.actions) - self.target_Q))
        self.Q_loss2 = tf.reduce_mean(tf.square(self.critic2(self.obs, self.actions) - self.target_Q))
        # self.p_reg = tf.reduce_mean(tf.square(self.action_raw))
        # if self.continous_action == False:
        #     self.P_loss = - tf.reduce_mean(self.critic_with_actor) + self.p_reg * 0.001
        # else:
        #     self.P_loss = - tf.reduce_mean(self.critic_with_actor) + self.p_reg * 0.001
        self.P_loss = - tf.reduce_mean(self.critic_with_actor1)

        # Optimizer

        self.optimizer = tf.train.AdamOptimizer(learning_rate=actor_lr)
        self.actor_train = minimize_and_clip(optimizer=self.optimizer, objective=self.P_loss,
                                             var_list=self.actor.trainable_vars(), clip_val=clip_value)
        self.critic_train1 = minimize_and_clip(optimizer=self.optimizer, objective=self.Q_loss1,
                                               var_list=self.critic1.trainable_vars(), clip_val=clip_value)
        self.critic_train2 = minimize_and_clip(optimizer=self.optimizer, objective=self.Q_loss2,
                                               var_list=self.critic2.trainable_vars(), clip_val=clip_value)
        # Update
        self.update_target_actor = []
        self.update_target_critic1 = []
        self.update_target_critic2 = []
        length_actor = len(self.target_actor.vars())
        for i in range(length_actor):
            self.update_target_actor.append(
                tf.assign(self.target_actor.vars()[i], (1 - self.tau) * self.target_actor.vars()[i] + self.tau * self.actor.vars()[i]))
        length_critic1 = len(self.target_critic1.vars())
        for i in range(length_critic1):
            self.update_target_critic1.append(
                tf.assign(self.target_critic1.vars()[i], (1 - self.tau) * self.target_critic1.vars()[i] + self.tau * self.critic1.vars()[i]))
        length_critic2 = len(self.target_critic2.vars())
        for i in range(length_critic2):
            self.update_target_critic2.append(
                tf.assign(self.target_critic2.vars()[i],
                          (1 - self.tau) * self.target_critic2.vars()[i] + self.tau * self.critic2.vars()[i]))


    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def step(self, obs):

        shape = (1, ) + obs.shape
        obs = obs.reshape(shape)
        feed_dict = {self.obs : obs}
        action = self.sess.run(self.action, feed_dict=feed_dict)
        # if noise:
        #     # action += self.noise.sample()
        #     None
        return action

    def store(self, obs, action, reward, obs_, terminal):

        self.memory.store(obs, action, reward, obs_, terminal)

    def update_target_net(self):
        time1 = time.time()
        self.sess.run(self.update_target_actor)
        self.sess.run(self.update_target_critic1)
        self.sess.run(self.update_target_critic2)
        time2 = time.time()
        # print(time2 - time1)


    def train_batch(self, update_policy):
        time1 = time.time()
        batch = self.memory.sample(self.batch_size)

        feed_dict_critic = {self.obs : batch[0],
                            self.actions : batch[1],
                            self.rewards : batch[2].reshape(self.batch_size, 1),
                            self.obs_ : batch[3],
                            self.terminals1 : batch[4].reshape(self.batch_size, 1)}
        # time2 = time.time()
        self.sess.run(self.critic_train1, feed_dict=feed_dict_critic)
        self.sess.run(self.critic_train2, feed_dict=feed_dict_critic)
        # time3 = time.time()
        if update_policy:
            feed_dict_actor = {self.obs : batch[0]}
            self.sess.run(self.actor_train, feed_dict=feed_dict_actor)
        # time4 = time.time()
        # if update_policy:
            # print(time2 - time1, time3 - time2, time4 - time3)

    def noise_reset(self):
        self.noise.reset()

# self.sess.run([self.critic1.vars()[0][0][[0]], self.critic2.vars()[0][0][[0]],
#                self.target_critic1.vars()[0][0][[0]], self.target_critic2.vars()[0][0][[0]],
#                self.actor.vars()[0][0][[0]], self.target_actor.vars()[0][0][[0]]])