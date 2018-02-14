import collections
import gym
import matplotlib.pyplot as plt
import numpy as np
import pdb
import random
import tensorflow as tf
import time

Transition = collections.namedtuple('Transition', ('state',
                                                   'action',
                                                   'reward',
                                                   'target'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = collections.deque()

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        if len(self.memory) >= self.capacity:
            self.memory.popleft()
        self.memory.append(Transition(*args))

    def sample(self, size):
        batch_size = min(size, len(self.memory))
        return random.sample(self.memory, batch_size)

def main():
    # parameters
    # network
    gamma = 0.9
    learning_rate = 0.001
    # game
    nepoch = 10000
    maxstep = 300
    ntest = 10
    # rl
    # eps_end is reach at the last epoch
    eps0 = 0.5
    epsn = 0.01
    memory_size = 1000
    batch_size = 50
    
    # =====build network========================================================
    # network structure
    I_SIZE = 4;
    H_SIZE = 20;
    O_SIZE = 2;
    # define placeholder for inputs to network
    tf_state = tf.placeholder(tf.float32, shape=[None, I_SIZE])
    tf_action = tf.placeholder(tf.uint8, shape=[None])
    tf_reward = tf.placeholder(tf.float32, shape=[None])
    tf_target = tf.placeholder(tf.float32, shape=[None])
    # add hidden layer
    W1 = tf.Variable(tf.random_normal([I_SIZE, H_SIZE], stddev=0.5))
    b1 = tf.Variable(tf.random_normal([1, H_SIZE], stddev=0.5))
    # add output layer
    W2 = tf.Variable(tf.random_normal([H_SIZE, O_SIZE], stddev=0.5))
    b2 = tf.Variable(tf.random_normal([1, O_SIZE], stddev=0.5))
    # nn output for state and next state
    prediction = tf.matmul(tf.nn.relu(tf.matmul(tf_state, W1) + b1), W2) + b2
    # loss function
    onehot = tf.one_hot(tf_action, depth=O_SIZE, axis=1)
    prediction_value = tf.reduce_sum(tf.multiply(prediction, onehot), axis=1)
    #test = tf.square(tf_target - prediction_value)
    loss = tf.reduce_mean(tf.square(tf_target - prediction_value))
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #===========================================================================
    
    # start gym, tf session, memory
    env = gym.make('CartPole-v0')
    #env = gym.make('Hopper-v1')
    #print(env.action_space)
    #print(env.observation_space)
    sess = tf.Session()
    #sess.run(tf.global_variables_initializer())
    tf.train.Saver().restore(sess, "./main/save.ckpt")
    memory = ReplayMemory(memory_size)
    record = np.zeros(nepoch)
    
    # main loop
    for epo in range(nepoch):
        state0 = env.reset()
        score = 0.
        #print(state0)
        for step in range(maxstep):
            #env.render()
            
            # epsilon greedy
            eps = eps0*np.power(epsn/eps0, epo/(nepoch-1.))
            rand = np.random.random()
            if rand < eps:
                action = env.action_space.sample()
            else:
                value_state0 = sess.run(prediction, feed_dict={tf_state:state0[np.newaxis,:]})
                action = np.argmax(value_state0)
            
            # do it, calculate target Q value
            state1, reward, done, info = env.step(action)
            reward = -1. if done else 0.1
            #reward = 1 - 0.4*(abs(state1[0])+abs(state1[2]))
            value_state1 = sess.run(prediction, feed_dict={tf_state:state1[np.newaxis,:]})
            target = reward + gamma*np.max(value_state1)
            
            # push to memory
            memory.push(state0, action, reward, target)
            state0 = state1
            score += reward
            
            #print('{} {} {} {}'.format(state1, action, reward, done))
            if done:
                #print(score/float(step), end='')
                break
        
        # if memory is large enough, train
        if len(memory) == memory.capacity:
            batch = Transition(*zip(*memory.sample(batch_size)))
            sess.run(train, feed_dict={tf_state:batch.state,
                                       tf_action:batch.action,
                                       tf_reward:batch.reward,
                                       tf_target:batch.target})
        record[epo] = score
        
        # test the network without randomness
        if epo % 100 == 0:
            test_reward = 0.
            for i in range(ntest):
                state0 = env.reset()
                for j in range(maxstep):
                    env.render()
                    action = np.argmax(sess.run(prediction,
                                       feed_dict={tf_state:state0[np.newaxis,:]}))
                    state0, reward, done, info= env.step(action)
                    test_reward += reward
                    time.sleep(0.01)
                    if done:
                        break
            avg_reward = test_reward / ntest
            print('episode: ', epo, 'test average reward: ', avg_reward)
            # if the game is solved, quit
            if avg_reward >= 200:
                break
    
    plt.plot(record)
    plt.show()
    tf.train.Saver().save(sess, "./main/save.ckpt")
    env.close()
    sess.close()

if __name__ == "__main__":
    main()
