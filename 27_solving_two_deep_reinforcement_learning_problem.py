import gym
import numpy as np

class tilecoder:

    def __init__(self, numTilings, tilesId):
        self.maxIn = env.observation_space.high
        self.minIn = env.observation_space.low
        self.numTilings = numTilings
        # Define 1-D Size of tiling
        self.tilesId = tilesId
        self.dim = len(self.maxIn)
        # In this case with an tilesId of 18 and numTilings of 4
        # each tile has 324 (18**dim) tiles with a total of 1296 tiles
        self.numTiles = (self.tilesId**self.dim) * self.numTilings
        self.actions = env.action_space.n
        # Combine these 1296 tiles with 3 possible actions
        # and we now have possible 3888 tiles
        self.n = self.numTiles * self.actions
        # Defines the physical size of the tile based in possinble variables
        self.tileSize = np.divide(np.subtract(self.maxIn, self.minIn), self.tilesId-1)

    # Takes our current state and returns 4 integers / tile indices
    def getFeatures(self, variables):
        # Ensure lowest possible input is always 0
        self.variables = np.subtract(variables, self.minIn)
        tileIndices = np.zeros(self.numTilings)
        # Will take in state space and convert into tile indices
        matrix = np.zeros([self.numTilings, self.dim])
        for i in range(self.numTilings):
            for i2 in range(self.dim):
                 matrix[i, i2] = int(self.variables[i2]/ self.tileSize[i2] \
                    + i / self.numTilings)
        for i in range(1,self.dim):
            matrix[:,i] *= self.tilesId**i
        for i in range(self.numTilings):
            tileIndices[i] = (i * (self.tilesId**self.dim)\
                + sum(matrix[i,:]))
        return tileIndices

    # Assigns action values for all possible actions
    def getQ(self, features, theta):
        Q = np.zeros(self.actions)
        for i in range(self.actions):
            Q[i] = tile.getVal(theta, features, i)
        return Q
    
    # Calculate action values based upon theta
    def getVal(self, theta, features, action):
        val = 0
        for i in features:
            index = int(i + (self.numTiles*action))
            val += theta[index]
        return val

    # Creates a one hot vector for features so that theta can be updated
    def oneHotVector(self, features, action):
        oneHot = np.zeros(self.n)
        for i in features:
            index = int(i + (self.numTiles*action))
            oneHot[index] = 1
        return oneHot

env = gym.make("MountainCar-v0")

tile = tilecoder(4,18)
theta = np.random.uniform(-0.001, 0, size=(tile.n))
# Custom alpha learned to generalize based upon number of tilings
alpha = (.1/tile.numTilings)*3.2
# Discounting not needed since reward gives -1 reward each time step
gamma = 1
numEpisodes = 10000
stepPerEpisode = 200
rewardTracker = []
render = False
solved = False

import time

for episodeNum in range(1, numEpisodes+1):
    G = 0
    state = env.reset()
    for step in range(stepPerEpisode):
        if render:
            env.render()
        F = tile.getFeatures(state)
        Q = tile.getQ(F, theta)
        action = np.argmax(Q)
        state2, reward, done, info = env.step(action)
        G += reward
        delta = reward - Q[action]
        if done == True:
            theta += np.multiply((alpha*delta), tile.oneHotVector(F, action))
            rewardTracker.append(G)
            if episodeNum % 100 == 0:
                print("Total Episodes = {}  Episode Reward = {} Average Reward = {:04.1f}"
                        .format(episodeNum, G, np.mean(rewardTracker)))
            break
        Q = tile.getQ(tile.getFeatures(state2), theta)
        delta += gamma*np.max(Q)
        theta += np.multiply((alpha*delta), tile.oneHotVector(F, action))
        state = state2
    
    if solved !=True:
        if episodeNum > 100:
            if sum(rewardTracker[episodeNum-100:episodeNum])/100 >= -110:
                print('Solved in {} Episodes'.format(episodeNum))
                render = True
                solved = True