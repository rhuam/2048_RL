import pickle
from pprint import pprint

import logic
import numpy as np
import math


class QLPlayer():

    def __init__(self, matriz, actions, alpha=0.5, gamma=0.95, epsilon=0.9):
        self.set_state(matriz)
        self.action = actions

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_old = epsilon
        self.epsilon = epsilon

        self.q_table = {i: dict() for i in self.action}
        self._reward = 0


    def newgame(self, matriz):
        self.set_state(matriz)
        self.epsilon = self.epsilon_old

    @property
    def reward(self):
        return self._reward

    def set_reward(self, state_old):
        pay_old = [0,0,0,0,0,0,0,0,0,0]
        pay = [0,0,0,0,0,0,0,0,0,0]
        score = 0
        for i, p in enumerate(pay):
            x = self.state.count(2**(i+1)) - state_old.count(2**(i+1))
            pay[i] = x if x >= 0 else 0
            score += (pay[i] * 2**(i+1))**2

        # print(pay, score)
        self._reward = score

    def ql(self):
        self.state_rep = ''.join(str(e) for e in self.state)
        action = self.epsilon_greedy()

        state_old = self.state
        done = self.act(action)
        self.set_reward(state_old)

        if not self.state_rep in self.q_table[action].keys():
            self.q_table[action][self.state_rep] = 0

        self.q_table[action][self.state_rep] += self.alpha * (self._reward - self.q_table[action][self.state_rep])

        return done

    def epsilon_greedy(self, train=False):
        max = 0
        act = np.random.randint(0, 4)

        self.epsilon = self.epsilon * self.gamma

        if np.random.rand() < self.epsilon:
            return self.action[act]
        else:
            for i, a in enumerate(self.action):
                if self.state_rep in self.q_table[a].keys():
                    if self.q_table[a][self.state_rep] > max:
                        max = self.q_table[a][self.state_rep]
                        act = i
        return self.action[act]

    def set_state(self, matriz):
        self.matriz = logic.add_two(matriz)
        self.state = [int(math.log2(_)) if _ > 0 else 0 for s in matriz for _ in s]
        self.state_rep = ''.join(str(e) for e in self.state)

    def act(self, action):
        if action == 'up':
            matriz, done = logic.up(self.matriz)
            self.set_state(matriz)
        elif action == 'down':
            matriz, done = logic.down(self.matriz)
            self.set_state(matriz)
        elif action == 'right':
            matriz, done = logic.right(self.matriz)
            self.set_state(matriz)
        elif action == 'left':
            matriz, done = logic.left(self.matriz)
            self.set_state(matriz)

        return not self.state.count(0) == 0


def init_matrix():
    matrix = logic.new_game(4)
    matrix = logic.add_two(matrix)
    matrix = logic.add_two(matrix)

    return matrix




if __name__ == '__main__':
    from tqdm import tqdm
    state = init_matrix()
    player = QLPlayer(state, ['up', 'down', 'right', 'left'])
    done = True

    for i in tqdm(range(1000000)):
        while done:
            done = player.ql()
            # print(player.state)

        # print("New Game")
        state = init_matrix()
        player.newgame(state)
        done = True

    with open('model.pickle', 'wb') as f:
        pickle.dump(player, f)

    with open('model.pickle', 'rb') as f:
        player = pickle.load(f)

    while done:
        done = player.ql()
        print(player.matriz)

    # pprint(player.q_table)
