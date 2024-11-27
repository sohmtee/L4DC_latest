import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridEnv(gym.Env):
    def __init__(self, grid_size=4, coins_count=0, horizon=6, success_probability = 0.91):
        super().__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        self.coins_count = coins_count
        lows = [0,0]
        self.feature_dim = 4 + coins_count
        highs = [grid_size-1, grid_size-1]
        self.num_states = grid_size * grid_size * (2 ** coins_count)
        for i in range(coins_count):
            lows.append(0)
            highs.append(1)
        self.observation_space = spaces.Box(low=np.array(lows), high=np.array(highs), dtype=np.int32), # agent position and coin picked
        
        self.init_goal = True
        self.horizon = horizon
        self.reward_classes = coins_count + 3
        if coins_count == 0:
            self.reward_classes += 1
        self.success_probability = success_probability


    def _generate_new_position(self, existing_positions):
        new_pos = (self.np_random.integers(self.grid_size), self.np_random.integers(self.grid_size))
        trials = 0
        while new_pos in existing_positions:
            new_pos = (self.np_random.integers(self.grid_size), self.np_random.integers(self.grid_size))
            trials += 1
            if trials == 20:
                raise Exception ("Cannot generate empty position.")
        return new_pos
    def _generate_non_conflicting_position(self, count):
        existing_positions = []
        for _ in range(count):
            new_pos = self._generate_new_position(existing_positions)
            existing_positions.append(new_pos)
        return existing_positions
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        if self.init_goal:
            self.init_goal = False
            new_positions_count = 3 + self.coins_count
            self.objects_positions = self._generate_non_conflicting_position(new_positions_count)
            self.goal_position = self.objects_positions[0]
            self.lightning_position = self.objects_positions[1]
            self.mountain_position = self.objects_positions[2]
            self.coins_position = [self.objects_positions[3+i] for i in range(self.coins_count)]
                

        self.coins_collected = np.array([False for _ in range(self.coins_count)], dtype=np.bool8)
        self.agent_position = self._generate_new_position([self.mountain_position])
        self.done = False
        self._t = 0
        return self._get_observation()

    def _normalized_manhattan_distance(self, state1, state2):
        return int(abs(state1[0] - state2[0]) + abs(state1[1] - state2[1]))/((self.grid_size-1) * 2)
    
    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        reward = 0
        self._t +=1
        if self.np_random.random() > self.success_probability:
            actions = [i for i in range(self.action_space.n)]
            actions.remove(action)
            index = self.np_random.integers(self.action_space.n - 1)
            action = actions[index]
       
       
        if self.agent_position in self.coins_position:
            for i in range(self.coins_count): #Todo: find a faster way to do it
                if self.agent_position == self.coins_position[i]:
                    self.coins_collected[i] = True
                    break
        # Check current cell
        if self.agent_position == self.lightning_position:
            self.done = True
        elif self.agent_position == self.goal_position:
            if np.sum(self.coins_collected) == self.coins_count:
                self.done = True
                reward = self.reward_classes - 1
        if self._t == self.horizon and not self.done:
            self.done = True
            reward = 1 + np.sum(self.coins_collected)
            if self.coins_count == 0 and self._normalized_manhattan_distance(self.agent_position, self.goal_position) >= self._normalized_manhattan_distance(self.agent_position, self.lightning_position):
                reward = 2
            

        if  (self.agent_position != self.lightning_position and self.agent_position != self.goal_position):
            # Update agent position
            x, y = self.agent_position
            if action == 0 and x > 0:  # Up
                x -= 1
            elif action == 1 and x < self.grid_size - 1:  # Down
                x += 1
            elif action == 2 and y > 0:  # Left
                y -= 1
            elif action == 3 and y < self.grid_size - 1:  # Right
                y += 1

            
            if (x,y) != self.mountain_position:
                self.agent_position = (x, y)
        
        info = {}
        if self.done:
            info["encoding"] = self._get_encoding()
        return self._get_observation(), reward, self.done, info
    
    
    def _get_observation(self):
        obs = np.zeros((2 + self.coins_count), dtype=np.int32)
        obs[0:2] = self.agent_position
        obs[2:] = self.coins_collected
        return obs

    def _get_encoding(self):
        feature = np.zeros(self.feature_dim)
        feature[0] = self._normalized_manhattan_distance(self.agent_position, self.goal_position)
        feature[1] = self._normalized_manhattan_distance(self.agent_position, self.lightning_position)
        feature[2] = int(self.agent_position == self.goal_position)
        feature[3] = int(self.agent_position == self.lightning_position)
        if self.coins_count > 0:
            feature[4:] = self.coins_collected
        feature = feature / (np.sqrt(self.feature_dim))
        assert np.linalg.norm(feature) <= 1
        return feature

    def get_state_index(self, state):
        coins = 0
        if len(state) > 2:
            coins = state[2:]
            coins = coins.dot(2**np.arange(coins.size)[::-1])
        index = coins * self.grid_size * self.grid_size + state[0] * self.grid_size + state[1]
        return index