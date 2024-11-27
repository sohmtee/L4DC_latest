import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gymnasium as gym
from env import GridEnv
class Trainer():
    def __init__(self, hyper_params, omega_star):
        self.hyper_params = hyper_params
        self.omega_star = omega_star

    def generate_trajectory(self, theta):
        trajectory = []
        state = self.env.reset()
        done = False
        while not done:
            state_index = self.env.get_state_index(state)
            action_probs = self.softmax_policy(theta[state_index])
            if self.hyper_params["uniform_policy"]:
                action = self.rng.choice(self.num_actions)
            else:
                action = self.rng.choice(self.num_actions, p=action_probs)
            next_state, reward, done, info = self.env.step(action)
            trajectory.append((state, action))
            state = next_state
        trajectory.append((state, -1))
        return trajectory, info["encoding"], reward

    def compute_feature_matrix(self, trajectory):
        #original_feature = manhattan_distance(trajectory[-1][0], treasure_pos)
        #assert original_feature <= 1
        original_feature = torch.as_tensor(trajectory[1], dtype=torch.float32)
        with torch.no_grad():
            feature_matrix = torch.zeros(self.num_classes, self.num_classes * self.feature_dim, dtype=torch.float32)
            for class_index in range(self.num_classes):
                start_idx = class_index * self.feature_dim
                end_idx = start_idx + self.feature_dim
                feature_matrix[class_index, start_idx:end_idx] = original_feature
        return feature_matrix

    def softmax(self, x):
        exp_x = torch.exp(x - torch.max(x))
        return exp_x / exp_x.sum()

    def softmax_policy(self, theta):
        return self.softmax(theta)

    def estimation_error(self, w_star, w_hat):
        return torch.norm(w_star - w_hat, p=2).item()

    def compute_estimated_reward(self, w, trajectory):
        with torch.no_grad():
            phi = self.compute_feature_matrix(trajectory)  
            logits = phi @ w 
            probs = self.softmax(logits) 
            class_values = torch.arange(self.num_classes, dtype=torch.float32)
            reward = torch.sum(class_values * probs)
            return reward.item()

    def compute_policy_gradient(self, theta, trajectories, rewards):
        grad = np.zeros_like(theta)
        for traj, reward in zip(trajectories, rewards):
            traj = traj[0]
            for state, action in traj[:-1]:
                state_index = self.env.get_state_index(state)
                action_probs = self.softmax_policy(theta[state_index])
                grad[state_index, action] += reward * (1 - action_probs[action])
                for a in range(self.num_actions):
                    if a != action:
                        grad[state_index, a] -= reward * action_probs[a]
        return grad

    def true_feedback_prob(self, trajectory):
        with torch.no_grad():
            phi = self.compute_feature_matrix(trajectory)
            logits = phi @ self.omega_star 
            probs = self.softmax(logits)
            return probs

    def compute_true_reward(self, trajectory):
        with torch.no_grad():
            phi = self.compute_feature_matrix(trajectory)  
            logits = phi @ self.omega_star 
            probs = self.softmax(logits) 
            class_values = torch.arange(self.num_classes, dtype=torch.float32)
            reward = torch.sum(class_values * probs)
            return reward.item()
        
    def project(self, w):
        B = self.hyper_params["B"]
        with torch.no_grad():
            w_norm = torch.norm(w,p=2) 
            if w_norm > B:
                w.copy_(w * B/w_norm)

    def loss_function(self, trajectories, feedbacks, w_estimate):
        loss = 0
        indices = feedbacks.clone()
        indices = indices.unsqueeze(1).unsqueeze(2)
        indices = indices.expand(-1, -1, trajectories.shape[2])
        feature_true_list = torch.gather(trajectories, 1, indices)
        feature_true_list = feature_true_list.squeeze(1)
        loss = feature_true_list @ w_estimate - torch.log(torch.exp(trajectories @ w_estimate).sum(dim=1))
        # with torch.no_grad():
        #     prediciton = torch.sum(torch.argmax(trajectories @ w_estimate, dim=1) == feedbacks)/len(feedbacks)
        #     print(prediciton)
        return -loss.mean()


    def inner_loop_pgd(self, trajectories, feedbacks):
        self.w_estimate = nn.Parameter(torch.zeros(self.num_classes * self.feature_dim, dtype=torch.float32, requires_grad=True))
        nn.init.normal_(self.w_estimate)
        max_iterations = self.hyper_params["inner_loop_gd_max_iterations"]
        
        self.project(self.w_estimate)
        optimizer = torch.optim.SGD([self.w_estimate], lr=self.hyper_params["learning_rate"])
        counter = 0

        for i in range(max_iterations):
            with torch.no_grad():
                w_old = self.w_estimate.detach().clone()
            
            optimizer.zero_grad()
            loss = self.loss_function(trajectories, feedbacks, self.w_estimate) + self.hyper_params["regularization"] * torch.norm(self.w_estimate)
            loss.backward()
            optimizer.step()
            self.project(self.w_estimate)
            
            with torch.no_grad():
                if torch.norm(self.w_estimate-w_old, 2) < self.hyper_params["tolerance"]:
                    counter += 1
                else:
                    counter = 0
                if counter == 5:
                    break


    ##########  w update at intervals to speed up run  ###########

    def run_full_algorithm(self, num_runs):
        T = self.hyper_params["T"]
        reward_all_runs = np.zeros((num_runs, T))
        # est_errors_all_runs = np.zeros((num_runs, T))
        self.rng = np.random.default_rng(seed=self.hyper_params["seed"])
        torch.manual_seed(self.hyper_params["seed"])
        self.env = GridEnv(self.hyper_params["grid_size"], self.hyper_params["coins_count"], self.hyper_params["horizon"])
        self.env.reset(seed=self.hyper_params["env_seed"])
        self.num_actions = self.env.action_space.n
        self.feature_dim = self.env.feature_dim
        self.num_classes = self.env.reward_classes

        for run in range(num_runs):
            print(f'Run {run+1}/{num_runs}')
            theta = torch.as_tensor(self.rng.random((self.env.num_states, self.num_actions)), dtype=torch.float64)
            all_trajectories = torch.zeros(self.hyper_params["T"], self.num_classes, self.num_classes * self.feature_dim, dtype=torch.float32)
            all_feedbacks = torch.zeros(self.hyper_params["T"], dtype=torch.int64)
            self.w_estimate = torch.zeros(self.num_classes * self.feature_dim, dtype=torch.float32)
            
            for t in range(T):
                if t % self.hyper_params["print_frequency"] == 0:
                    print(f'Episode {t}/{T}')
                if self.hyper_params["train_policy"]:
                    while True:
                        prev_theta = np.copy(theta)
                        trajectories = [self.generate_trajectory(theta) for _ in range(200)]
                        rewards = [self.compute_estimated_reward(self.w_estimate, traj) for traj in trajectories]

                        grad = self.compute_policy_gradient(theta, trajectories, rewards)
                        theta += self.hyper_params["alpha"] * grad
                        
                        if np.linalg.norm(theta - prev_theta) <= self.hyper_params["epsilon"]:
                            # print(np.linalg.norm(theta - prev_theta))
                            break

                trajectory = self.generate_trajectory(theta)
                if self.hyper_params["use_env_reward"]:
                    feedback = trajectory[2]
                else:
                    feedback_probs = self.true_feedback_prob(trajectory)
                    feedback = torch.tensor(self.rng.choice(self.num_classes, p=feedback_probs.numpy()))

                all_trajectories[t, :] = self.compute_feature_matrix(trajectory)
                all_feedbacks[t] = feedback

                if (t + 1) % self.hyper_params["update_interval"] == 0:
                    self.inner_loop_pgd(all_trajectories[:t+1, :], all_feedbacks[:t+1])
                    # print(f'w estimate at episode {t+1}', w_estimate)
                    
                    # error = estimation_error(w_estimate, omega_star)
                    # est_errors_all_runs[run, t] = error

                if self.hyper_params["train_policy"]:
                    eval_trajectories = [self.generate_trajectory(theta) for _ in range(200)]
                    true_rewards = [self.compute_true_reward(traj) for traj in eval_trajectories]
                    reward_all_runs[run, t] = np.mean(true_rewards)

        return reward_all_runs
    


if __name__ == "__main__":
    hyper_params = {}
    hyper_params["grid_size"] = 4
    hyper_params["coins_count"] = 1
    hyper_params["horizon"] = 25
    hyper_params["tolerance"] = 1e-7
    hyper_params["seed"] = 1235
    hyper_params["env_seed"] = hyper_params["seed"]
    hyper_params["learning_rate"] = 1
    hyper_params["regularization"] = 0.01
    hyper_params["epsilon"] = 0.2
    hyper_params["alpha"] = 0.001
    hyper_params["B"] = 1000

    hyper_params["T"] = 2000
    hyper_params["update_interval"]= 100
    hyper_params["uniform_policy"] = False
    hyper_params["use_env_reward"] = False
    hyper_params["train_policy"] = True

    hyper_params["inner_loop_gd_max_iterations"] = 10000
    hyper_params["print_frequency"] = 100
    # omega_star = torch.as_tensor(np.array([-3.7038, -5.4144, -1.6567,  6.7748, -7.1607, 17.8211, -3.2246, -1.6937,
    #     15.6044, -7.9830, -1.5229, -3.2256, -4.7237, -4.4601,  6.4348, -1.8263]), dtype=torch.float32)
    


    # K = 4, coin = 1 (new)
    omega_star = torch.as_tensor(np.array([-1.9108, -3.7382, -1.9097,  6.5837, -0.6338,  5.3681,  7.1738,  6.0408,
        -1.5342, -8.6076, -0.1140,  0.0766, -3.8470, -2.7990,  8.0839, -3.3415,
        -3.5125, -0.2810, -2.2345,  1.1508]), dtype=torch.float32)

    
    num_runs = 2
    trainer = Trainer(hyper_params, omega_star)
    reward_all_runs = trainer.run_full_algorithm(num_runs)
    print(trainer.w_estimate)
    
    
    avg_rewards = np.mean(reward_all_runs, axis=0)
    std_rewards = np.std(reward_all_runs, axis=0)
    
    plt.figure()
    plt.plot(range(hyper_params["T"]), avg_rewards, label='Average True Reward')
    plt.fill_between(range(hyper_params["T"]),
                     avg_rewards - 2 * std_rewards,
                     avg_rewards + 2 * std_rewards,
                     color='b', alpha=0.2, label="95% Confidence Interval")
    plt.xlabel('Episodes')
    plt.ylabel('Average True Reward')
    plt.title('Average Reward vs. Episodes')
    plt.legend()
    plt.show()