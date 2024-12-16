import inspect
from stable_baselines3 import PPO
from CybORG import CybORG
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.Wrappers import ChallengeWrapper, BlueTableWrapper, EnumActionWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import torch
import numpy as np
from CybORG.Agents import RedMeanderAgent, B_lineAgent
import shap
from pprint import pprint
import subprocess
import inspect
import time
from statistics import mean, stdev
import shap
from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from CybORG.Agents.MainAgent import MainAgent
import random
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pprint import pprint


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class eyasAgent(BaseAgent):
    def __init__(self, model_file: str = None):
        if model_file is not None:
            self.model = PPO.load(model_file)
        else:
            self.model = None

    def train(self, timesteps=100000):
        path = str(inspect.getfile(CybORG))
        path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
        cyborg = ChallengeWrapper(env=CybORG(path, 'sim'), agent_name='Blue')

        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                                 name_prefix='cyborg_blue')
        eval_callback = EvalCallback(cyborg, best_model_save_path='./logs/',
                                     log_path='./logs/', eval_freq=5000,
                                     deterministic=True, render=False)

        self.model = PPO('MlpPolicy', cyborg, verbose=1)
        self.model.learn(total_timesteps=timesteps, callback=[checkpoint_callback, eval_callback])

    def get_action(self, observation, action_space):
        if self.model is None:
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
            cyborg = ChallengeWrapper(env=CybORG(path, 'sim'), agent_name='Blue')
            self.model = PPO('MlpPolicy', cyborg)

        # Ensure the observation is a numpy array and has the correct shape
        observation = np.array(observation).reshape(1, -1)

        action, _states = self.model.predict(observation)
        return int(action)  # Ensure action is an integer

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass
    # Wrapper for the environment
    

if __name__ == "__main__":
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    agent = eyasAgent()
    agent.train(timesteps=500)  # Increase timesteps for more effective training

    CYBORG = CybORG(path, 'sim', agents={'Red': B_lineAgent})
    env = BlueTableWrapper(EnumActionWrapper(CYBORG), output_mode='vector')

    results = env.reset(agent='Blue')
    blue_obs = results.observation
    #print(blue_obs, "Tabel")
    def wrap(env):
        return ChallengeWrapper2(env=env, agent_name='Blue')

    for i in range(10):  # Run more steps for better analysis
        blue_obs = np.array(blue_obs).reshape((1, -1))
        action = agent.get_action(blue_obs, env.get_action_space(agent='Blue'))
        results = env.step(agent='Blue', action=action)
        blue_obs = results.observation
        print(f"Step {i+1}:")
        print(f"Observation: {blue_obs}")
        print(f"Action: {action}")
        print(f"Reward: {results.reward}")
        print(f"Done: {results.done}")
        print(f"Info: {results.info}")
        print(76 * '-')
        #print("shape",blue_obs.shape)
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

    env = CybORG(path, 'sim')
    blue_obs = env.get_observation('Blue')

    for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:

        cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
        wrapped_cyborg = wrap(cyborg)

        action_space = wrapped_cyborg.get_action_space(agent)
        print(action_space)

    pprint(blue_obs['User4'])



    # Custom predict function to return a consistent format
    def custom_predict(obs):
        obs = obs.reshape(1, -1)  # Reshape the observation to 2D
        action, _ = agent.model.predict(obs)  # Get the action from the model
        action = int(action)  # Ensure that the action is a single integer value
        return np.array([action])  # Return as a 2D array for SHAP

"""
    custom_predict(blue_obs)

    # Use the custom predict function to create a SHAP explainer
    explainer = shap.KernelExplainer(custom_predict, blue_obs.reshape(1, -1))

    # Calculate SHAP values for the observation
    shap_values = explainer(blue_obs.reshape(1, -1))
    print(shap_values)
    shap.summary_plot(shap_values.values, blue_obs)

    # Example for a dependence plot for the first feature
    shap.dependence_plot(0, shap_values.values, blue_obs) 
       """

