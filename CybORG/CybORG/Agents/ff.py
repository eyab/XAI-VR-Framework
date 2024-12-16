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

MAX_EPS = 100 
agent_name = 'Blue'
random.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Custom predict function for SHAP
def custom_predict(observation):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Make sure the observation is a 2D array
    if len(observation.shape) == 1:
        observation_copy = observation.reshape(1, -1)
    else:
        observation_copy = observation.copy()

    # Ensure the observation is padded to match the network's expected input size (e.g., 62)
    if observation_copy.shape[1] == 52:  # Example of the input size expected by the policy
        observation_copy = np.pad(observation_copy, ((0, 0), (0, 10)), mode='constant')

    # Convert the padded observation to a PyTorch tensor
    observation_tensor = torch.FloatTensor(observation_copy).to(agent.device)
    
    # Get action logits from the policy's actor network
    with torch.no_grad():
        logits = agent.policy.actor(observation_tensor)

    # Return the logits as a NumPy array
    return logits.cpu().numpy()


# Wrapper for the environment
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name='Blue')

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    commit_hash = "Not using git"
    name = "John Hannay"
    team = "CardiffUni"
    name_of_agent = "PPO + Greedy decoys"

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # Initialize the agent
    agent = MainAgent()

    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in [30, 50, 100]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg)

            observation = wrapped_cyborg.reset()

            # Ensure observation is 2D

            action_space = wrapped_cyborg.get_action_space(agent_name)
            
            # Define SHAP explainer with the custom predict function
            background_data = np.random.randn(100, observation.reshape(1, -1).shape[1])  # Adjust to match the observation shape
            #print(f"Background Data: {background_data}")

            # Run custom predict and print output
            prediction_output = custom_predict(observation)
            #print(f"Custom Predict Output: {prediction_output}")

            explainer = shap.KernelExplainer(custom_predict, background_data)
            #print(f"SHAP Explainer: {explainer}")

            total_reward = []
            actions = []
            
            for i in range(MAX_EPS):
                r = []
                a = []

                for j in range(num_steps):
                    #print(observation)
                    #print(action_space)
                    observation = observation.flatten()
                   
                    action = agent.get_action(observation, action_space)

                    observation, rew, done, info = wrapped_cyborg.step(action)

                    # Ensure observation is 2D after each step
                    observation = observation.reshape(1, -1)  # Ensure observation is 2D
                    #print(f"Obs: {observation}")

                    path = str(inspect.getfile(CybORG))
                    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'

                    env = CybORG(path, 'sim')

                    results = env.reset(agent='Blue')
                    obs = results.observation
                    #print("new obs ", obs)




                    # SHAP: Only run explanation every 10 episodes
                    #if i % 100 == 0:
                        #shap_values_all = explainer.shap_values(background_data)

                        # Extract the host names from the Blue agent's observation
                        blue_obs = wrapped_cyborg.get_observation('Blue')
                        host_names = [key for key in blue_obs.keys() if key != 'success']  # Exclude 'success'

                        # Create feature names for SHAP using actual host names
                        feature_names = []
                        for host in host_names:
                            feature_names.extend([f"{host}_Activity", f"{host}_Compromised"])
                        print(feature_names)

                        # Use the feature names in SHAP summary plot
                        #shap.summary_plot(shap_values_all, background_data, feature_names=feature_names)

                                               
                        


                        # Now plot SHAP summary
                        #shap.summary_plot(shap_values_all, background_data)


                                                                        



                    r.append(rew)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

                agent.end_episode()
                total_reward.append(sum(r))
                actions.append(a)
                observation = wrapped_cyborg.reset()

            # SHAP Summary plot
        

            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')
    
    #shap.summary_plot(shap_values_all, background_data)


I also noticed you mentioned a group meeting on Thursday, and I’d be happy to attend if that works for you and would be helpful.










###################
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

MAX_EPS = 100 
agent_name = 'Blue'
random.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Custom predict function for SHAP
def custom_predict(observation):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Make sure the observation is a 2D array
    if len(observation.shape) == 1:
        observation_copy = observation.reshape(1, -1)
    else:
        observation_copy = observation.copy()

    # Ensure the observation is padded to match the network's expected input size (e.g., 62)
    if observation_copy.shape[1] == 52:  # Example of the input size expected by the policy
        observation_copy = np.pad(observation_copy, ((0, 0), (0, 10)), mode='constant')

    # Convert the padded observation to a PyTorch tensor
    observation_tensor = torch.FloatTensor(observation_copy).to(agent.device)
    
    # Get action logits from the policy's actor network
    with torch.no_grad():
        logits = agent.policy.actor(observation_tensor)

    # Return the logits as a NumPy array
    return logits.cpu().numpy()


# Wrapper for the environment
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name='Blue')

# Function to extract feature names dynamically from the entire observation
def extract_feature_names(observation):
    feature_names = []
    for host, data in observation.items():
        if host != 'success':  # Ignore the success key
            feature_names.append(f'{host}_Activity')
            feature_names.append(f'{host}_Compromised')
    return feature_names

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    commit_hash = "Not using git"
    name = "John Hannay"
    team = "CardiffUni"
    name_of_agent = "PPO + Greedy decoys"

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # Initialize the agent
    agent = MainAgent()

    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    env = CybORG(path, 'sim')
    results = env.reset(agent='Blue')
    obs = results.observation

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in [30, 50, 100]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg)

            observation = wrapped_cyborg.reset()


            # Ensure observation is 2D

            action_space = wrapped_cyborg.get_action_space(agent_name)
            
            # Define SHAP explainer with the custom predict function
            # Ensure the observation is reshaped correctly
            observation = observation.reshape(1, -1)
   

            # Now, create the background data based on the actual observation size (which should be 27 features)
            background_data = np.random.randn(100, observation.shape[1])  # 100 samples, with 27 features from the observation
            #background_data = np.random.randn(100, observation.reshape(1, -1).shape[1])  # Adjust to match the observation shape

            # Run custom predict and print output
            prediction_output = custom_predict(observation)
            
            feature_names = extract_feature_names(obs)

            explainer = shap.KernelExplainer(custom_predict, background_data)

            total_reward = []
            actions = []
            
            for i in range(MAX_EPS):
                r = []
                a = []

                for j in range(num_steps):
                    observation = observation.flatten()
                    action = agent.get_action(observation, action_space)

                    observation, rew, done, info = wrapped_cyborg.step(action)
                    print(observation)

                    # Ensure observation is 2D after each step
                    observation = observation.reshape(1, -1)  # Ensure observation is 2D
                    


                    # SHAP: Only run explanation every 10 episodes
                    if i % 100 == 0:
                        print(f"Obs after reshaping: {observation}")
                        shap_values_all = explainer.shap_values(background_data)

                        print(f"Shape of shap_values_all: {np.array(shap_values_all).shape}")
                        print(shap_values_all)
                        print(f"Shape of background_data: {np.array(background_data).shape}")
                        print(background_data)
                        print(f"Number of feature names: {len(feature_names)}")
                        print(feature_names)

                        
                        # Ensure SHAP summary plot uses the correct feature names
                        shap.summary_plot(shap_values_all, background_data)

                    r.append(rew)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

                agent.end_episode()
                total_reward.append(sum(r))
                actions.append(a)
                observation = wrapped_cyborg.reset()

            # SHAP Summary plot
            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')

