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
from CybORG.Shared.Actions import Analyse
from Wrappers.BlueTableWrapper import BlueTableWrapper 
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2


MAX_EPS = 2
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

# Map SHAP values to observation
def map_shap_to_observation(shap_values, observation, feature_names):
    mapped_features = {}
    for idx, value in enumerate(shap_values[0][:len(feature_names)]):
        # Get the feature name and corresponding value in the raw observation
        feature_name = feature_names[idx]
        if '_Activity' in feature_name:
            # Map to the actual activity feature in the raw observation
            host = feature_name.split('_')[0]
            activity = observation.get(host, {}).get('Processes', 'Unknown')
            mapped_features[feature_name] = {'SHAP Value': value, 'Observed Activity': activity}
        elif '_Compromised' in feature_name:
            # Map to the compromised status in the raw observation
            host = feature_name.split('_')[0]
            compromised_status = observation.get(host, {}).get('Sessions', 'Unknown')
            mapped_features[feature_name] = {'SHAP Value': value, 'Compromised Status': compromised_status}
    return mapped_features

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
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    env = CybORG(path, 'sim')
    results = env.reset(agent='Blue')
    obs = results.observation

    


    action_id_to_name = {
    1: "Monitor",
    2: "Analyse Defender",
    3: "Analyse Enterprise0",
    4: "Analyse Enterprise1",
    5: "Analyse Enterprise2",
    6: "Analyse Op_Host0",
    7: "Analyse Op_Host1",
    8: "Analyse Op_Host2",
    9: "Analyse Op_Server0",
    10: "Analyse User0",
    11: "Analyse User1",
    12: "Analyse User2",
    13: "Analyse User3",
    14: "Analyse User4",
    15: "Remove Defender",
    16: "Remove Enterprise0",
    17: "Remove Enterprise1",
    18: "Remove Enterprise2",
    19: "Remove Op_Host0",
    20: "Remove Op_Host1",
    21: "Remove Op_Host2",
    22: "Remove Op_Server0",
    23: "Remove User0",
    24: "Remove User1",
    25: "Remove User2",
    26: "Remove User3",
    27: "Remove User4",
    28: "DecoyApache Defender",
    29: "DecoyApache Enterprise0",
    30: "DecoyApache Enterprise1",
    31: "DecoyApache Enterprise2",
    32: "DecoyApache Op_Host0",
    33: "DecoyApache Op_Host1",
    34: "DecoyApache Op_Host2",
    35: "DecoyApache Op_Server0",
    36: "DecoyApache User0",
    37: "DecoyApache User1",
    38: "DecoyApache User2",
    39: "DecoyApache User3",
    40: "DecoyApache User4",
    41: "DecoyFemitter Defender",
    42: "DecoyFemitter Enterprise0",
    43: "DecoyFemitter Enterprise1",
    44: "DecoyFemitter Enterprise2",
    45: "DecoyFemitter Op_Host0",
    46: "DecoyFemitter Op_Host1",
    47: "DecoyFemitter Op_Host2",
    48: "DecoyFemitter Op_Server0",
    49: "DecoyFemitter User0",
    50: "DecoyFemitter User1",
    51: "DecoyFemitter User2",
    52: "DecoyFemitter User3",
    53: "DecoyFemitter User4",
    54: "DecoyHarakaSMPT Defender",
    55: "DecoyHarakaSMPT Enterprise0",
    56: "DecoyHarakaSMPT Enterprise1",
    57: "DecoyHarakaSMPT Enterprise2",
    58: "DecoyHarakaSMPT Op_Host0",
    59: "DecoyHarakaSMPT Op_Host1",
    60: "DecoyHarakaSMPT Op_Host2",
    61: "DecoyHarakaSMPT Op_Server0",
    62: "DecoyHarakaSMPT User0",
    63: "DecoyHarakaSMPT User1",
    64: "DecoyHarakaSMPT User2",
    65: "DecoyHarakaSMPT User3",
    66: "DecoyHarakaSMPT User4",
    67: "DecoySmss Defender",
    68: "DecoySmss Enterprise0",
    69: "DecoySmss Enterprise1",
    70: "DecoySmss Enterprise2",
    71: "DecoySmss Op_Host0",
    72: "DecoySmss Op_Host1",
    73: "DecoySmss Op_Host2",
    74: "DecoySmss Op_Server0",
    75: "DecoySmss User0",
    76: "DecoySmss User1",
    77: "DecoySmss User2",
    78: "DecoySmss User3",
    79: "DecoySmss User4",
    80: "DecoySSHD Defender",
    81: "DecoySSHD Enterprise0",
    82: "DecoySSHD Enterprise1",
    83: "DecoySSHD Enterprise2",
    84: "DecoySSHD Op_Host0",
    85: "DecoySSHD Op_Host1",
    86: "DecoySSHD Op_Host2",
    87: "DecoySSHD Op_Server0",
    88: "DecoySSHD User0",
    89: "DecoySSHD User1",
    90: "DecoySSHD User2",
    91: "DecoySSHD User3",
    92: "DecoySSHD User4",
    93: "DecoySvchost Defender",
    94: "DecoySvchost Enterprise0",
    95: "DecoySvchost Enterprise1",
    96: "DecoySvchost Enterprise2",
    97: "DecoySvchost Op_Host0",
    98: "DecoySvchost Op_Host1",
    99: "DecoySvchost Op_Host2",
    100: "DecoySvchost Op_Server0",
    101: "DecoySvchost User0",
    102: "DecoySvchost User1",
    103: "DecoySvchost User2",
    104: "DecoySvchost User3",
    105: "DecoySvchost User4",
    106: "DecoyTomcat Defender",
    107: "DecoyTomcat Enterprise0",
    108: "DecoyTomcat Enterprise1",
    109: "DecoyTomcat Enterprise2",
    110: "DecoyTomcat Op_Host0",
    111: "DecoyTomcat Op_Host1",
    112: "DecoyTomcat Op_Host2",
    113: "DecoyTomcat Op_Server0",
    114: "DecoyTomcat User0",
    115: "DecoyTomcat User1",
    116: "DecoyTomcat User2",
    117: "DecoyTomcat User3",
    118: "DecoyTomcat User4",
    119: "DecoyVsftpd Defender",
    120: "DecoyVsftpd Enterprise0",
    121: "DecoyVsftpd Enterprise1",
    122: "DecoyVsftpd Enterprise2",
    123: "DecoyVsftpd Op_Host0",
    124: "DecoyVsftpd Op_Host1",
    125: "DecoyVsftpd Op_Host2",
    126: "DecoyVsftpd Op_Server0",
    127: "DecoyVsftpd User0",
    128: "DecoyVsftpd User1",
    129: "DecoyVsftpd User2",
    130: "DecoyVsftpd User3",
    131: "DecoyVsftpd User4",
    132: "Restore Defender",
    133: "Restore Enterprise0",
    134: "Restore Enterprise1",
    135: "Restore Enterprise2",
    136: "Restore Op_Host0",
    137: "Restore Op_Host1",
    138: "Restore Op_Host2",
    139: "Restore Op_Server0",
    140: "Restore User0",
    141: "Restore User1",
    142: "Restore User2",
    143: "Restore User3",
    144: "Restore User4"
}
    



    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in [5]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg)
            observation = wrapped_cyborg.reset()
            # Ensure observation is 2D, this is a wrapped actions so I am assumign its coming from the one that we defined and not defualt action space 
            action_space = wrapped_cyborg.get_action_space(agent_name)
            #action_space = cyborg.get_action_space(agent_name)
            print (action_space)
     
            # Define SHAP explainer with the custom predict function
            # Ensure the observation is reshaped correctly
            observation_flat = observation.reshape(1, -1)
   
            # Now, create the background data based on the actual observation size (which should be 27 features)
            background_data = np.random.randn(100, observation_flat.shape[1])  # 100 samples, with 27 features from the observation
            print(background_data.shape)

            # Run custom predict and print output
            prediction_output = custom_predict(observation_flat)
            #print("This is logic", prediction_output)
            
            feature_names = extract_feature_names(obs)

            explainer = shap.KernelExplainer(custom_predict, background_data)

            total_reward = []
            actions = []
            shap_values_all = []


            for i in range(MAX_EPS):
                r = []
                a = []
                for j in range(num_steps):
                    observation_flat = observation.flatten()
                    action = agent.get_action(observation, action_space)
                    observation, rew, done, info = wrapped_cyborg.step(action)
                    action_name = action_id_to_name.get(action, "Unknown Action")

                    # SHAP: Generate explanation for each action (vectorized observation)
                    shap_values = explainer.shap_values(observation_flat)
                    shap_values_all.append(shap_values[0])  # Save SHAP values for this observation
                    
                    # Identify the most influential feature for this action
                    max_influence_feature = None
                    max_shap_value = None

                    for feature, shap_value in zip(feature_names, shap_values[0]):
                        if shap_value > 0 and (max_shap_value is None or shap_value > max_shap_value):
                            max_shap_value = shap_value
                            max_influence_feature = feature


                    # Print action and explanation
                    print(f"\n=== Action taken: Action ID {action}, Action Name: {action_name} ===")
                    if max_influence_feature:
                        print(f"Most influential feature (positive impact): {max_influence_feature}, SHAP Value: {max_shap_value}")
                        print(f"Explanation: The agent took the action '{action_name}' primarily because the '{max_influence_feature}' feature was most influential, indicating that this feature had the most positive impact on the agent's decision.\n")
                    else:
                        print("No positive SHAP values found for this action.\n")

                
                    r.append(rew)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

                agent.end_episode()
                total_reward.append(sum(r))
                actions.append(a)
                observation = wrapped_cyborg.reset()


            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')



# Generate SHAP summary plot
#shap.summary_plot(shap_values_all, background_data)
shap.summary_plot(np.array(shap_values_all), feature_names=feature_names)


