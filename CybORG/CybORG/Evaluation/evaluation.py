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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


MAX_EPS = 2
agent_name = 'Blue'
random.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def custom_predict(observation):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(agent.policy.actor)

    # Make sure the observation is a 2D array
    if len(observation.shape) == 1:
        observation_copy = observation.reshape(1, -1)
    else:
        observation_copy = observation.copy()

    # Ensure the observation matches the input size (62)
    if observation_copy.shape[1] > 62:  # Truncate if too many features
        observation_copy = observation_copy[:, :62]
    elif observation_copy.shape[1] < 62:  # Pad with zeros if too few features
        observation_copy = np.pad(observation_copy, ((0, 0), (0, 62 - observation_copy.shape[1])), mode='constant')

    # Convert to PyTorch tensor
    observation_tensor = torch.FloatTensor(observation_copy).to(agent.device)

    # Get action logits from the policy's actor network
    with torch.no_grad():
        logits = agent.policy.actor(observation_tensor)

    # Return the logits as a NumPy array
    return logits.cpu().numpy()




    # Return the logits as a NumPy array
    return logits.cpu().numpy()

# Wrapper for the environment
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name='Blue')


def extract_feature_names_and_values(observation):
    """
    Extract feature names and numeric values dynamically from the observation dictionary.
    """
    feature_names = []
    flattened_values = []

    def traverse_dict(prefix, obj):
        # Handle dictionaries
        if isinstance(obj, dict):
            for key, value in obj.items():
                traverse_dict(f"{prefix}_{key}" if prefix else key, value)
        # Handle lists
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                traverse_dict(f"{prefix}_{idx}", item)
        # Handle primitive values
        else:
            # Only include numeric values
            if isinstance(obj, (int, float, bool)):
                feature_names.append(prefix)
                flattened_values.append(float(obj))
            else:
                # Skip non-numeric values
                pass

    for key, value in observation.items():
        if key == 'success':  # Skip the success field
            continue
        traverse_dict(key, value)

    return feature_names, flattened_values




def map_shap_to_observation(shap_values, feature_names, flattened_values):
    """
    Map SHAP values to the extracted feature names and their corresponding observation values.
    """
    mapped_features = {}
    for idx, value in enumerate(shap_values[0][:len(feature_names)]):
        feature_name = feature_names[idx]
        observed_value = flattened_values[idx]
        mapped_features[feature_name] = {'SHAP Value': value, 'Observed Value': observed_value}
    return mapped_features

def extract_issues_from_observation(observation):
    """
    Enhanced function to extract issues from the observation with expanded checks for session types, 
    privileged users, and other potential anomalies.
    """

    print(observation)

    issues = []

    def traverse(prefix, obj):
        # Handle dictionaries
        if isinstance(obj, dict):
            for key, value in obj.items():
                traverse(f"{prefix}_{key}" if prefix else key, value)
        # Handle lists
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                traverse(f"{prefix}_{idx}", item)
        # Handle values
        else:
            # File-specific issues
            if prefix.endswith("Density") and isinstance(obj, (int, float)):
                if obj >= 0.8:  # High density threshold for malware
                    issues.append({
                        'type': 'File Issue',
                        'description': f"High density ({obj}) indicating potential malware. High density files often resemble malicious patterns.",
                        'key': prefix
                    })
            elif prefix.endswith("Signed") and obj is False:
                issues.append({
                    'type': 'File Issue',
                    'description': "Unsigned file detected. Unsigned files lack validation and may be compromised.",
                    'key': prefix
                })
            # Session-specific issues
            elif prefix.endswith("Username") and obj in ["SYSTEM", "Administrator"]:
                issues.append({
                    'type': 'Session Issue - Privileged User',
                    'description': f"Privileged user ({obj}) activity detected. Privileged accounts are high-value targets for attackers.",
                    'key': prefix
                })
            elif prefix.endswith("Timeout") and isinstance(obj, int) and obj == 0:
                issues.append({
                    'type': 'Session Issue - Timeout',
                    'description': "Session timeout detected. Inactive sessions could lead to reduced visibility.",
                    'key': prefix
                })
            elif prefix.endswith("Type"):
                if "VELOCIRAPTOR" in str(obj):
                    issues.append({
                        'type': 'Session Issue - Type',
                        'description': f"Unexpected session type detected: {obj}.",
                        'key': prefix
                    })
            # Connection-specific issues
            elif prefix.endswith("Connections") and isinstance(obj, list):
                if len(obj) > 5:  # Example: flagging if too many connections
                    issues.append({
                        'type': 'Connection Issue',
                        'description': f"High number of connections ({len(obj)}). May indicate unexpected or malicious communication.",
                        'key': prefix,
                        'value': obj
                    })
            # User Group-specific issues
            elif prefix.endswith("GID") and obj == 0:
                issues.append({
                    'type': 'User Group Issue',
                    'description': "User belongs to root group (GID 0), which has high privileges.",
                    'key': prefix
                })

    traverse("", observation)
    return issues

def get_explanation_type():
    """
    Prompt the user to select the type of explanation: User or System Designer.
    """
    while True:
        print("\nWho are you generating this report for?")
        print("1. End User")
        print("2. System Designer")
        choice = input("Enter 1 or 2: ")
        if choice == "1":
            return "user"
        elif choice == "2":
            return "designer"
        else:
            print("Invalid choice. Please enter 1 or 2.")




def log_shap_explanation(red_action, blue_action, shap_values, mapped_shap_values, issues, explanation_type):
    """
    Logs SHAP-based explanations tailored for users or system designers.

    :param red_action: The action taken by the Red agent.
    :param blue_action: The action taken by the Blue agent.
    :param shap_values: SHAP values explaining the decision.
    :param mapped_shap_values: Mapped SHAP values to features.
    :param issues: Extracted issues/anomalies from observations.
    :param explanation_type: 'user' for end users, 'designer' for system designers.
    """
    print("\n" + "=" * 90)
    print("\n=== Actions Taken ===")
    print(f"- Blue Agent Action: {blue_action}")
    print(f"- Red Agent Action: {red_action}")
    print("-" * 50)

    if explanation_type == "designer":
        print("\n=== Explanation for System Designer ===")
        print(
            "The Blue agent analyzed the current environment and performed the selected action. "
            "This decision was influenced by the environment's features, which were evaluated "
            "using SHAP (SHapley Additive exPlanations) to quantify their contributions. Below is a detailed explanation."
        )

        if shap_values is not None and shap_values.any():
            try:
                # Filter for relevant features based on the current action or context
                relevant_features = {
                    k: v for k, v in mapped_shap_values.items() if blue_action.split()[1] in k
                }
                if not relevant_features:
                    relevant_features = mapped_shap_values  # Fallback to all features if none match

                # Determine the most influential feature
                max_influence_feature = max(
                    relevant_features,
                    key=lambda k: abs(np.mean(relevant_features[k]['SHAP Value']))
                )
                shap_value = np.mean(relevant_features[max_influence_feature]['SHAP Value'])
                observed_value = relevant_features[max_influence_feature]['Observed Value']

                print("\nMost Influential Feature:")
                print(f"- Feature Name: {max_influence_feature}")
                print(f"- SHAP Value: {shap_value:.4f}")
                print(f"- Observed Value: {observed_value}")
                print("Explanation:")

                # Contextual reasoning based on feature patterns
                if "Sessions" in max_influence_feature:
                    print(
                        f"The feature '{max_influence_feature}' represents session-specific activity. "
                        f"Observed value '{observed_value}' suggests active sessions on {max_influence_feature.split('_')[0]}, "
                        "which could indicate potential exploitation or lateral movement attempts by the adversary."
                    )
                elif "Processes" in max_influence_feature:
                    print(
                        f"The feature '{max_influence_feature}' relates to process activity. "
                        f"Observed value '{observed_value}' indicates a potentially critical or high-privilege process "
                        "that might be a target for compromise or misuse."
                    )
                elif "User Info" in max_influence_feature:
                    print(
                        f"The feature '{max_influence_feature}' pertains to user privileges or roles. "
                        f"Observed value '{observed_value}' highlights potential privilege escalation or unauthorized access risks."
                    )
                elif "Services" in max_influence_feature:
                    print(
                        f"The feature '{max_influence_feature}' refers to network services. "
                        f"Observed value '{observed_value}' indicates a service that may have been targeted for exploitation."
                    )
                else:
                    print(
                        f"The feature '{max_influence_feature}' contributes significantly to the agent's decision, "
                        "but its direct context needs further analysis."
                    )

                # Tie the feature to the Blue agent's action
                print(
                    f"The Blue agent's action, '{blue_action}', was aimed at mitigating this potential threat. "
                    f"By performing '{blue_action.split()[0]}', the agent sought to address vulnerabilities or disrupt adversary actions."
                )

                # Include Red agent's action as context
                if red_action:
                    print(
                        f"This decision also considered the adversary's recent action, '{red_action}', "
                        "which may have influenced the observed behavior."
                    )

            except Exception as e:
                print(f"Error determining most critical feature: {e}")
        else:
            print("No significant factors were identified in this decision.")

    print("-" * 50)


















def summarize_results(shap_values_all, feature_names):
    """
    Summarize the most influential features across all episodes.
    """
    feature_influence = {}
    for shap_values in shap_values_all:
        for shap_value, feature in zip(shap_values, feature_names):
            # Aggregate SHAP values if they are arrays
            if isinstance(shap_value, np.ndarray):
                shap_value = np.abs(shap_value).sum()  # Use sum of absolute values
            if feature not in feature_influence:
                feature_influence[feature] = 0
            feature_influence[feature] += abs(shap_value)

    # Sort features by their total influence
    sorted_influence = sorted(feature_influence.items(), key=lambda x: x[1], reverse=True)

    print("\n=== Top Influential Features Across All Episodes ===")
    for feature, total_influence in sorted_influence[:5]:
        print(f"{feature}: {total_influence:.5f}")










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

    # Initialize BlueTableWrapper (anomaly detection wrapper)
    blue_wrapper = BlueTableWrapper(env=env, agent=agent)
    blue_wrapper.reset(agent='Blue')  # Setting the baseline here to avoid any issues

 

    #Prompt for explanation type once at the beginning
    explanation_type = get_explanation_type()
    print(f"Generating explanations for: {'End User' if explanation_type == 'user' else 'System Designer'}")

   
    action_id_to_name = {
    0: "Sleep",
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
    
    for num_steps in [30]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg)
            observation = wrapped_cyborg.reset()

            action_space = wrapped_cyborg.get_action_space(agent_name)
            #print(action_space)

            # Ensure observation is reshaped for SHAP processing
            observation_flat = observation.reshape(1, -1)
            background_data = np.random.randn(100, observation_flat.shape[1])
            #print(background_data.shape)

            


            explainer = shap.KernelExplainer(custom_predict, background_data)

            total_reward = []
            actions = []
            shap_values_all = []

            for i in range(MAX_EPS):
                r = []
                a = []
                for j in range(num_steps):
                    # Get Blue and Red actions
                    blue_action = str(cyborg.get_last_action('Blue'))
                    red_action = str(cyborg.get_last_action('Red'))
                    observation_flat = observation.flatten()
                    action = agent.get_action(observation, action_space)
                    observation, rew, done, info = wrapped_cyborg.step(action)
                    action_name = action_id_to_name.get(action, "Unknown Action")


                    # Extract issues from the observation
                    issues = extract_issues_from_observation(obs)

                    # Step 2: Extract SHAP values and map them to features
                    feature_names, flattened_values = extract_feature_names_and_values(obs)
                    observation_flat = np.array(flattened_values, dtype=np.float32).reshape(1, -1)
                    



                                        # Compute SHAP values for the current observation
                    shap_values = explainer.shap_values(observation_flat)

                    # Filter SHAP values to match the number of extracted feature names
                    shap_values_filtered = shap_values[0][:len(feature_names)]

                    # Map filtered SHAP values to feature names and observed values
                    mapped_shap_values = map_shap_to_observation(shap_values_filtered, feature_names, flattened_values)




        
                    # SHAP prediction
                    prediction_output = custom_predict(observation_flat)
                    shap_values = explainer.shap_values(observation_flat)

            
                
          
                    shap_values_all.append(shap_values[0])

                    max_influence_feature = None
                    max_shap_value = None
                    
                    anomalies = blue_wrapper._detect_anomalies(obs)
                    #print(f"Anomalies detected: {anomalies}")



                    # Iterate over feature names and SHAP values
                    for feature, shap_value in zip(feature_names, shap_values[0]):
                        if isinstance(shap_value, (np.ndarray, list)):  # If shap_value is an array
                            shap_value = shap_value[0]  # Extract the first element or process as needed

                        if shap_value > 0 and (max_shap_value is None or shap_value > max_shap_value):
                            max_shap_value = shap_value
                            max_influence_feature = feature




                    # Detect anomalies using BlueTableWrapper
                    if isinstance(observation, dict):  # Ensure it's a dict for anomaly detection
                        anomalies = blue_wrapper._detect_anomalies(observation)
                        #print(f"Anomalies detected: {anomalies}")

                    # Extract detailed context for the most influential feature
                    detailed_context = None
                    if max_influence_feature:
                        host = max_influence_feature.split('_')[0]
                        if host in anomalies:
                            process_info = anomalies.get(host, {}).get('Processes', [])
                            if process_info:
                                detailed_context = f"{host}'s process (PID: {process_info[0].get('PID')}) was critical in deciding the action."

                                        # Verify alignment of SHAP values and features

                

                    # Check the remaining SHAP values beyond the feature names
                    if len(shap_values[0]) > len(feature_names):
                        extra_values = shap_values[0][len(feature_names):]
                        #print("Extra SHAP Values:", extra_values)

                    # Log the explanation based on the type
                    log_shap_explanation(
                        red_action=red_action,
                        blue_action=blue_action,
                        shap_values=shap_values,
                        mapped_shap_values=mapped_shap_values,
                        issues=issues,
                        explanation_type=explanation_type
                    )
        

                

                    r.append(rew)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

                agent.end_episode()
                total_reward.append(sum(r))
                actions.append(a)
                observation = wrapped_cyborg.reset()

            summarize_results(shap_values_all, feature_names)


            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')

            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')


#print(f"Number of features SHAP expects: {len(shap_values_all[0])}")
#print(f"Number of feature names provided: {len(feature_names)}")

# Aggregate SHAP values
feature_influence = {}
for shap_values in shap_values_all:
    for shap_value, feature in zip(shap_values, feature_names):
        if feature not in feature_influence:
            feature_influence[feature] = 0
        feature_influence[feature] += abs(shap_value)

sorted_influence = sorted(feature_influence.items(), key=lambda x: x[1], reverse=True)

print("\n=== Top Influential Features Across All Episodes ===")
for feature, total_influence in sorted_influence[:5]:
    print(f"{feature}: {total_influence:.5f}")


if len(shap_values_all[0]) != len(feature_names):
    shap_values_all = [values[:len(feature_names)] for values in shap_values_all]

# Generate SHAP summary plot
#shap.summary_plot(np.array(shap_values_all), feature_names=feature_names)