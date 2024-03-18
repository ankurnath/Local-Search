# get hyparameters from the papars
import os
import pickle
from collections import defaultdict

characters_to_remove = ["(", ")", ","]

parameters=set(["nb_steps","init_network_params","init_weight_std",
                "double_dqn","clip_Q_targets",
                "replay_start_size","replay_buffer_size"
               ,"update_target_frequency","update_learning_rate",
                "initial_learning_rate","peak_learning_rate",
               "peak_learning_rate_step","final_learning_rate",
                "final_learning_rate_step","update_frequency","minibatch_size",
               "max_grad_norm","weight_decay","update_exploration",
                "initial_exploration_rate",
               "final_exploration_rate","final_exploration_step","adam_epsilon",
               'save_network_frequency',"test_frequency"])


config={}
for parameter in parameters:
    config[parameter]=defaultdict(list)



root_folder="experiments"
for folder in os.listdir(root_folder):
    if folder.endswith("spin"):
        sub_folder=os.path.join(root_folder,folder)
        type_of_graph=folder.split('_')[0]
        spin=folder.split('_')[1].split("spin")[0]
#         print(type_of_graph,spin)
        
#         print(os.listdir(os.path.join(sub_folder,"train")))
        
        for file in os.listdir(os.path.join(sub_folder,"train")):
            if file.endswith("eco.py"):
                file_path=os.path.join(sub_folder,"train",file)
                with open(file_path, 'r') as file:
                    file_content = file.read()
                    
                    for line in file_content.split("\n"):
                        line = line.strip()
                        if line and len(line.split('='))==2:
                            
                            key, value = line.split('=')
                            key=key.strip()
                            value=value.strip()


                            for char in characters_to_remove:
                                value  = value .replace(char, "")

                            if key in parameters:
                                val=value.split(" ")[0]
                                if val=="None":
                                    val=None
                                elif val=="True":
                                    val=True
                                elif val=="False":
                                    val=False
                                else:
                                    val=float(val)
                                    
                                    
                                config[key][int(spin)].append((val,type_of_graph))


# config
train_config={}

for parameter in parameters:
    train_config[parameter]={}
    
    for spin in config[parameter]:
        
        if config[parameter][spin][0][0]!=config[parameter][spin][1][0]:
            print(f"{parameter} configuration for spin {spin} are different for ER and BA graphs")
            print(f"{parameter} configuration for {config[parameter][spin][0][1]}:{config[parameter][spin][0][0]}")
            print(f"{parameter} configuration for {config[parameter][spin][1][1]}:{config[parameter][spin][1][0]}")
        
        
        # We take the configuration of BA graphs if there is a conflict
            
        train_config[parameter][spin]=config[parameter][spin][0][0]

# File path to save the pickle file
file_path = 'config.pkl'
# Save the dictionary to a pickle file
with open(file_path, 'wb') as f:
    pickle.dump(train_config, f)

print("Dictionary saved to", file_path)        