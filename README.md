# TSI_RL_HighwayMerge
Deep Multi-Agent Reinforcement Learning for Highway On-Ramp Merging in Mixed Traffic - Topics in Intelligent Systems

# Code Replication
create a python virtual environment: `conda create -n marl_cav python=3.6 -y`


active the virtual environment: `conda activate marl_cav`


install pytorch (torch>=1.2.0): `conda install pytorch==1.7.0 torchvision==0.8.1 torchaudio==0.7.0 -c pytorch`


install the requirements: `pip install -r requirements.txt`

# Usage
To run the code, use the following commands:

For training a model:

`python run_xxx.py --option train`

For evaluating a trained model:

`python run_xxx.py --option eva --model-dir path_to_model`

The config files contain the parameters for the MARL policies.


