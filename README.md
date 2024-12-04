# README

## TSI_RL_HighwayMerge
Assignment og the course Topics on Intelligent Systems (MSc in Artificial Intelligence, 1st semester)

## A little context
### Overview
The aim of this project was to replicate and enhance a paper related to the relation of Multi-Agent Systems and Machine Learning. The chosen paper was "Deep Multi-Agent Reinforcement Learning for Highway On-Ramp Merging in Mixed Traffic". This project that can be viewed on teh following GitHub repository: [MARL_CAVs](https://github.com/DongChen06/MARL_CAVs?tab=readme-ov-file)

### The paper
The paper explores the application of multi-agent reinforcement learning (MARL) to address the challenges of highway on-ramp merging in mixed traffic scenarios involving autonomous vehicles (AVs) and human-driven vehicles (HDVs). A decentralized MARL framework is proposed, enabling AVs to learn safe and efficient merging policies while adapting to dynamic HDV behaviors. The problem is modeled in a gym-like simulation environment with varying traffic densities (easy, medium, hard). Key components include a state space capturing vehicle positions, speeds, and distances; an action space comprising acceleration, deceleration, and lane changes; and a reward function incentivizing safety and efficiency. To enhance learning and safety, the framework incorporates parameter sharing for scalability, action masking to exclude unsafe actions, and a priority-based safety supervisor to manage merging urgency. 

## Training Environment
### On the paper
- **Simulated Scenario**: The model was trained in a highway environment with two main lanes and one merging lane, replicating real-world traffic conditions.

METER IMAGEM QUE USAMOS NO POWERPOINT

#### Key Features and Techniques:
- **Action Masking**: Unsafe actions, such as leaving the highway boundaries, were disabled during training to ensure the agents' safety and focus on valid maneuvers.
- **Curriculum Learning**: While mentioned as a potential enhancement, this technique—progressively introducing more complex scenarios—was not implemented in this study.
- **Priority-Based Safety Supervisor**: This innovative mechanism evaluated the safety of lane-changing actions by predicting potential collisions up to 8 steps ahead, ensuring that only safe actions were executed.
- **Shared Network**: A shared network structure allowed the agents to learn collectively, improving training efficiency and leveraging shared experiences among them.

These components contributed to creating a robust framework for training autonomous vehicles to merge safely and efficiently in challenging highway scenarios.

### **Improvements on the Original Framework**

The goal of our approach was to recreate and enhance the study presented in the paper, incorporating additional features and modifications to improve learning efficiency, safety, and adaptability in highway merging scenarios.

#### **Curriculum Learning Implementation**:  
   While the paper mentioned curriculum learning, it was not implemented in the original study. We introduced this technique to facilitate faster and safer learning by gradually increasing the difficulty levels:
   - **Difficulty Level 1**: Controlling 1 to 3 vehicles in a simple environment.
   - **Difficulty Level 2**: Controlling 2 to 4 vehicles, increasing complexity.
   - **Difficulty Level 3**: Controlling 4 to 6 vehicles, representing more challenging scenarios.

 #### **PPO Model**:  
   We opted for the Proximal Policy Optimization (PPO) algorithm, as it showed superior baseline performance compared to A2C in the paper’s benchmarks. PPO is known for its stability and efficiency in reinforcement learning tasks.
   
#### **Reward Function Adjustments**
We experimented with different reward adjustments to evaluate their impact on agent performance:
- **Collision Penalty**: Adjusted from -200 to -100 and -1000 to study the influence on safety.  
- **Speed Incentive**: Increased from 1 to 2 and 4, promoting faster but controlled driving.  
- **Penalty for Staying in the Merging Lane**: Modified from 4 to 2 and 6 to encourage quicker merging.  
- **Proximity Reward**: Incentivized maintaining a safe distance (e.g., 4 meters) from the car in front, testing variations of 2 and 6 to optimize behavior.

#### **Environment Modifications**
To expand the study, we explored different highway configurations:
- **Single Lane**: Simplifying the environment to focus on merging behavior.  
- **Three-Lane Highway**: Increasing complexity for more realistic scenarios.

### **Performance Metrics**
We analyzed the agents' performance based on:
- **Average Speed**: To evaluate efficiency.  
- **Collision Rates**: To measure safety and identify reward structures yielding optimal results.

### **Additional Features**
1. **Behavioral Variations in Non-Controlled Vehicles**:  
   We simulated varying driving styles to test adaptability:  
   - **Aggressive Behavior**: Vehicles drove faster, executed quicker lane changes, and induced abrupt braking in others.  
   - **Cautious Behavior**: Vehicles maintained larger distances, drove slower, and performed safer lane changes.

2. **Creative Idea – Delivery Vehicles ("Couriers")**:  
   Inspired by real-world delivery objectives, we introduced a scenario where the agent's goal was to reach the destination as quickly as possible:
   - Rewards were weighted more heavily toward speed.  
   - Agents took faster actions while avoiding major disruptions to surrounding vehicles.
  

## Setup

### Create a Python Virtual Environment

`conda create -n marl_cav python=3.6 -y`

### Activate the Virtual Environment

`conda activate marl_cav`

### Install PyTorch and its sddociated libraries

`install pytorch (torch>=1.2.0): `conda install pytorch==1.7.0 torchvision==0.8.1 torchaudio==0.7.0 -c pytorch`

### Install Additional Requirements

`pip install -r requirements.txt`

## Usage

### Training a Model

`python run_xxx.py`

### Evaluating a Trained Model:

`python run_xxx.py --option eva --model-dir path_to_model`

Replace path_to_model with the path to the directory containing your trained model.

