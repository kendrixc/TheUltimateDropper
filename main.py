try:
    from malmo import MalmoPython
except:
    import MalmoPython

import os
import sys
import time
import json
import random
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt 
import numpy as np
from numpy.random import randint
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Hyperparameters
SIZE = 50
REWARD_DENSITY = .1
PENALTY_DENSITY = .02
OBS_SIZE = 11
DEPTH = 20
MAX_EPISODE_STEPS = 1000
MAX_GLOBAL_STEPS = 100000
REPLAY_BUFFER_SIZE = 10000
EPSILON_DECAY = .999
MIN_EPSILON = .1
BATCH_SIZE = 128
GAMMA = .9
TARGET_UPDATE = 100
LEARNING_RATE = 1e-4
START_TRAINING = 130
LEARN_FREQUENCY = 1
ACTION_DICT = {
    0: 'forward',  
    1: 'back',  
    2: 'left', 
    3: 'right',
    4: 'nothing'
}
dist = [0]

my_mission, my_clients, my_mission_record = None, None, None


# Q-Value Network
class QNetwork(nn.Module):

    def __init__(self, obs_size, action_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(np.prod(obs_size), 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 50),
                                 nn.ReLU(),
                                 nn.Linear(50, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, action_size))
        
    def forward(self, obs):
        """
        Estimate q-values given obs

        Args:
            obs (tensor): current obs, size (batch x obs_size)

        Returns:
            q-values (tensor): estimated q-values, size (batch x action_size)
        """
        batch_size = obs.shape[0]
        obs_flat = obs.view(batch_size, -1)
        return self.net(obs_flat)


def GetMissionXML():
    air = '\n\t\t\t'.join([f'<DrawBlock x="{-1*i}" y="250" z="-746" type="air"/>' for i in range(610, 615)])

    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                <About>
                    <Summary>TheUltimateDropper</Summary>
                </About>

                <ServerSection>
                    <ServerInitialConditions>
                        <Time>
                            <StartTime>12000</StartTime>
                            <AllowPassageOfTime>true</AllowPassageOfTime>
                        </Time>
                        <Weather>clear</Weather>
                    </ServerInitialConditions>
                    <ServerHandlers>
                        <FileWorldGenerator src="C:\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Python_Examples\Project\TheUltimateDropper\DropperMap"/>
                        <DrawingDecorator>''' + air + '''</DrawingDectorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>

                <AgentSection mode="Survival">
                    <Name>TheUltimateDropper_Agent</Name>
                    <AgentStart>
                        <Placement x="-611.5" y="252" z="-745.5" pitch="90" yaw="180"/>
                    </AgentStart>
                    <AgentHandlers>
                        <ContinuousMovementCommands/>
                        <ObservationFromFullStats/>
                        <ObservationFromGrid>
                            <Grid name="floorAll">
                                <min x="-''' + str(int(OBS_SIZE/2)) + '''" y="-19" z="-''' + str(int(OBS_SIZE/2)) + '''"/>
                                <max x="''' + str(int(OBS_SIZE/2)) + '''" y="0" z="''' + str(int(OBS_SIZE/2)) + '''"/>
                            </Grid>
                        </ObservationFromGrid>
                        <AgentQuitFromReachingCommandQuota total="'''+str(MAX_EPISODE_STEPS)+'''" />
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''


def get_action(obs, q_network, epsilon, allow_break_action):
    """
    Select action according to e-greedy policy

    Args:
        obs (np-array): current observation, size (obs_size)
        q_network (QNetwork): Q-Network
        epsilon (float): probability of choosing a random action

    Returns:
        action (int): chosen action [0, action_size)
    """

    #print(epsilon)
    #if np.random.ranf() <= epsilon:
    #    print("RANDOM")
    return np.random.choice([0, 1, 2, 3, 4])

    print("OTHER")
    # Prevent computation graph from being calculated
    with torch.no_grad():
        # Calculate Q-values fot each action
        obs_torch = torch.tensor(obs.copy(), dtype=torch.float).unsqueeze(0)
        action_values = q_network(obs_torch)

        # Remove attack/mine from possible actions if not facing a diamond
        if not allow_break_action:
            action_values[0, 3] = -float('inf')  

        # Select action with highest Q-value
        action_idx = torch.argmax(action_values).item()
        
    return action_idx


def init_malmo(agent_host):
    """
    Initialize new malmo mission.
    """
    global my_mission, my_clients, my_mission_record
    max_retries = 3
    if my_mission is None:
        my_mission = MalmoPython.MissionSpec(GetMissionXML(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)
       
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available
        
    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_clients, my_mission_record, 0, "TheUltimateDropper")
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2)

    return agent_host


def get_observation(world_state):
    """
    Use the agent observation API to get a 2 x 5 x 5 grid around the agent. 
    The agent is in the center square facing up.

    Args
        world_state: <object> current agent world state

    Returns
        observation: <np.array>
        0 = AIR
        1 = NOT WATER AND NOT AIR
        2 = WATER
    """
    obs = np.zeros((DEPTH, OBS_SIZE, OBS_SIZE))
    pos = None
    
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        
        if len(world_state.errors) > 0:
            raise AssertionError('Could not load grid.')

        if world_state.number_of_observations_since_last_state > 0:
            # First we get the json from the observation API
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            # Get observation
            pos = (observations['XPos'], observations['YPos'], observations['ZPos'])
            grid = observations['floorAll']
            grid_binary = []
            for x in grid:
                if x == 'water':
                    grid_binary.append(2)
                elif x == 'air':
                    grid_binary.append(0)
                else:
                    grid_binary.append(1)
                    
            obs = np.reshape(grid_binary, (DEPTH, OBS_SIZE, OBS_SIZE))
            # Rotate observation with orientation of agent
            yaw = observations['Yaw']
            if yaw == 270:
                obs = np.rot90(obs, k=1, axes=(1, 2))
            elif yaw == 0:
                obs = np.rot90(obs, k=2, axes=(1, 2))
            elif yaw == 90:
                obs = np.rot90(obs, k=3, axes=(1, 2))
            
            break

    return obs, pos


def prepare_batch(replay_buffer):
    """
    Randomly sample batch from replay buffer and prepare tensors

    Args:
        replay_buffer (list): obs, action, next_obs, reward, done tuples

    Returns:
        obs (tensor): float tensor of size (BATCH_SIZE x obs_size
        action (tensor): long tensor of size (BATCH_SIZE)
        next_obs (tensor): float tensor of size (BATCH_SIZE x obs_size)
        reward (tensor): float tensor of size (BATCH_SIZE)
        done (tensor): float tensor of size (BATCH_SIZE)
    """
    batch_data = random.sample(replay_buffer, BATCH_SIZE)
    obs = torch.tensor([x[0] for x in batch_data], dtype=torch.float)
    action = torch.tensor([x[1] for x in batch_data], dtype=torch.long)
    next_obs = torch.tensor([x[2] for x in batch_data], dtype=torch.float)
    reward = torch.tensor([x[3] for x in batch_data], dtype=torch.float)
    done = torch.tensor([x[4] for x in batch_data], dtype=torch.float)
    
    return obs, action, next_obs, reward, done
  

def learn(batch, optim, q_network, target_network):
    """
    Update Q-Network according to DQN Loss function

    Args:
        batch (tuple): tuple of obs, action, next_obs, reward, and done tensors
        optim (Adam): Q-Network optimizer
        q_network (QNetwork): Q-Network
        target_network (QNetwork): Target Q-Network
    """
    obs, action, next_obs, reward, done = batch

    optim.zero_grad()
    values = q_network(obs).gather(1, action.unsqueeze(-1)).squeeze(-1)
    target = torch.max(target_network(next_obs), 1)[0]
    target = reward + GAMMA * target * (1 - done)
    loss = torch.mean((target - values) ** 2)
    loss.backward()
    optim.step()

    return loss.item()


def log_returns():
    plt.figure()
    plt.plot(np.arange(1, 1 + len(dist)), dist)
    plt.title('Distance Travelled')
    plt.ylabel('Distance (in Blocks)')
    plt.xlabel('Iteration')
    plt.savefig('random_agent.png')


def train(agent_host):
    """
    Main loop for the DQN learning algorithm

    Args:
        agent_host (MalmoPython.AgentHost)
    """
    # Init networks
    q_network = QNetwork((DEPTH, OBS_SIZE, OBS_SIZE), len(ACTION_DICT))
    target_network = QNetwork((DEPTH, OBS_SIZE, OBS_SIZE), len(ACTION_DICT))
    target_network.load_state_dict(q_network.state_dict())

    # Init optimizer
    optim = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    # Init replay buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # Init vars
    global_step = 0
    num_episode = 0
    epsilon = 1
    start_time = time.time()
    returns = []
    steps = []

    # Begin main loop
    loop = tqdm(total=MAX_GLOBAL_STEPS, position=0, leave=False)
    while global_step < MAX_GLOBAL_STEPS:
        time.sleep(1)
        episode_step = 0
        episode_return = 0
        episode_loss = 0
        done = False

        # Setup Malmo
        agent_host = init_malmo(agent_host)
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:",error.text)
        obs, pos = get_observation(world_state)

        # Run episode
        while world_state.is_mission_running:
            # Get action
            allow_break_action = obs[1, int(OBS_SIZE/2)-1, int(OBS_SIZE/2)] == 1
            action_idx = get_action(obs, q_network, epsilon, allow_break_action)
            
            # forward
            if action_idx == 0:
                agent_host.sendCommand('move 1')
            # back
            elif action_idx == 1:
                agent_host.sendCommand('move -1')
            # left
            elif action_idx == 2:
                agent_host.sendCommand('strafe -1')
            #right
            elif action_idx == 3:
                agent_host.sendCommand('strafe 1')
            # don't move
            elif action_idx == 4:
                pass
            
            # We have to manually calculate terminal state to give malmo time to register the end of the mission
            # If you see "commands connection is not open. Is the mission running?" you may need to increase this
            episode_step += 1
            
            # Get next observation
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            next_obs, pos = get_observation(world_state) 

            # Get reward
            reward = 0
            if pos is not None:
                reward = ((252 - pos[1]) / 10)**2
                dist[-1] = 252 - pos[1]
            if next_obs[0][4][4] == 2 or next_obs[1][4][4] == 2:
                reward += 500
                done = True
            
            print(reward)
            # if block beneth the player is water +100

            # should make this more complex in the future such as
            # if its a clear path to water or +air blocks beneth player
            # (less blocks more reward)
     
            # Store step in replay buffer
            replay_buffer.append((obs, action_idx, next_obs, reward, done))
            obs = next_obs

            # Learn
            global_step += 1
            if global_step > START_TRAINING and global_step % LEARN_FREQUENCY == 0:
                batch = prepare_batch(replay_buffer)
                loss = learn(batch, optim, q_network, target_network)
                episode_loss += loss

                if epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY

                if global_step % TARGET_UPDATE == 0:
                    target_network.load_state_dict(q_network.state_dict())
    
        
        num_episode += 1
        returns.append(episode_return)
        steps.append(global_step)
        avg_return = sum(returns[-min(len(returns), 10):]) / min(len(returns), 10)
        loop.update(episode_step)
        loop.set_description('Episode: {} Steps: {} Time: {:.2f} Loss: {:.2f} Last Return: {:.2f} Avg Return: {:.2f}'.format(
            num_episode, global_step, (time.time() - start_time) / 60, episode_loss, episode_return, avg_return))

        if num_episode > 20:
            log_returns()
            exit(1)

        dist.append(0)


if __name__ == '__main__':
    # Create default Malmo objects:
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    train(agent_host)
