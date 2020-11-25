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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Hyperparameters
SIZE = 50
REWARD_DENSITY = .1
PENALTY_DENSITY = .02
OBS_SIZE = 31 # MUST BE AN ODD NUMBER
DEPTH = 40
MAX_EPISODE_STEPS = 300
MAX_GLOBAL_STEPS = 100000
REPLAY_BUFFER_SIZE = 10000
EPSILON_DECAY = .999
MIN_EPSILON = .1
BATCH_SIZE = 64
GAMMA = .9
TARGET_UPDATE = 25
LEARNING_RATE = 1e-4
START_TRAINING = 100

NUM_ACTIONS = 5
my_mission, my_clients, my_mission_record = None, None, None

dist = [0]
AIR, OTHER_BLOCK, WATER = 0, 1, 2
LEVEL = 0
# be sure to change this to YOUR PATH
path = 'C:/Users/AnthonyN/Desktop/TheUltimateDropper/DropperMap'

level_coords = [(-611.5, 252, -745.5),
                (-634.5, 252, -690.5),
                (-581.5, 252, -698.5),
                (-555.5, 252, -750.5),
                (-524.5, 252, -755.5),
                (-456.5, 252, -749.5),
                (-442.5, 252, -672.5),
                (-527.5, 252, -661.5),
                (-487.5, 252, -625.5),
                (-445.5, 252, -622.5),
                (-416.5, 252, -628.5),
                (-363.5, 248, -644.5),
                (-361.5, 240, -708.5)]

def GetMissionXML():

    # change the starting position based on the level chosen
    pos = f'x="{level_coords[LEVEL][0]}" y="{level_coords[LEVEL][1]}" z="{level_coords[LEVEL][2]}"'

    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                <About><Summary>TheUltimateDropper</Summary></About>

                <ServerSection>
                    <ServerInitialConditions>
                        <Time>
                            <StartTime>12000</StartTime>
                            <AllowPassageOfTime>true</AllowPassageOfTime>
                        </Time>
                        <Weather>clear</Weather>
                    </ServerInitialConditions>
                    <ServerHandlers>
                        <FileWorldGenerator src="''' + path + '''"/>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>

                <AgentSection mode="Survival">
                    <Name>TheUltimateDropper_Agent</Name>
                    <AgentStart>
                        <Placement ''' + pos + ''' pitch="90" yaw="180"/>
                    </AgentStart>
                    
                    <AgentHandlers>
                        <ContinuousMovementCommands/>
                        <ObservationFromFullStats/>
                        <ObservationFromGrid>
                            <Grid name="floorAll">
                                <min x="-''' + str(int(OBS_SIZE/2)) + '''" y="-''' + str(DEPTH - 1) + '''" z="-''' + str(int(OBS_SIZE/2)) + '''"/>
                                <max x="''' + str(int(OBS_SIZE/2)) + '''" y="0" z="''' + str(int(OBS_SIZE/2)) + '''"/>
                            </Grid>
                        </ObservationFromGrid>
                        <AgentQuitFromReachingCommandQuota total="'''+str(MAX_EPISODE_STEPS)+'''" />
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''


def create_model(obs_size):
    inputs = layers.Input(shape = obs_size)
    layer1 = layers.Conv3D(32, kernel_size = (3, 3, 3), activation = 'relu', padding = 'same')(inputs)
    layer2 = layers.MaxPooling3D((2, 2, 2), padding = 'same')(layer1)
    layer3 = layers.Conv3D(64, (3, 3, 3), activation = 'relu', padding = 'same')(layer2)
    layer4 = layers.MaxPooling3D((2, 2, 2), padding = 'same')(layer3)
    layer5 = layers.Conv3D(128, (3, 3, 3), activation = 'relu', padding = 'same')(layer4)
    layer6 = layers.MaxPooling3D(pool_size = (2, 2, 2), padding = 'same')(layer5)
    layer7 = layers.Flatten()(layer6)
    layer8 = layers.Dense(128, activation = 'relu')(layer7)
    action = layers.Dense(NUM_ACTIONS, activation = 'linear')(layer8)
    return keras.Model(inputs = inputs, outputs = action)


def get_action(obs, model, epsilon, allow_break_action):
    """Select action according to e-greedy policy"""
    
    if np.random.ranf() <= epsilon:
        action = np.random.choice(NUM_ACTIONS)
        print(f'R {action} ', end = '')
    else:
        obs = tf.convert_to_tensor(obs)
        obs = tf.expand_dims(obs, 0)
        action_probs = model(obs, training = False)
        action = tf.argmax(action_probs[0].numpy())
        print(f'A {action} ', end = '')

    return action
    
def get_observation(world_state):
    """Use the agent observation API to get a DEPTH x OBS_SIZE/2 x OBS_SIZE/2 cube around the agent"""
    
    obs, pos = np.zeros((DEPTH, OBS_SIZE, OBS_SIZE)), None
    
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

            try:
                grid = observations['floorAll']
            except KeyError:
                continue
            
            grid_binary = []
            for x in grid:
                if x == 'water': grid_binary.append(2)
                elif x == 'air': grid_binary.append(0)
                else: grid_binary.append(1)
                    
            obs = np.reshape(grid_binary, (DEPTH, OBS_SIZE, OBS_SIZE))
            break

    return obs, pos


def prepare_batch(replay_buffer):
    """Randomly sample batch from replay buffer and prepare tensors"""
    if BATCH_SIZE < len(replay_buffer):
        batch_data = random.sample(replay_buffer, BATCH_SIZE)
    else:
        batch_data = replay_buffer
    obs = tf.convert_to_tensor([x[0] for x in batch_data], dtype = tf.float32)
    action = tf.convert_to_tensor([x[1] for x in batch_data], dtype = tf.int32)
    next_obs = tf.convert_to_tensor([x[2] for x in batch_data], dtype = tf.float32)
    reward = tf.convert_to_tensor([x[3] for x in batch_data], dtype = tf.float32)
    done = tf.convert_to_tensor([x[4] for x in batch_data], dtype = tf.float32)
    return obs, action, next_obs, reward, done
  

def learn(batch, model, model_target, optim, loss_func):
    """Update CNN according to DQN Loss function"""
    print('Learning...')
    obs, action, next_obs, reward, done = batch
    future_rewards = model_target.predict(next_obs)
    updated_q_values = reward + GAMMA * tf.reduce_max(future_rewards, axis = 1) * (1 - done)
    masks = tf.one_hot(action, NUM_ACTIONS)
    with tf.GradientTape() as tape:
        q_values = model(obs)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis = 1)
        loss = loss_func(updated_q_values, q_action)

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))
    print('Done learning.')
    return loss
    

def train(agent_host):
    """Main loop for the DQN learning algorithm"""

    # Init networks
    model = create_model((DEPTH, OBS_SIZE, OBS_SIZE, 1))
    model_target = create_model((DEPTH, OBS_SIZE, OBS_SIZE, 1))

    # Init optimizer
    optim = keras.optimizers.Adam(learning_rate = LEARNING_RATE, clipnorm = 1.0)
    # Init replay buffer
    replay_buffer = deque(maxlen = REPLAY_BUFFER_SIZE)
    # Init loss function
    loss_func = keras.losses.Huber()
    # Init vars
    global_step = 0
    num_episode = 0
    epsilon = 1
    start_time = time.time()
    returns = []
    steps = []

    # Begin main loop
    loop = tqdm(total = MAX_GLOBAL_STEPS, position = 0, leave = False)
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
            action_idx = get_action(obs, model, epsilon, allow_break_action)
            
            # forward
            if action_idx == 0: agent_host.sendCommand('move 1')
            # back
            elif action_idx == 1: agent_host.sendCommand('move -1')
            # left
            elif action_idx == 2: agent_host.sendCommand('strafe -1')
            #right
            elif action_idx == 3: agent_host.sendCommand('strafe 1')
            # don't move
            elif action_idx == 4: pass
            
            # We have to manually calculate terminal state to give malmo time to register the end of the mission
            # If you see "commands connection is not open. Is the mission running?" you may need to increase this
            episode_step += 1
            
            # Get next observation
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            try:
                next_obs, pos = get_observation(world_state) 
            except KeyError:
                print('Ran into KeyError, continuing...')
                continue

            # Get reward
            reward = 0
            if pos is not None:
                # make distance to ground an exponential function
                # so that it gets rewarded more for getting closer
                reward += 2 * np.exp((252 - pos[1])/45)
                #reward = 2 * (252 - pos[1])
                dist[-1] = 252 - pos[1]
                # reward it for having air blocks in the path around and directly below
                # it, give it a negative reward for other blocks, and a big
                # positive reward for having water in the path
                s = int(OBS_SIZE / 2)
                reward_obs = next_obs[:,s-2 : s+3,s-2 : s+3]
                #reward += 1 * len(reward_obs[reward_obs == AIR])
                reward += 30 * len(reward_obs[reward_obs == WATER])
            # Store step in replay buffer
            replay_buffer.append((obs, action_idx, next_obs, reward, done))
            obs = next_obs
           
            global_step += 1
        # Learn only after each death since it takes too much time
        # to train in the middle of the drop
        batch = prepare_batch(replay_buffer)
        loss = learn(batch, model, model_target, optim, loss_func)
        episode_loss += loss

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY

        if global_step % TARGET_UPDATE == 0:
            model_target.set_weights(model.get_weights())
    
        num_episode += 1
        returns.append(episode_return)
        steps.append(global_step)
        avg_return = sum(returns[-min(len(returns), 10):]) / min(len(returns), 10)
        loop.update(episode_step)
        loop.set_description('Episode: {} Steps: {} Time: {:.2f} Loss: {:.2f} Last Return: {:.2f} Avg Return: {:.2f}'.format(
            num_episode, global_step, (time.time() - start_time) / 60, episode_loss, episode_return, avg_return))

        if num_episode % 100 == 0:
            log_returns(num_episode)
            model.save(f'MODEL_INFO_{num_episode}')

        dist.append(0)


def log_returns(num):
    plt.figure()
    plt.plot(np.arange(1, 1 + len(dist)), dist)
    plt.title('Distance Travelled')
    plt.ylabel('Distance (in Blocks)')
    plt.xlabel('Iteration')
    plt.savefig(f'distance_plot_{num}.png')


def init_malmo(agent_host):
    """Initialize new malmo mission"""
    
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
