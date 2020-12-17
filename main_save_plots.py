try:
    from malmo import MalmoPython
except:
    import MalmoPython

import pydot as pyd
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
OBS_SIZE = 31 # MUST BE AN ODD NUMBER
DEPTH = 100
MAX_GLOBAL_STEPS = 10000000
REPLAY_BUFFER_SIZE = 100000
EPSILON_DECAY = .999
MIN_EPSILON = .1
BATCH_SIZE = 64
GAMMA = .9
TARGET_UPDATE = 25
LEARNING_RATE = 1e-4
START_EPISODE = 500

RUN_TESTS = True
LOAD_NUM = 100
MAX_NUM = 1200 # change this to the max model save

NUM_ACTIONS = 5
my_mission, my_clients, my_mission_record = None, None, None

dist = [0]
AIR, OTHER_BLOCK, WATER = 0, 1, 2
LEVEL = 0

# be sure to change this to YOUR PATH
path = 'C:\Malmo-0.37.0-Windows-64bit_withBoost_Python3.7\Python_Examples\Project\TheUltimateDropper\droppermap'

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
                                <min x="-''' + str(int(OBS_SIZE/2)) + '''" y="-''' + str(DEPTH - 2) + '''" z="-''' + str(int(OBS_SIZE/2)) + '''"/>
                                <max x="''' + str(int(OBS_SIZE/2)) + '''" y="1" z="''' + str(int(OBS_SIZE/2)) + '''"/>
                            </Grid>
                        </ObservationFromGrid>
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''


def create_model(obs_size):
    if RUN_TESTS:
        print(f'Loading model number {LOAD_NUM} for testing.')
        return keras.models.load_model(f'lvl_{LEVEL}_models/TARGET_MODEL_INFO_{LOAD_NUM}')

    if START_EPISODE > 0:
        print(f'Attempting to load model {START_EPISODE}')
        return keras.models.load_model(f'TARGET_MODEL_INFO_{START_EPISODE}')

    inputs = layers.Input(shape = obs_size)
    layer1 = layers.Conv3D(32, kernel_size = (3, 3, 3), activation = 'relu', kernel_initializer = 'he_uniform')(inputs)
    layer2 = layers.MaxPooling3D((2, 2, 2))(layer1)
    layer3 = layers.Conv3D(64, (3, 3, 3), activation = 'relu', kernel_initializer = 'he_uniform')(layer2)
    layer4 = layers.MaxPooling3D((2, 2, 2))(layer3)
    layer6 = layers.Flatten()(layer4)
    layer7 = layers.Dense(128, activation = 'relu', kernel_initializer = 'he_uniform')(layer6)
    layer8 = layers.Dropout(0.4)(layer7)
    layer9 = layers.Dense(64, activation = 'relu', kernel_initializer = 'he_uniform')(layer8)
    layer10 = layers.Dropout(0.4)(layer9)
    action = layers.Dense(NUM_ACTIONS, activation = 'softmax')(layer10)
    return keras.Model(inputs = inputs, outputs = action)


     
def get_action(obs, model, epsilon):
    """Select action according to e-greedy policy"""
    
    if np.random.ranf() <= epsilon:
        action = np.random.choice(NUM_ACTIONS)
        print(f'R {action} | ', end = '')
    else:
        obs = tf.convert_to_tensor(obs)
        obs = tf.expand_dims(obs, 0)
        action_probs = model(obs, training = False)
        action = tf.argmax(action_probs[0].numpy())
        action = random.choices(np.arange(NUM_ACTIONS), weights = action_probs[0].numpy(), k = 1)[0]
        print(f'A {action} | ', end = '')
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
                grid = np.array(observations['floorAll'])
            except KeyError:
                continue

            grid[(grid != 'water') & (grid != 'air')] = OTHER_BLOCK
            grid[grid == 'water'] = WATER
            grid[grid == 'air'] = AIR
            grid = grid.astype(float)
            obs = np.reshape(grid, (DEPTH, OBS_SIZE, OBS_SIZE))
            obs = obs[::-1,:,:]
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

def run_model_tests(agent_host):
    global LOAD_NUM
    test_data = []
    
    for num in np.arange(LOAD_NUM, MAX_NUM + 1, 100):
        LOAD_NUM = num
        model = create_model((DEPTH, OBS_SIZE, OBS_SIZE, 1))
        tf.keras.utils.plot_model(model, to_file='model_vis.png', show_shapes = False, show_layer_names = True, rankdir = 'TB', expand_nested = False, dpi = 96)
        done = False
        test_data.append(None)

        agent_host = init_malmo(agent_host)
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:",error.text)
        obs, pos = get_observation(world_state)

        # Run episode
        while world_state.is_mission_running and not done:
            # Get action
            action_idx = get_action(obs, model, 0)
            
            # forward
            if action_idx == 0:
                agent_host.sendCommand('move 1')
                agent_host.sendCommand('strafe 0')
            # back
            elif action_idx == 1:
                agent_host.sendCommand('move -1')
                agent_host.sendCommand('strafe 0')
            # left
            elif action_idx == 2:
                agent_host.sendCommand('strafe -1')
                agent_host.sendCommand('move 0')
            #right
            elif action_idx == 3:
                agent_host.sendCommand('strafe 1')
                agent_host.sendCommand('move 0')
            # don't move
            elif action_idx == 4:
                agent_host.sendCommand('strafe 0')
                agent_host.sendCommand('move 0')
            
            # Get next observation
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            try:
                next_obs, pos = get_observation(world_state) 
            except KeyError:
                print('Ran into KeyError, continuing...')
                continue

            in_water = False
            mid = int(OBS_SIZE / 2)
            near_blocks = next_obs[:2, mid-2:mid+3, mid-2:mid+3]
            if len(near_blocks[near_blocks == WATER]) > 0:
                in_water = True
                done = True
                print('***IN WATER***')
            if pos is not None:
                test_data[-1] = (252 - pos[1], in_water)
            obs = next_obs
            time.sleep(0.05)

   
    plt.figure()
    plt.title(f'Accuracy of Agent on Level {LEVEL}')
    plt.xlabel('Training Episode')
    plt.ylabel('Height from Starting Position in Blocks')
    lbls = np.arange(100, MAX_NUM + 1, 100)
    plt.xticks(np.arange(1, len(lbls)+1), labels = lbls, rotation = 70)
    for i in range(len(test_data)):
        c = 'blue' if test_data[i][1] else 'red'
        plt.bar(i+1, test_data[i][0], color = c)
    plt.tight_layout()
    plt.savefig(f'lvl_{LEVEL}_test.png')
    exit()

        

        
def train(agent_host):
    """Main loop for the DQN learning algorithm"""
    global LEVEL
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
    num_episode = START_EPISODE
    epsilon = 1.0
    start_time = time.time()
    returns = []
    steps = []

    # if starting not from ep 1 then set epsilon accordingly
    for _ in range(START_EPISODE):
        if epsilon > MIN_EPSILON: epsilon *= EPSILON_DECAY
        else: break

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
        while world_state.is_mission_running and not done:
            # Get action
            action_idx = get_action(obs, model, epsilon)
            
            # forward
            if action_idx == 0:
                agent_host.sendCommand('move 1')
                agent_host.sendCommand('strafe 0')
            # back
            elif action_idx == 1:
                agent_host.sendCommand('move -1')
                agent_host.sendCommand('strafe 0')
            # left
            elif action_idx == 2:
                agent_host.sendCommand('strafe -1')
                agent_host.sendCommand('move 0')
            #right
            elif action_idx == 3:
                agent_host.sendCommand('strafe 1')
                agent_host.sendCommand('move 0')
            # don't move
            elif action_idx == 4:
                agent_host.sendCommand('strafe 0')
                agent_host.sendCommand('move 0')
            
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
                dist[-1] = 252 - pos[1]
                #reward += np.exp((252 - pos[1])/45)
                r_dist = 252 - pos[1]

                # mid should be the x, z position of the player
                # in the observation
                mid = int(OBS_SIZE / 2)
                
                # find the distance to the closest water block
                closest_block_dist = np.inf
                if len(next_obs[next_obs == WATER]) > 0:
                    for y in range(DEPTH):
                        for x in range(OBS_SIZE):
                            for z in range(OBS_SIZE):
                                if next_obs[y, x, z] != WATER:
                                    continue
                                b_dist = np.sqrt((mid - x)**2 + (y)**2 + (mid - z)**2)
                                if b_dist < closest_block_dist:
                                    closest_block_dist = b_dist

                print(f'Dist to Water: {closest_block_dist} | ', end = '')                
                # use the distance to the water as a metric for distance rather than y-level
                if closest_block_dist > 0:
                    r_water = (1 / closest_block_dist)
                else:
                    r_water = 1
                    
                # if there are non air or water blocks in the surrounding
                # area then add negative reward
                near_blocks = next_obs[:10, mid-2:mid+3, mid-2:mid+3]
                r_blocks = len(near_blocks[near_blocks == OTHER_BLOCK])

                # if in the water then reward and end the mission
                near_blocks = near_blocks[:1,::]
                if len(near_blocks[near_blocks == WATER]) > 0:
                    reward = 100000
                    done = True
                    print('***IN WATER***', reward)
                    
            # make reward a linear combination of factors
            if reward != 100000 and pos is not None:
                # don't target water not at the bottom
                if pos[1] > 120: r_water = 0
                # dont discourage because blocks are around the water
                #if pos[1] < 25: r_blocks = 0
                if r_water == np.inf: r_water = 0
                reward = 0 * r_dist + 100 * r_water - 0 * r_blocks
                    
            print(f'Dist from Start: {dist[-1]} | Reward: {reward} | Epsilon: {epsilon}')
            episode_return += reward
            # Store step in replay buffer
            replay_buffer.append((obs, action_idx, next_obs, reward, done))
            obs = next_obs
           
            global_step += 1
            if episode_step > 20:
                done = True
                break
            time.sleep(0.33)
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
        dist.append(0)
        
        if num_episode % 50 == 0:
            log_returns(num_episode)
            model_target.save(f'TARGET_MODEL_INFO_{num_episode}')
                        
                
def log_returns(num):
    plt.figure()
    start_ep = START_EPISODE
    x = np.arange(start_ep, start_ep + len(dist[:-1]))
    m, b = np.polyfit(x, dist[:-1], 1)
    plt.scatter(x, dist[:-1])
    plt.plot(x, m*x + b)
    plt.title('Distance Travelled')
    plt.ylabel('Distance (in Blocks)')
    plt.xlabel('Episode')
    plt.savefig(f'ep_dist_plot_{num}.png')


def init_malmo(agent_host):
    """Initialize new malmo mission"""
    global my_mission, my_clients, my_mission_record
    max_retries = 100

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

    if RUN_TESTS:
        run_model_tests(agent_host)
    else:
        train(agent_host)
    
