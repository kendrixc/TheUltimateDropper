---
layout: default
title: Status Report
---


## Project Summary
The goal of this project is to create an AI that can play the map The Dropper by Bigre. In the map, the player must complete 16 different levels; in each level the agent starts off at the top of a large drop and must control themselves as they fall—with the goal of landing safely in water located at the bottom of each level.  Each level increases in difficulty by adding more blocks and structures the agent must avoid hitting.  For inputs—the agent will receive a map of the blocks located directly around and below it (20 x 10 x 10).  The AI then outputs which direction it believes the agent should move (NSEW); this is done continuously as the agent falls. The agent is rewarded based on the vertical distance it traveled before failing or succeeding. 

## Approach
Each observation is a numpy array of size (20 * 10 * 10) grid. If it observes water, air or neither water nor air the observations are set to 2,0 and 1 respectively. 
We used the Epsilon - Greedy algorithm to get the actions and the probability. This is either a random action if the probability is less than the threshold value or the agent chooses to move forward , move backward or strife left or strife right according to the observed value in the Q - table.
The reward the agent gets is initially set to None as there are no rewards at the start location. But as the agents move downwards towards the water it keeps increasing which is an absolute difference between the current location and the start location. If it’s observed that the agent touches the water( it’s destination) it gets a 200 point reward.   
 
## Evaluation
Currently the agent works by randomly choosing a direction to move in at each time step.  Below is a graph showing the distance traveled by the agent across 20 trials (higher is better with—252 meaning it made it to the bottom).

This clearly shows that the distance the agent travels is random and that the agent is not good. 

## Remaining Goals and Challenges
 
## Resources Used

