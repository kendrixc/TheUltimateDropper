---
layout: default
title: Proposal
---

## Summary

The goal of this project is to create an AI that can play the map *The Dropper* by Bigre.  In the map the player must complete 16 different levels; in each level, the player starts off at the top of a large drop and must fall down and land in the water to avoid dying.  On the way down there are several obstacles that will kill the player if they land on it, as well as the water at the bottom being positioned differently for each map.  For inputs the AI will receive the blocks located directly around and below it (to a certain distance, to simulate a player only being able to see a certain distance), it’s distance from the ground level, as well as the target water block.  The AI output will be fairly simple in that it must specify a direction (NSEW) to move in, as well as determining what block to initially fall from.


## AI/ML Algorithms

The current plan is to complete this project using reinforcement learning to train a neural network.   It will receive blocks and distances as input and will output weights of which direction it believes it should move (NSEW).


## Evaluation Plan

The primary goal of the AI is to survive falling, which will only be possible if it avoids obstacles and lands in the small area of water at the bottom.  In order to train the AI, it will be rewarded for getting further down the map than previous attempts, along with not straying too far away from the landing area (in the X and Y directions).  Simple cases will be just a straight path down and obstacle-less non-straight paths down.  In the best case scenario the AI will be able to play any of the 16 levels first try—without having to be trained for each map, which should enable it to play other dropper maps.  However, we expect that, much like a human, the AI will become overall better at playing the game, but will need to try out each map a few times to learn exactly what to do.
