---
layout: default
title: Proposal
---

## Summary

The goal of this project is to create an AI that can play the map *The Dropper* by Bigre.  In the map, the player must complete 16 different levels; in each level the player starts off at the top of a large drop and must fall and land in a small puddle of water to avoid dying.  On the way down there are several obstacles that will kill the player if they land on it moving too fast, as well as the water at the bottom being positioned differently for each map.  For inputs the AI will receive blocks located directly around and below it (to a certain distance, to simulate a player only being able to see a certain distance), it’s distance from the ground level, as well as the target water block(s).  The AI output will be simple in that it must specify a direction (NSEW) to strafe in at each time step, as well as determining what block to initially fall from.


## AI/ML Algorithms

We plan to use Q-Learning with a neural network to train the AI.  For input, the AI will receive map information about the blocks directly below and around it (to a certain distance), the current distance from the bottom, and its target at the bottom (water).  


## Evaluation Plan

Our primary goal will be to train an AI that can avoid obstacles and land safely at any point at the bottom of the map, with a secondary goal being able to land at a specific point (in the water).  The AI will be rewarded for getting further down the map than previous attempts, along with not straying too far away from the landing area (in the X and Y directions).  Simple test cases will consist of various obstacles to avoid (some big some small) and increasing the occurrence of said obstacles.  In the best case scenario, the AI will be able to play any of the 16 levels first try—without having to be trained for each map, which should enable it to play other dropper maps.  However, we expect that, much like a human, the AI will become overall better at playing the game, but will need to try out each map a few times to learn exactly what to do for each specific level, therefore a metric we can also measure is the average training time needed per level.


## Group Meet Time

Tuesdays 2:00pm – 3:30pm 
