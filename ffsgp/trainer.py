# Fassio's Genetic Programming Toolbox
#
# MIT License
# Copyright (c) 2024 Davide Fassio

# TODO: take inspiration from Operon for the evolutionary workflow
#
# initial_pop -> parents_pop => offspring_gen => offspring_pop
#                      ^-------------------------------v
# 
# To generate new offspring select two parents, do crossover, then with probability
# p_m do a mutation, then evaluate fitness.
#

# TODO: fitness evaluation:
# MSE: mean squared error
# NMSE: normalized MSE (divide by the variance)
# R^2: coefficient of determination

# TODO: survivor selection + elitism (top 1%)

# TODO: parent selectio use tournament selection + fitness hole to reduce tree size