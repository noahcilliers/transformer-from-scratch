"""
so we can try to reduce ppo to a modular training function

this way we can have many files with training functions and then we can just put them together like a build a bear

so this should be low level so we can build on top of it for different adaptations like rlhf if we need

but the basic principle is as follow

min clipping

importance sampling if we are taking multiple steps

so maybe we can have an outer function for the overall and then inner for straight ppo, this way the importance sampling c
"""
import torch
from torch.optim import optim
import copy
from numpy import np

def PPO(model, state_decider, reward_model, value_function, steps_per_turn, epochs, data_loader, compare_models):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for state in data_loader:
        # generate all the actions for a sequence
        states_actions = []
        for i in range(steps_per_turn):
            action = model(state)
            states_actions.append((state, action))
            state = state_decider(state, action)
            if i == steps_per_turn -1 :
                states_actions.append((state, None))


        reference_model = copy.deepcopy(model)
        for i in range(len(states_actions)):
            importance_ratio = compare_models(model, reference_model)
            reward = reward_model(states_actions[i][0], states_actions[i][1])
            value = value_function(states_actions[i+1][0])

            advantage = reward - value
            loss = min(importance_ratio * advantage, np.clip(importance_ratio, 1 - .02, 1 + .02) * advantage)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()





