#!/usr/bin/env python
# coding: utf-8

# #  Service API Example on Hartmann6
# 
# The Ax Service API is designed to allow the user to control scheduling of trials and data computation while having an easy to use interface with Ax.
# 
# The user iteratively:
# - Queries Ax for candidates
# - Schedules / deploys them however they choose
# - Computes data and logs to Ax
# - Repeat

# In[1]:


import numpy as np
from ax.service.ax_client import AxClient


# ## 1. Initialize Client
# 
# Create a client object to interface with Ax APIs. By default this runs locally without storage.

# In[3]:


ax = AxClient()


# ## 2. Set Up Experiment
# An experiment consists of a **search space** (parameters and parameter constraints) and **optimization configuration** (objective name, minimization setting, and outcome constraints). Note that:
# - Only `name`, `parameters`, and `objective_name` arguments are required.
# - Dictionaries in `parameters` have the following required keys: "name" - parameter name, "type" - parameter type ("range", "choice" or "fixed"), "bounds" for range parameters, "values" for choice parameters, and "value" for fixed parameters.
# - Dictionaries in `parameters` can optionally include "value_type" ("int", "float", "bool" or "str"), "log_scale" flag for range parameters, and "is_ordered" flag for choice parameters.
# - `parameter_constraints` should be a list of strings of form "p1 >= p2" or "p1 + p2 <= some_bound".
# - `outcome_constraints` should be a list of strings of form "constrained_metric <= some_bound".

# In[4]:


ax.create_experiment(
    name="hartmann_test_experiment",
    parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x3",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x4",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x5",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x6",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
    ],
    objective_name="hartmann6",
    minimize=True,  # Optional, defaults to False.
    parameter_constraints=["x1 + x2 <= 2.0"],  # Optional.
    outcome_constraints=["l2norm <= 1.25"],  # Optional.
)


# ## 3. Define how to evaluate trials
# When using Ax a service, evaluation of parameterizations suggested by Ax is done either locally or, more commonly, using an external scheduler. Below is a dummy evaluation function that outputs data for two metrics "hartmann6" and "l2norm". Note that all returned metrics correspond to either the `objective_name` set on experiment creation or the metric names mentioned in `outcome_constraints`.

# In[ ]:


def hartmann6(x: np.ndarray) -> float:
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = 10 ** (-4) * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )
        y = 0.0
        for j, alpha_j in enumerate(alpha):
            t = 0
            for k in range(6):
                t += A[j, k] * ((x[k] - P[j, k]) ** 2)
            y -= alpha_j * np.exp(-t)
        return y


# In[5]:


def evaluate(parameters):
    x = np.array([parameters.get(f"x{i+1}") for i in range(6)])
    # In our case, standard error is 0, since we are computing a synthetic function.
    return {"hartmann6": (hartmann6(x), 0.0), "l2norm": (np.sqrt((x ** 2).sum()), 0.0)}


# ## 4. Run optimization loop
# With the experiment set up, we can start the optimization loop.
# 
# At each step, the user queries the client for a new trial then submits the evaluation of that trial back to the client.
# 
# Note that Ax auto-selects an appropriate optimization algorithm based on the search space. For more advance use cases that require a specific optimization algorithm, pass a `generation_strategy` argument into the `AxClient` constructor.

# In[6]:


for i in range(10):
    print(f"Running trial {i+1}/10...")
    parameters, trial_index = ax.get_next_trial()
     # Local evaluation here can be replaced with deployment to external system.
    ax.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))


# ## 5. Retrieve best parameters
# 
# Once it's complete, we can access the best parameters found, as well as the corresponding metric values.

# In[8]:


best_parameters, metrics = ax.get_best_parameters()
best_parameters


# In[9]:


means, covariances = metrics
means["hartmann6"]


# ## 6. Special Cases

# **Evaluation failure**: should any optimization iterations fail during evaluation, `log_trial_failure` will ensure that the same trial is not proposed again.

# In[10]:


ax.log_trial_failure(trial_index=trial_index)


# **Adding custom trials**: should there be need to evaluate a specific parameterization, `attach_trial` will add it to the experiment.

# In[11]:


ax.attach_trial(parameters={"x1": 9.0, "x2": 9.0})

