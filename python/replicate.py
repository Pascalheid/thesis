"""
Running this module yields all the tables and figures that can be found in
my thesis.
"""
import itertools
import os
import pickle

import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.demand_function import get_demand

from python.auxiliary import get_extensive_specific_sensitivity
from python.auxiliary import get_iskhakov_results
from python.auxiliary import sensitivity_simulation
from python.auxiliary import simulate_data
from python.auxiliary import simulate_figure_3_and_4
from python.auxiliary import transform_grid
from python.auxiliary_figures import get_figure_2
from python.auxiliary_figures import get_figure_3_and_4
from python.auxiliary_figures import get_mse_figure
from python.auxiliary_figures import get_sensitivity_density_both
from python.auxiliary_figures import get_sensitivity_figure
from python.auxiliary_tables import get_table_1
from python.auxiliary_tables import get_table_2


# set up for table 1
discount_factor = [0.975, 0.985, 0.995, 0.999, 0.9995, 0.9999]
approach = ["NFXP", "MPEC"]
starting_cost_params = np.vstack((np.arange(4, 9), np.arange(1, 6)))
starting_expected_value_fun = np.zeros(175)
number_runs = 250
number_buses = 50
number_periods = 120
number_states = 175
number_cost_params = 2

# get results for table 1
# EITHER you can run the simulation (it took one night on my machine)
results = get_iskhakov_results(
    discount_factor,
    approach,
    starting_cost_params,
    starting_expected_value_fun,
    number_runs,
    number_buses,
    number_periods,
    number_states,
    number_cost_params,
)
# OR you can rely on the supplied results
results = pd.read_pickle("data/get_iskhakov_results")

# create table 1
table_1 = get_table_1(results, number_runs, starting_cost_params)

# get table 2
my_results_table_2, iskhakov_table_2 = get_table_2(results, discount_factor, approach)

# get figure 2
beta = 0.975
get_figure_2(results, beta)

# EITHER simulate the demand function over the Monte Carlo runs (took an hour for me)
# and get correlation of the estimated parameters as well as Figure 3 and 4
demand, rc_range, true_demand, correlation = simulate_figure_3_and_4(
    results, beta, number_states, number_buses
)
get_figure_3_and_4(demand, rc_range, true_demand)

# OR alternatively load the previously simulated results
names = ["demand", "rc_range", "true_demand", "correlation"]
temp = {}
for name in names:
    temp[name] = pd.read_pickle(f"data/{name}.pickle")
get_figure_3_and_4(temp["demand"], temp["rc_range"], temp["true_demand"])

# EITHER get simulated data on which the sensitivity analysis is applied,
# running it yourself
# set up for simulation of data sets
disc_fac = 0.975
num_buses = 50
num_periods = 120
num_states = 400
number_cost_params = 2
cost_params = np.array([11.7257, 2.4569])
trans_params = np.array(
    [
        0.0937 / 2,
        0.0937 / 2,
        0.4475 / 2,
        0.4475 / 2,
        0.4459 / 2,
        0.4459 / 2,
        0.0127 / 2,
        0.0127 / 2,
        0.0002 / 2,
        0.0002 / 2,
    ]
)
scale = 1e-3

# simulate data sets with grid size 400
np.random.seed(123)
seeds = np.random.randint(1000, 1000000, 250)
simulated_data_400 = []
for seed in seeds:
    df = simulate_data(
        seed,
        disc_fac,
        num_buses,
        num_periods,
        num_states,
        cost_params,
        trans_params,
        lin_cost,
        scale,
    )
    simulated_data_400.append(df)

# derive smaller gird size data sets from that
simulated_data_200 = []
simulated_data_100 = []
for data in simulated_data_400:
    data_200 = data.copy().apply(transform_grid)
    simulated_data_200.append(data_200)
    simulated_data_100.append(data_200.copy().apply(transform_grid))

# transform all data sets such that usage is correct
for data_list in [simulated_data_100, simulated_data_200, simulated_data_400]:
    for data in data_list:
        data["usage"][1:] = (
            data["state"][1:].astype(int).to_numpy()
            - data["state"][:-1].astype(int).to_numpy()
        )
        data.reset_index(inplace=True)
        replacements = data.loc[data["decision"] == 1].index + 1
        data.loc[data.index.intersection(replacements), "usage"] = data.reindex(
            replacements
        )["state"]
        data.set_index(["Bus_ID", "period"], drop=True, inplace=True)
        data.loc[(slice(None), 0), "usage"] = np.nan

# save data sets
pickle.dump(simulated_data_400, open("data/simulated_data_400.pickle", "wb"))
pickle.dump(simulated_data_200, open("data/simulated_data_200.pickle", "wb"))
pickle.dump(simulated_data_100, open("data/simulated_data_100.pickle", "wb"))


# OR alternatively just load the data sets
simulated_data_400 = pickle.load(open("data/simulated_data_400.pickle", "rb"))
simulated_data_200 = pickle.load(open("data/simulated_data_200.pickle", "rb"))
simulated_data_100 = pickle.load(open("data/simulated_data_100.pickle", "rb"))


# EITHER run sensitivity analysis from whose results all the following figures
# in the sensitivity section are derived

# set up specifications for which the simulated data sets should be estimated
# changing specifications
discount_factors = [0.975, 0.985]
cost_functions = ["linear", "quadratic", "cubic"]
scales = [1e-3, 1e-5, 1e-8]
grid_sizes = [100, 200, 400]
gradients = ["Yes", "No"]
approaches = ["NFXP", "MPEC"]
specifications = list(
    itertools.product(
        discount_factors, zip(cost_functions, scales), grid_sizes, gradients, approaches
    )
)
specifications = list(zip(specifications, np.arange(len(specifications))))
# fixed specifications
number_runs = 250

# run the extensive simulation with multiple cores using the BHHH for NFXP and MPEC
# only works if before pip install -e . when being in the thesis directory
sensitivity_results = Parallel(n_jobs=os.cpu_count(), verbose=50)(
    delayed(sensitivity_simulation)(specification, number_runs, "estimagic_bhhh")
    for specification in specifications
)

# or alternatively normal run without joblib
sensitivity_data = []
for specification in specifications:
    sensitivity_data.append(
        sensitivity_simulation(specification, number_runs, "estimagic_bhhh")
    )


# add the simulation for NFXP with the scipy L-BFGS-B
discount_factors = [0.975, 0.985]
cost_functions = ["linear", "quadratic", "cubic"]
scales = [1e-3, 1e-5, 1e-8]
grid_sizes = [100, 200, 400]
gradients = ["Yes", "No"]
approaches = ["NFXP"]
specifications = list(
    itertools.product(
        discount_factors, zip(cost_functions, scales), grid_sizes, gradients, approaches
    )
)
specifications = list(zip(specifications, np.arange(len(specifications))))
# fixed specifications
number_runs = 250

# only works if before pip install -e . when being in the thesis directory
sensitivity_results = Parallel(n_jobs=os.cpu_count(), verbose=50)(
    delayed(sensitivity_simulation)(specification, number_runs, "scipy_L-BFGS-B")
    for specification in specifications
)

# or alternatively normal run without joblib
sensitivity_data = []
for specification in specifications:
    sensitivity_data.append(
        sensitivity_simulation(specification, number_runs, "scipy_L-BFGS-B")
    )


# OR as the simulation takes very long, you can also just load the results below
# load data
sensitivity_results_bhhh = pd.read_pickle("data/sensitivity_results_full_bhhh.pickle")
sensitivity_results_lbfgsb = pd.read_pickle(
    "data/sensitivity_results_full_lbfgsb.pickle"
)

# Get the true demand (the true QoI) of the underlying data generating process
cost_params = np.array([11.7257, 2.4569])
trans_params = np.array(
    [
        0.0937 / 2,
        0.0937 / 2,
        0.4475 / 2,
        0.4475 / 2,
        0.4459 / 2,
        0.4459 / 2,
        0.0127 / 2,
        0.0127 / 2,
        0.0002 / 2,
        0.0002 / 2,
    ]
)
demand_params = np.concatenate((trans_params, cost_params))
init_dict = {
    "model_specifications": {
        "discount_factor": 0.975,
        "number_states": 400,
        "maint_cost_func": "linear",
        "cost_scale": 1e-3,
    },
}
demand_dict = {
    "RC_lower_bound": 11,
    "RC_upper_bound": 11,
    "demand_evaluations": 1,
    "tolerance": 1e-10,
    "num_periods": 12,
    "num_buses": 50,
}
true_demand = (
    get_demand(init_dict, demand_dict, demand_params)["demand"]
    .astype(float)
    .to_numpy()[0]
)


# get figures 5 to 13 based on the simulation results from above
# partial sensitivity
# model dimension
# only cost
specifications = [
    [0.975, "linear", 400, "Yes"],
    [0.975, "quadratic", 400, "Yes"],
    [0.975, "cubic", 400, "Yes"],
]
labels = ["correct", "quadratic", "cubic"]
figure_name = "5"
sensitivity_data = get_extensive_specific_sensitivity(
    sensitivity_results_lbfgsb, specifications
)[1]
get_sensitivity_figure(
    sensitivity_data, specifications, labels, figure_name, legend=True
)

# cost and discount factor
specifications = [
    [0.975, "linear", 400, "Yes"],
    [0.985, "linear", 400, "Yes"],
    [0.985, "quadratic", 400, "Yes"],
    [0.985, "cubic", 400, "Yes"],
]
labels = ["correct", r"$\beta$", r"$\beta$" " &\n quadratic", r"$\beta$" " &\n cubic"]
figure_name = "6"
sensitivity_data = get_extensive_specific_sensitivity(
    sensitivity_results_lbfgsb, specifications
)[1]
get_sensitivity_figure(sensitivity_data, specifications, labels, figure_name)

# numerical dimension
# just grid size
specifications = [
    [0.975, "linear", 400, "Yes"],
    [0.975, "linear", 200, "Yes"],
    [0.975, "linear", 100, "Yes"],
]
labels = ["correct", "grid\n200", "grid\n100"]
figure_name = "7"
sensitivity_data = get_extensive_specific_sensitivity(
    sensitivity_results_lbfgsb, specifications
)[1]
get_sensitivity_figure(sensitivity_data, specifications, labels, figure_name)

# gradient and grid size
specifications = [
    [0.975, "linear", 400, "Yes"],
    [0.975, "linear", 400, "No"],
    [0.975, "linear", 200, "No"],
    [0.975, "linear", 100, "No"],
]
labels = ["correct", "num.\ngradient", "num.&\ngrid 200", "num.&\ngrid 100"]
figure_name = "8"
sensitivity_data = get_extensive_specific_sensitivity(
    sensitivity_results_lbfgsb, specifications
)[1]
get_sensitivity_figure(sensitivity_data, specifications, labels, figure_name)

# all together
specifications = [
    [0.975, "linear", 400, "Yes"],
    [0.985, "linear", 400, "Yes"],
    [0.985, "quadratic", 400, "Yes"],
    [0.985, "cubic", 400, "Yes"],
    [0.985, "cubic", 400, "No"],
    [0.985, "cubic", 200, "No"],
    [0.985, "cubic", 100, "No"],
]
labels = len(specifications) * [""]
figure_name = "9"
sensitivity_data = get_extensive_specific_sensitivity(
    sensitivity_results_lbfgsb, specifications
)[1]
get_sensitivity_figure(sensitivity_data, specifications, labels, figure_name)

# get MSE figure
discount_factors = [0.975, 0.985]
cost_functions = ["linear", "quadratic", "cubic"]
grid_sizes = [100, 200, 400]
gradients = ["Yes", "No"]
specs = list(itertools.product(discount_factors, cost_functions, grid_sizes, gradients))
specifications = []
for spec in specs:
    specifications.append(list(spec))
sensitivity_data = get_extensive_specific_sensitivity(
    sensitivity_results_lbfgsb, specifications
)[1]
get_mse_figure(sensitivity_data, specifications, "figure_10")

# look at the distributions of the QoI per specification for MPEC
get_sensitivity_density_both(
    sensitivity_results_lbfgsb, "MPEC", "figure_11", np.arange(72), mark=[27, 45]
)


# get figure 5 using the NFXP with BHHH instead of L-BFGS-B
specifications = [
    [0.975, "linear", 400, "Yes"],
    [0.975, "quadratic", 400, "Yes"],
    [0.975, "cubic", 400, "Yes"],
]
labels = ["correct", "quadratic", "cubic"]
figure_name = "12"
sensitivity_data = get_extensive_specific_sensitivity(
    sensitivity_results_bhhh, specifications
)[1]
get_sensitivity_figure(
    sensitivity_data, specifications, labels, figure_name, legend=True
)

# look at the distributions of the QoI per specification for NFXP with L-BFGS-B
get_sensitivity_density_both(
    sensitivity_results_lbfgsb, "NFXP", "figure_13", np.arange(72), mark=[27, 45]
)
