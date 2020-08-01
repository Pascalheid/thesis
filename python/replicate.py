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

from python.auxiliary import get_difference_approach
from python.auxiliary import get_difference_approach_per_run
from python.auxiliary import get_iskhakov_results
from python.auxiliary import get_specific_sensitivity
from python.auxiliary import partial_sensitivity
from python.auxiliary import sensitivity_simulation
from python.auxiliary import simulate_data
from python.auxiliary import simulate_figure_3_and_4
from python.auxiliary import transform_grid
from python.auxiliary_figures import get_figure_2
from python.auxiliary_figures import get_figure_3_and_4
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
# if you want to run the simulation (it took one night on my machine)
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
# if you want rely on the supplied results
results = pd.read_pickle("data/get_iskhakov_results")

# create table 1
table_1 = get_table_1(results, number_runs, starting_cost_params)

# get table 2
my_results_table_2, iskhakov_table_2 = get_table_2(results, discount_factor, approach)

# get figure 2
beta = 0.975
get_figure_2(results, beta)

# simulate the demand function over the Monte Carlo runs (took an hour for me) and
# get correlation of the estimated parameters as well as Figure 3 and 4
demand, rc_range, true_demand, correlation = simulate_figure_3_and_4(
    results, beta, number_states, number_buses
)
get_figure_3_and_4(demand, rc_range, true_demand)

# alternatively load the previously simulated results
names = ["demand", "rc_range", "true_demand", "correlation"]
temp = {}
for name in names:
    temp[name] = pd.read_pickle(f"data/{name}.pickle")
get_figure_3_and_4(temp["demand"], temp["rc_range"], temp["true_demand"])

# get simulated data on which the sensitivity analysis is applied, running it yourself
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


# alternatively just load the data sets
simulated_data_400 = pickle.load("data/simulated_data_400.pickle", "rb")
simulated_data_200 = pickle.load("data/simulated_data_200.pickle", "rb")
simulated_data_100 = pickle.load("data/simulated_data_100.pickle", "rb")


# run sensitivity analysis from whose results all the tables and figures
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

# run the extensive simulation with multiple cores
# only works if before pip install -e . when being in the thesis directory
sensitivity_results = Parallel(n_jobs=os.cpu_count(), verbose=50)(
    delayed(sensitivity_simulation)(specification, number_runs)
    for specification in specifications
)

# or alternatively normal run without joblib
sensitivity_data = []
for specification in specifications:
    sensitivity_data.append(sensitivity_simulation(specification, number_runs))

# The simulation takes very long so you can also juts load the results below
# load data
sensitivity_data = []
for specification in np.arange(46):
    sensitivity_data.append(
        pd.read_pickle(
            "data/sensitivity/sensitivity_specification_"
            + str(specification)
            + ".pickle"
        )
    )
# transform data
sensitivity_results = sensitivity_data[0]
for data in sensitivity_data[1:]:
    sensitivity_results = sensitivity_results.append(data)

# just for now but delete later or keep but declare it asan overview
# Get means and standard deviations for each specification
table_1_temp = (
    sensitivity_results.loc[sensitivity_results["Converged"] == 1]
    .astype(float)
    .groupby(
        level=[
            "Discount Factor",
            "Cost Function",
            "Grid Size",
            "Analytical Gradient",
            "Approach",
        ]
    )
)

statistics = ["Mean", "Standard Deviation"]
index = pd.MultiIndex.from_product(
    [discount_factors, cost_functions, grid_sizes, gradients, approaches, statistics],
    names=[
        "Discount Factor",
        "Cost Function",
        "Grid Size",
        "Analytical Gradient",
        "Approach",
        "Statistic",
    ],
)
table_1 = pd.DataFrame(index=index, columns=sensitivity_results.columns)
table_1.loc(axis=0)[:, :, :, :, :, "Mean"] = table_1_temp.mean()
table_1.loc(axis=0)[:, :, :, :, :, "Standard Deviation"] = table_1_temp.std()
table_1_temp = (
    sensitivity_results["Converged"]
    .astype(float)
    .groupby(
        level=[
            "Discount Factor",
            "Cost Function",
            "Grid Size",
            "Analytical Gradient",
            "Approach",
        ]
    )
)
table_1.loc[
    (slice(None), slice(None), slice(None), slice(None), slice(None), "Mean"),
    "Converged",
] = table_1_temp.mean()
table_1.loc[
    (
        slice(None),
        slice(None),
        slice(None),
        slice(None),
        slice(None),
        "Standard Deviation",
    ),
    "Converged",
] = table_1_temp.std()
table_1 = table_1.astype(float)


# Get true demand of the underlying data generating process
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


# get partial sensitivity
partial_disc_fac = partial_sensitivity(
    sensitivity_results, discount_factors, "Discount Factor"
)
partial_cost_func = partial_sensitivity(
    sensitivity_results, cost_functions, "Cost Function"
)
partial_grid_size = partial_sensitivity(sensitivity_results, grid_sizes, "Grid Size")
partial_gradient = partial_sensitivity(
    sensitivity_results, gradients, "Analytical Gradient"
)

# get overall difference across NFXP and MPEC
overall_difference = get_difference_approach(sensitivity_results)

# difference NFXP and MPEC per run
overall_per_run_difference = get_difference_approach_per_run(sensitivity_results)

# partial change from correct specification
specifications = [
    [0.975, "linear", 400, "Yes"],
    [0.985, "linear", 400, "Yes"],
    [0.975, "quadratic", 400, "Yes"],
    [0.975, "cubic", 400, "Yes"],
    [0.975, "linear", 200, "Yes"],
    [0.975, "linear", 100, "Yes"],
    [0.975, "linear", 400, "No"],
]

small_perturbations = get_specific_sensitivity(sensitivity_results, specifications)[1]

# partial change full (best case scenario)
specifications = [[0.975, "linear", 400, "Yes"], [0.985, "linear", 400, "Yes"]]
change_full_disc_fac = get_specific_sensitivity(sensitivity_results, specifications)[1]

specifications = [
    [0.975, "linear", 400, "Yes"],
    [0.975, "quadratic", 400, "Yes"],
    [0.975, "cubic", 400, "Yes"],
]
change_full_cost_func = get_specific_sensitivity(sensitivity_results, specifications)[1]

specifications = [
    [0.975, "linear", 400, "Yes"],
    [0.975, "linear", 200, "Yes"],
    [0.975, "linear", 100, "Yes"],
]
change_full_grid_size = get_specific_sensitivity(sensitivity_results, specifications)[1]

specifications = [[0.975, "linear", 400, "Yes"], [0.975, "linear", 400, "No"]]
change_full_gradient = get_specific_sensitivity(sensitivity_results, specifications)[1]
