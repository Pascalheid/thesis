"""
Sensititvity that is ready
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

from python.sensitivity_functions import sensitivity_simulation
from python.sensitivity_functions import simulate_data
from python.sensitivity_functions import transform_grid

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


# # set up estimation
# # changing specifications
# discount_factors = [0.975, 0.995, 0.9999]
# cost_functions = ["linear", "square root", "quadratic"]
# scales = [1e-3, 1e-2, 1e-5]
# grid_sizes = [100, 200, 400]
# gradients = ["Yes", "No"]
# approaches = ["NFXP", "MPEC"]
# specifications = list(itertools.product(
#     discount_factors, zip(cost_functions, scales), grid_sizes, gradients, approaches))

# # fixed specifications
# number_runs = 5
# number_buses = 50
# number_periods = 120

# # Initialize the set up for the nested fixed point algorithm
# stopping_crit_fixed_point = 1e-13
# switch_tolerance_fixed_point = 1e-2

# # Initialize the set up for MPEC
# rel_ipopt_stopping_tolerance = 1e-6

# # initialize empty data frame for the results
# index_names = ["Discount Factor", "Cost Function",
#              "Grid Size", "Analytical Gradient",
#              "Approach"]
# index = pd.MultiIndex.from_product([discount_factors,
#                                     cost_functions,
#                                     grid_sizes,
#                                     gradients,
#                                     approaches,
#                                     range(number_runs)],
#                                    names=[*index_names, "Run"])

# columns=["RC", "theta_11", "theta_12", "theta_13", "theta_30", "theta_31",
#          "theta_32", "theta_33", "theta_34", "theta_35", "theta_36",
#          "theta_37", "theta_38", "theta_39", "theta_310",
#          "CPU Time", "Converged", "# of Major Iter.", "# of Func. Eval.",
#          "# of Func. Eval. (Total)", "# of Bellm. Iter.", "# of N-K Iter."]

# results = pd.DataFrame(index=index, columns=columns)

# init_dict_nfxp = {
#     "model_specifications": {},
#     "optimizer": {
#         "approach": "NFXP",
#         "algorithm": "estimagic_bhhh",
#     },
#     "alg_details": {
#         "threshold": stopping_crit_fixed_point,
#         "switch_tol": switch_tolerance_fixed_point,
#     },
# }

# init_dict_mpec = {
#     "model_specifications": {},
#     "optimizer": {
#         "approach": "MPEC",
#         "algorithm": "ipopt",
#         "tol": rel_ipopt_stopping_tolerance,
#     },
# }
# column_slicer_nfxp = ["CPU Time", "Converged", "# of Major Iter.", "# of Func. Eval.",
#                       "# of Bellm. Iter.", "# of N-K Iter."]
# column_slicer_mpec = ["CPU Time", "Converged", "# of Major Iter.", "# of Func. Eval.",
#                       "# of Func. Eval. (Total)"]
# for specification in specifications:
#     indexer = list(specification)
#     indexer[1] = list(indexer[1])[0]
#     specification=dict(zip(index_names, specification))

#     # load data
#     data_sets = pickle.load(open("data/simulated_data_" + str(
#         specification["Grid Size"]) + ".pickle", "rb"))

#     # prepare initialization dicts
#     if specification["Approach"] == "NFXP":
#         init_dict_nfxp["model_specifications"]["discount_factor"] = specification[
# "Discount Factor"]
#         init_dict_nfxp["model_specifications"]["number_states"] = specification["Grid Size"]
#         init_dict_nfxp["model_specifications"]["maint_cost_func"] = specification[
# "Cost Function"][0]
#         init_dict_nfxp["model_specifications"]["cost_scale"] = specification[
# "Cost Function"][1]
#         init_dict_nfxp["optimizer"]["gradient"] = specification["Analytical Gradient"]

#         for run in np.arange(number_runs):
#             # Run estimation
#             data = data_sets[run]
#             transition_result_nfxp, cost_result_nfxp = estimate(init_dict_nfxp, data)
#             results.loc[(*indexer, run), (slice("RC", "theta_13"))][
#                 :len(cost_result_nfxp["x"])] = cost_result_nfxp["x"]
#             results.loc[(*indexer, run), (slice("theta_30", "theta_310"))][
#                 :len(transition_result_nfxp["x"])] = transition_result_nfxp["x"]
#             results.loc[(*indexer, run), column_slicer_nfxp] = process_result(
#                 specification["Approach"], cost_result_nfxp)

#     elif specification["Approach"] == "MPEC":
#         init_dict_mpec["model_specifications"]["discount_factor"] = specification[
# "Discount Factor"]
#         init_dict_mpec["model_specifications"]["number_states"] = specification["Grid Size"]
#         init_dict_mpec["model_specifications"]["maint_cost_func"] = specification[
# "Cost Function"][0]
#         init_dict_mpec["model_specifications"]["cost_scale"] = specification[
# "Cost Function"][1]
#         init_dict_mpec["optimizer"]["gradient"] = specification["Analytical Gradient"]
#         if specification["Cost Function"][0] in ["linear", "square root", "hyperbolic"]:
#             num_cost_params = 2
#         elif specification["Cost Function"][0] == "quadratic":
#             num_cost_params = 3
#         else:
#             num_cost_params = 4
#         init_dict_mpec["optimizer"]["set_lower_bounds"] = np.concatenate(
#             (np.full(specification["Grid Size"], -np.inf), np.full(num_cost_params, 0.0)))
#         init_dict_mpec["optimizer"]["set_upper_bounds"] = np.concatenate(
#             (np.full(specification["Grid Size"], 50.0), np.full(num_cost_params, np.inf)))

#         for run in np.arange(number_runs):
#             # Run estimation
#             data = data_sets[run]
#             transition_result_mpec, cost_result_mpec = estimate(init_dict_mpec, data)
#             results.loc[(*indexer, run), (slice("RC", "theta_13"))][
#                 :len(cost_result_mpec["x"][
#                     specification["Grid Size"]:])] = cost_result_mpec["x"][
#                         specification["Grid Size"]:]
#             results.loc[(*indexer, run), (slice("theta_30", "theta_310"))][
#                 :len(transition_result_mpec["x"])] = transition_result_mpec["x"]
#             results.loc[(*indexer, run), column_slicer_mpec] = process_result(
#                 specification["Approach"], cost_result_mpec)


# set up estimation
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

# only works if before pip install -e . when bein in the thesis directory
sensitivity_results = Parallel(n_jobs=os.cpu_count(), verbose=50)(
    delayed(sensitivity_simulation)(specification, number_runs)
    for specification in specifications
)

# normal run without joblib
sensitivity_data = []
for specification in specifications:
    sensitivity_data.append(sensitivity_simulation(specification, number_runs))

# using multiprocessing
# chunks = [specifications[i::os.cpu_count()] for i in range(os.cpu_count())]
# pool = Pool(processes=os.cpu_count())
# mp_sensitivity_simulation = partial(sensitivity_simulation, number_runs=number_runs)
# sensitivity_data = pool.map(lambda , chunks)

sensitivity_results = sensitivity_data[0]
for data in sensitivity_data[1:]:
    sensitivity_results = sensitivity_results.append(data)

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
table_1 = table_1.astype(float).round(3)


# Get true demand
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


# Former attempt for grid size tranformation
# simulate data
# disc_fac = 0.975
# num_buses = 50
# num_periods = 120
# num_states = 175
# number_cost_params = 2
# cost_params = np.array([11.7257, 2.4569])
# trans_params = np.array([0.0937, 0.4475, 0.4459, 0.0127, 0.0002])
# scale = 1e-3

# np.random.seed(123)
# seeds = np.random.randint(1000, 1000000, 250)
# simulated_data = []
# for index, seed in enumerate(seeds):
#     df = simulate_data(seed, disc_fac, num_buses, num_periods, num_states,
#                        cost_params, trans_params, lin_cost, scale)
#     simulated_data.append(df)

# # save data
# pickle.dump(simulated_data, open("data/simulated_data.pickle", "wb"))

# # double and triple grid size
# simulated_data_double = Parallel(n_jobs=os.cpu_count())(delayed(grid_loop)(
#     df, 2, num_buses, num_periods) for df in simulated_data)
# pickle.dump(simulated_data_double, open("data/simulated_data_double.pickle", "wb"))

# simulated_data_triple = Parallel(n_jobs=os.cpu_count())(delayed(grid_loop)(
#     df, 3, num_buses, num_periods) for df in simulated_data)
# pickle.dump(simulated_data_triple, open("data/simulated_data_triple.pickle", "wb"))


# @jit(nopython=True)
# def transform_grid(df, scaling, num_buses, num_periods):
#     if scaling == 2:
#         for bus in np.arange(1, num_buses+1):
#             for period in np.arange(num_periods):
#                 if period == 0 or 2*df.loc[(bus, period), "state"]>=df.loc[(
#                         bus, period-1), "state"] or df.loc[(bus, period-1), "decision"] == 1:
#                     df.loc[(bus, period), "state"] = np.random.choice([
#                         2*df.loc[(bus, period), "state"], 2*df.loc[
# (bus, period), "state"] + 1])
#                 else:
#                     df.loc[(bus, period), "state"] = df.loc[(bus, period-1), "state"]

#     if scaling == 3:
#         for bus in np.arange(1, num_buses+1):
#             for period in np.arange(1, num_periods):
#                 if period == 0 or 3*df.loc[(bus, period), "state"]>=df.loc[(
#                         bus, period-1), "state"] or df.loc[(bus, period-1), "decision"] == 1:
#                     df.loc[(bus, period), "state"] = np.random.choice([
#                         3*df.loc[(bus, period), "state"], 3*df.loc[
# (bus, period), "state"] + 1,
#                         3*df.loc[(bus, period), "state"] + 2])
#                 elif 3*df.loc[(bus, period), "state"] - df.loc[
#                         (bus, period-1), "state"] == -1 and df.loc[
#                             (bus, period-1), "decision"] == 0:
#                                 df.loc[(bus, period), "state"] = np.random.choice([
#                                     3*df.loc[(bus, period), "state"] + 1,
#                                     3*df.loc[(bus, period), "state"] + 2])
#                 elif 3*df.loc[(bus, period), "state"] - df.loc[
#                         (bus, period-1), "state"] == -2 and df.loc[
#                             (bus, period-1), "decision"] == 0:
#                                 df.loc[(bus, period), "state"] = df.loc[
#                                     (bus, period-1), "state"]

#     return df


# def grid_loop(df, scaling, num_buses, num_periods):
#     transformed_data = transform_grid(df.copy(), scaling, num_buses, num_periods)
#     return transformed_data


sensitivity_results_1 = pd.read_pickle(
    "data/sensitivity/sensitivity_specification_0.pickle"
)
for spec in np.arange(1, len(specifications)):
    sensitivity_results_1 = sensitivity_results_1.append(
        pd.read_pickle(
            "data/sensitivity/sensitivity_specification_" + str(spec) + ".pickle"
        )
    )
