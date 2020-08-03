"""
Functions for the sensitivity analysis
"""
import pickle

import numpy as np
import pandas as pd
from ruspy.estimation.estimation import estimate
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.demand_function import get_demand
from ruspy.model_code.fix_point_alg import calc_fixp
from ruspy.simulation.simulation import simulate


def simulate_data(
    seed,
    disc_fac,
    num_buses,
    num_periods,
    num_states,
    cost_params,
    trans_params,
    cost_func,
    scale,
):

    init_dict = {
        "simulation": {
            "discount_factor": disc_fac,
            "periods": num_periods,
            "seed": seed,
            "buses": num_buses,
        },
    }

    costs = calc_obs_costs(num_states, cost_func, cost_params, scale)
    trans_mat = create_transition_matrix(num_states, trans_params)
    ev = calc_fixp(trans_mat, costs, disc_fac)[0]
    df = simulate(init_dict["simulation"], ev, costs, trans_mat)

    return df


def transform_grid(column):
    if column.name == "state":
        column = np.floor(column / 2)
    return column


def process_result(approach, cost_result):
    if approach == "NFXP":
        result = np.array([])
        for name in [
            "fun",
            "time",
            "status",
            "n_iterations",
            "n_evaluations",
            "n_contraction_steps",
            "n_newt_kant_steps",
        ]:
            result = np.concatenate((result, np.array([cost_result[name]])))

    else:
        result = np.array([])
        for name in [
            "fun",
            "time",
            "status",
            "n_iterations",
            "n_evaluations",
            "n_evaluations_total",
        ]:
            result = np.concatenate((result, np.array([cost_result[name]])))

    return result


def get_qoi(init_dict, params):
    demand_dict = {
        "RC_lower_bound": 11,
        "RC_upper_bound": 11,
        "demand_evaluations": 1,
        "tolerance": 1e-10,
        "num_periods": 12,
        "num_buses": 50,
    }
    demand = get_demand(init_dict, demand_dict, params)
    demand = demand["demand"].astype(float).to_numpy()[0]
    return demand


def sensitivity_simulation(specification, number_runs):

    # Initialize the set up for the nested fixed point algorithm
    stopping_crit_fixed_point = 1e-13
    switch_tolerance_fixed_point = 1e-2

    # Initialize the set up for MPEC
    rel_ipopt_stopping_tolerance = 1e-6

    # get specifications in order
    index_names = [
        "Discount Factor",
        "Cost Function",
        "Grid Size",
        "Analytical Gradient",
        "Approach",
    ]
    identifier = specification[1]
    indexer = list(specification[0])
    indexer[1] = list(indexer[1])[0]
    specification = dict(zip(index_names, specification[0]))

    # load data
    data_sets = pickle.load(
        open("data/simulated_data_" + str(specification["Grid Size"]) + ".pickle", "rb")
    )

    # set up empty dataframe for results
    index = pd.MultiIndex.from_product(
        [*[[element] for element in indexer], range(number_runs)],
        names=[*index_names, "Run"],
    )

    columns = [
        "RC",
        "theta_11",
        "theta_12",
        "theta_13",
        "theta_30",
        "theta_31",
        "theta_32",
        "theta_33",
        "theta_34",
        "theta_35",
        "theta_36",
        "theta_37",
        "theta_38",
        "theta_39",
        "theta_310",
        "Likelihood",
        "Demand",
        "CPU Time",
        "Converged",
        "# of Major Iter.",
        "# of Func. Eval.",
        "# of Func. Eval. (Total)",
        "# of Bellm. Iter.",
        "# of N-K Iter.",
    ]

    results = pd.DataFrame(index=index, columns=columns)

    if specification["Approach"] == "NFXP":
        init_dict_nfxp = {
            "model_specifications": {
                "discount_factor": specification["Discount Factor"],
                "number_states": specification["Grid Size"],
                "maint_cost_func": specification["Cost Function"][0],
                "cost_scale": specification["Cost Function"][1],
            },
            "optimizer": {
                "approach": "NFXP",
                "algorithm": "estimagic_bhhh",
                "gradient": specification["Analytical Gradient"],
            },
            "alg_details": {
                "threshold": stopping_crit_fixed_point,
                "switch_tol": switch_tolerance_fixed_point,
            },
        }
        column_slicer_nfxp = [
            "Likelihood",
            "CPU Time",
            "Converged",
            "# of Major Iter.",
            "# of Func. Eval.",
            "# of Bellm. Iter.",
            "# of N-K Iter.",
        ]

        for run in np.arange(number_runs):
            # Run estimation
            data = data_sets[run]
            transition_result_nfxp, cost_result_nfxp = estimate(init_dict_nfxp, data)
            results.loc[(*indexer, run), (slice("RC", "theta_13"))][
                : len(cost_result_nfxp["x"])
            ] = cost_result_nfxp["x"]
            results.loc[(*indexer, run), (slice("theta_30", "theta_310"))][
                : len(transition_result_nfxp["x"])
            ] = transition_result_nfxp["x"]
            results.loc[(*indexer, run), column_slicer_nfxp] = process_result(
                specification["Approach"], cost_result_nfxp
            )
            results.loc[(*indexer, run), "Demand"] = get_qoi(
                init_dict_nfxp,
                np.concatenate((transition_result_nfxp["x"], cost_result_nfxp["x"])),
            )
            results.to_pickle(
                "data/sensitivity/sensitivity_specification_"
                + str(identifier)
                + ".pickle"
            )

    elif specification["Approach"] == "MPEC":
        if specification["Cost Function"][0] in ["linear", "square root", "hyperbolic"]:
            num_cost_params = 2
        elif specification["Cost Function"][0] == "quadratic":
            num_cost_params = 3
        else:
            num_cost_params = 4

        init_dict_mpec = {
            "model_specifications": {
                "discount_factor": specification["Discount Factor"],
                "number_states": specification["Grid Size"],
                "maint_cost_func": specification["Cost Function"][0],
                "cost_scale": specification["Cost Function"][1],
            },
            "optimizer": {
                "approach": "MPEC",
                "algorithm": "ipopt",
                "gradient": specification["Analytical Gradient"],
                "tol": rel_ipopt_stopping_tolerance,
                "set_lower_bounds": np.concatenate(
                    (
                        np.full(specification["Grid Size"], -np.inf),
                        np.full(num_cost_params, 0.0),
                    )
                ),
                "set_upper_bounds": np.concatenate(
                    (
                        np.full(specification["Grid Size"], 50.0),
                        np.full(num_cost_params, np.inf),
                    )
                ),
            },
        }

        column_slicer_mpec = [
            "Likelihood",
            "CPU Time",
            "Converged",
            "# of Major Iter.",
            "# of Func. Eval.",
            "# of Func. Eval. (Total)",
        ]

        for run in np.arange(number_runs):
            # Run estimation
            data = data_sets[run]
            transition_result_mpec, cost_result_mpec = estimate(init_dict_mpec, data)
            results.loc[(*indexer, run), (slice("RC", "theta_13"))][
                : len(cost_result_mpec["x"][specification["Grid Size"] :])
            ] = cost_result_mpec["x"][specification["Grid Size"] :]
            results.loc[(*indexer, run), (slice("theta_30", "theta_310"))][
                : len(transition_result_mpec["x"])
            ] = transition_result_mpec["x"]
            results.loc[(*indexer, run), column_slicer_mpec] = process_result(
                specification["Approach"], cost_result_mpec
            )
            results.loc[(*indexer, run), "Demand"] = get_qoi(
                init_dict_mpec,
                np.concatenate(
                    (
                        transition_result_mpec["x"],
                        cost_result_mpec["x"][specification["Grid Size"] :],
                    )
                ),
            )
            results.to_pickle(
                "data/sensitivity/sensitivity_specification_"
                + str(identifier)
                + ".pickle"
            )

    return results


# try to circumvent stuff
# def get_qoi_for_real(sensitivity_data):
#     init_dict = {
#         "model_specifications": {},
#         }
#     demand_dict = {
#         "RC_lower_bound": 11,
#         "RC_upper_bound": 11,
#         "demand_evaluations": 1,
#         "tolerance": 1e-10,
#         "num_periods": 12,
#         "num_buses": 50,
#     }
#     index = sensitivity_data.index
#     for row in range(sensitivity_data.shape[0]):
#         # set up initialization dict
#         if index[row][1] == "linear":
#             scale = 1e-3
#         elif index[row][1] == "square root":
#             scale = 1e-2
#         elif index[row][1] == "hyperbolic":
#             scale = 1e-1
#         elif index[row][1] == "quadratic":
#             scale = 1e-5
#         elif index[row][1] == "cubic":
#             scale = 1e-8

#         init_dict["model_specifications"]["discount_factor"] = index[row][0]
#         init_dict["model_specifications"]["maint_cost_func"] = index[row][1]
#         init_dict["model_specifications"]["number_states"] = index[row][2]
#         init_dict["model_specifications"]["cost_scale"] = scale

#         # get the estimated strucutral parameters
#         cost_params = sensitivity.loc[index[row], slice("RC", "theta_13")].astype(
#             float).to_numpy()
#         cost_params = cost_params[~np.isnan(cost_params)]
#         trans_params = sensitivity.loc[index[row], slice("theta_30", "theta_310")].astype(
#             float).to_numpy()
#         trans_params = trans_params[~np.isnan(trans_params)]
#         demand_params = np.concatenate((trans_params, cost_params))

#         # get qoi
#         demand = get_demand(init_dict, demand_dict, demand_params)[
#             "demand"].astype(float).to_numpy()[0]
