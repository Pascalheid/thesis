"""
Contains all functions that are needed in intermediary steps in order to obtain
certain tables and figures of the thesis.
"""
import pickle

import numpy as np
import pandas as pd
import scipy.io
from ruspy.estimation.estimation import estimate
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.demand_function import get_demand
from ruspy.model_code.fix_point_alg import calc_fixp
from ruspy.simulation.simulation import simulate


def get_iskhakov_results(
    discount_factor,
    approach,
    starting_cost_params,
    starting_expected_value_fun,
    number_runs,
    number_buses,
    number_periods,
    number_states,
    number_cost_params,
):
    """
    Run the Monte Carlo Simulation to replicate Iskhakov et al. (2016)

    Parameters
    ----------
    discount_factor : list
        beta vector for which to run the simulation.
    approach : list
        run with NFXP and/or MPEC.
    starting_cost_params : numpy.array
        contains the starting values for the cost parameters.
    starting_expected_value_fun : numpy.array
        contains the starting values of the expected values for MPEC.
    number_runs : float
        number of runs per beta and starting vector combination.
    number_buses : int
        number of buses per data set.
    number_periods : int
        number of months per data set.
    number_states : int
        number of grid points in which the mileage state is discretized.
    number_cost_params : int
        number of cost parameters.

    Returns
    -------
    results : pd.DataFrame
        contains the estimates for the structural parameters per run.

    """

    # Initialize the set up for the nested fixed point algorithm
    stopping_crit_fixed_point = 1e-13
    switch_tolerance_fixed_point = 1e-2

    # Initialize the set up for MPEC
    lower_bound = np.concatenate(
        (np.full(number_states, -np.inf), np.full(number_cost_params, 0.0))
    )
    upper_bound = np.concatenate(
        (np.full(number_states, 50.0), np.full(number_cost_params, np.inf))
    )
    rel_ipopt_stopping_tolerance = 1e-6

    init_dict_nfxp = {
        "model_specifications": {
            "number_states": number_states,
            "maint_cost_func": "linear",
            "cost_scale": 1e-3,
        },
        "optimizer": {
            "approach": "NFXP",
            "algorithm": "estimagic_bhhh",
            # implies that we use analytical first order derivatives as opposed
            # to numerical ones
            "gradient": "Yes",
        },
        "alg_details": {
            "threshold": stopping_crit_fixed_point,
            "switch_tol": switch_tolerance_fixed_point,
        },
    }

    init_dict_mpec = {
        "model_specifications": {
            "number_states": number_states,
            "maint_cost_func": "linear",
            "cost_scale": 1e-3,
        },
        "optimizer": {
            "approach": "MPEC",
            "algorithm": "ipopt",
            # implies that we use analytical first order derivatives as opposed
            # to numerical ones
            "gradient": "Yes",
            "tol": rel_ipopt_stopping_tolerance,
            "set_lower_bounds": lower_bound,
            "set_upper_bounds": upper_bound,
        },
    }

    # Initialize DataFrame to store the results of each run of the Monte Carlo simulation
    index = pd.MultiIndex.from_product(
        [
            discount_factor,
            range(number_runs),
            range(starting_cost_params.shape[1]),
            approach,
        ],
        names=["Discount Factor", "Run", "Start", "Approach"],
    )

    columns = [
        "RC",
        "theta_11",
        "theta_30",
        "theta_31",
        "theta_32",
        "theta_33",
        "CPU Time",
        "Converged",
        "# of Major Iter.",
        "# of Func. Eval.",
        "# of Bellm. Iter.",
        "# of N-K Iter.",
    ]

    results = pd.DataFrame(index=index, columns=columns)

    # Main loop to calculate the results for each run
    for factor in discount_factor:
        # load simulated data
        mat = scipy.io.loadmat(
            "data/RustBusTableXSimDataMC250_beta" + str(int(100000 * factor))
        )

        for run in range(number_runs):
            if run in np.arange(10, number_runs, 10):
                results.to_pickle("data/intermediate/results_" + str(factor))
            data = process_data(mat, run, number_buses, number_periods)

            for start in range(starting_cost_params.shape[1]):
                # Adapt the Initiation Dictionairy of NFXP for this run
                init_dict_nfxp["model_specifications"]["discount_factor"] = factor
                init_dict_nfxp["optimizer"]["params"] = pd.DataFrame(
                    starting_cost_params[:, start], columns=["value"]
                )

                # Run NFXP using ruspy
                transition_result_nfxp, cost_result_nfxp = estimate(
                    init_dict_nfxp, data
                )

                # store the results of this run
                results.loc[factor, run, start, "NFXP"] = process_result_iskhakov(
                    "NFXP", transition_result_nfxp, cost_result_nfxp, number_states
                )

                # Adapt the Initiation Dictionairy of MPEC for this run
                init_dict_mpec["model_specifications"]["discount_factor"] = factor
                init_dict_mpec["optimizer"]["params"] = np.concatenate(
                    (starting_expected_value_fun, starting_cost_params[:, start])
                )

                # Run MPEC using ruspy
                transition_result_mpec, cost_result_mpec = estimate(
                    init_dict_mpec, data
                )

                # store the results of this run
                results.loc[factor, run, start, "MPEC"].loc[
                    ~results.columns.isin(["# of Bellm. Iter.", "# of N-K Iter."])
                ] = process_result_iskhakov(
                    "MPEC", transition_result_mpec, cost_result_mpec, number_states
                )

    return results


def process_data(df, run, number_buses, number_periods):
    """
    prepare the raw data set from matlab for the Monte Carlo simulation in
    ``get_iskhakov_results``.

    Parameters
    ----------
    df : pd.DataFrame
        contains the raw data of Iskhakov et al. created with their original
        matlab code.
    run : int
        indicates the run in the Monte Carlo simulation.
    number_buses : int
        number of buses per data set.
    number_periods : int
        number of months per data set.

    Returns
    -------
    data : pd.DataFrame
        the processed data set that can be used in the ruspy estimate function.

    """
    state = df["MC_xt"][:, :, run] - 1
    decision = df["MC_dt"][:, :, run]
    usage = df["MC_dx"][:-1, :, run] - 1
    first_usage = np.full((1, usage.shape[1]), np.nan)
    usage = np.vstack((first_usage, usage))

    data = pd.DataFrame()
    state_new = state[:, 0]
    decision_new = decision[:, 0]
    usage_new = usage[:, 0]

    for i in range(0, len(state[0, :]) - 1):
        state_new = np.hstack((state_new, state[:, i + 1]))
        decision_new = np.hstack((decision_new, decision[:, i + 1]))
        usage_new = np.hstack((usage_new, usage[:, i + 1]))

    data["state"] = state_new
    data["decision"] = decision_new
    data["usage"] = usage_new

    iterables = [range(number_buses), range(number_periods)]
    index = pd.MultiIndex.from_product(iterables, names=["Bus_ID", "period"])
    data.set_index(index, inplace=True)

    return data


def process_result_iskhakov(approach, transition_result, cost_result, number_states):
    """
    process the raw results from a Monte Carlo simulation run in the
    ``get_iskhakov_results`` function.

    Parameters
    ----------
    approach : string
        indicates whether the raw results were created from the NFXP or MPEC.
    transition_result : dict
        the result dictionairy of ruspy for the tranisition parameters.
    cost_result : dict
        the result dictionairy of ruspy for the cost parameters.
    number_states : int
        number of grid points in which the mileage state is discretized.

    Returns
    -------
    result : numpy.array
        contains the transformed results of a Monte Carlo simulation run.

    """
    if approach == "NFXP":
        result = np.concatenate((cost_result["x"], transition_result["x"][:4]))

        for name in [
            "time",
            "status",
            "n_iterations",
            "n_evaluations",
            "n_contraction_steps",
            "n_newt_kant_steps",
        ]:
            result = np.concatenate((result, np.array([cost_result[name]])))

    else:
        result = np.concatenate(
            (cost_result["x"][number_states:], transition_result["x"][:4])
        )

        for name in ["time", "status", "n_iterations", "n_evaluations"]:
            result = np.concatenate((result, np.array([cost_result[name]])))

    return result


def simulate_figure_3_and_4(results, beta, number_states, number_buses):
    """
    Get the implied demand function for certain parameter estimates for Figure
    3 and 4.

    Parameters
    ----------
    results : pd.DataFrame
        the results of the Monte Carlo simulation in Iskhakov et al. (2016)
        created by ``get_iskhakov_results``.
    beta : float
        indicates for which of the beta the demand function is supposed to be
        derived.
    number_states : int
        number of grid points in which the mileage state is discretized.
    number_buses : int
        number of buses per data set.

    Returns
    -------
    demand : pd.DataFrame
        contains the demand over a range of replacement costs depending on
        the estimated structural parameters of a Monte Carlo run.
    rc_range : np.array
        range over which the demand is calculated.
    true_demand : pd.DataFrame
        contains the demand derived from true structural parameters.
    correlation : pd.DataFrame
        the correlation matrix of the estimated structural parameters across
        all Monte Carlo runs.

    """
    init_dict = {
        "model_specifications": {
            "discount_factor": beta,
            "number_states": number_states,
            "maint_cost_func": "linear",
            "cost_scale": 1e-3,
        },
        "optimizer": {
            "approach": "NFXP",
            "algorithm": "estimagic_bhhh",
            "gradient": "Yes",
        },
        "alg_details": {"threshold": 1e-13, "switch_tol": 1e-2},
    }
    demand_dict = {
        "RC_lower_bound": 2,
        "RC_upper_bound": 13,
        "demand_evaluations": 100,
        "tolerance": 1e-10,
        "num_periods": 12,
        "num_buses": number_buses,
    }

    # get true demand function
    true_params = np.array([0.0937, 0.4475, 0.4459, 0.0127, 0.0002, 11.7257, 2.4569])
    true_demand = get_demand(init_dict, demand_dict, true_params)[0]["demand"].astype(
        float
    )

    # setup loop for demand calculation
    results_beta = (
        results.loc[
            (beta, slice(None), 0, "MPEC"),
            ("RC", "theta_11", "theta_30", "theta_31", "theta_32", "theta_33"),
        ]
        .astype(float)
        .to_numpy()
    )
    rc_range = np.linspace(
        demand_dict["RC_lower_bound"],
        demand_dict["RC_upper_bound"],
        demand_dict["demand_evaluations"],
    )
    demand = pd.DataFrame(index=rc_range)
    demand.index.name = "RC"
    for j in range(len(results_beta)):
        trans_params = results_beta[j, 2:]
        trans_params = np.append(trans_params, 1 - sum(trans_params))
        params = np.concatenate((trans_params, results_beta[j, :2]))
        demand[str(j)] = get_demand(init_dict, demand_dict, params)[0]["demand"]

    # save the data
    demand.to_pickle("data/demand.pickle")

    # get correlation
    results_beta = results.loc[
        (beta, slice(None), 0, "MPEC"),
        ("RC", "theta_11", "theta_30", "theta_31", "theta_32", "theta_33"),
    ].astype(float)
    correlation = results_beta.corr()

    values = [demand, rc_range, true_demand, correlation]
    names = ["demand", "rc_range", "true_demand", "correlation"]
    for value, name in zip(values, names):
        pd.to_pickle(value, f"data/{name}.pickle")

    return demand, rc_range, true_demand, correlation


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
    """
    simulates a single data set with a given specification using the ``simulate``
    function of ruspy.

    Parameters
    ----------
    seed : int
        seed for the simulation function.
    disc_fac : float
        the discount factor in the Rust Model.
    num_buses : int
        the amount of buses that should be simulated.
    num_periods : int
        The number of periods that should be simulated for each bus.
    num_states : int
        the number of states for the which the mileage state is discretized.
    cost_params : np.array
        the cost parameters for which the data is simulated.
    trans_params : np.array
        the cost parameters for which the data is simulated..
    cost_func : callable
        the cost function that underlies the data generating process.
    scale : float
        the scale of the cost function.

    Returns
    -------
    df : pd.DataFrame
        Simulated data set for the given data generating process.

    """
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
    """
    transforms the grid state for lower grid size.

    Parameters
    ----------
    column : pd.Series
        column that contains the discretized mileage of a data set.

    Returns
    -------
    column : pd.Series
        transformed column for state corresping to half of the grid size.

    """
    if column.name == "state":
        column = np.floor(column / 2)
    return column


def process_result(approach, cost_result, alg_nfxp):
    """
    process the raw results from a Monte Carlo simulation run in the
    ``sensitivity_simulation`` function.

    Parameters
    ----------
    approach : string
        indicates whether the raw results were created from the NFXP or MPEC.
    cost_result : dict
        the result dictionairy of ruspy for the cost parameters.

    Returns
    -------
    result : numpy.array
        contains the transformed results of a Monte Carlo simulation run.

    """

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
        if alg_nfxp == "scipy_L-BFGS-B":
            if result[2] == "success":
                result[2] = 1
            else:
                result[2] = 0

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
    """
    calculates the quantitiy of interest for a given estimated parameter vector
    in a certain specification.

    Parameters
    ----------
    init_dict : dict
        dictionairy needed for the estimation procedure which gives info about
        the model specification to the demand function calculation.
    params : np.array
        contains the estimated transition and cost parameters.

    Returns
    -------
    demand : float
        the resulting quantitiy of interest.

    """
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


def sensitivity_simulation(
    specification, number_runs, alg_nfxp, tolerance=None, max_cont=20, max_nk=20
):
    """
    performs a certain number of estimations with certain specifications
    on simulated data.

    Parameters
    ----------
    specification : tuple
        contains the information about which discount factor, cost function,
        grid size, derivative and approach is used for the estimation.
    number_runs : int
        number of runs per specification.
    alg_nfxp : string
        the algorithm used for the NFXP.
    tolerance : dict
        specifies the stopping tolerance for the optimizer of the NFXP.
    max_cont : int
        maximum number of contraction steps for the NFXP.
    max_nk : int
        maximum number of Newton-Kantorovich steps for the NFXP.

    Returns
    -------
    results : pd.DataFrame
        contains results such as likelihood, estimated parameters etc per run.

    """
    # set default tolerance
    if tolerance is None:
        if alg_nfxp == "estimagic_bhhh":
            tolerance = {"tol": {"abs": 1e-05, "rel": 1e-08}}
        elif alg_nfxp == "scipy_L-BFGS-B":
            tolerance = {"gtol": 1e-05}

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
                "algorithm": alg_nfxp,
                "gradient": specification["Analytical Gradient"],
                "algo_options": tolerance,
            },
            "alg_details": {
                "threshold": stopping_crit_fixed_point,
                "switch_tol": switch_tolerance_fixed_point,
                "max_contr_steps": max_cont,
                "max_newt_kant_steps": max_nk,
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
            print(specification, run)
            # Run estimation
            data = data_sets[run]
            try:
                transition_result_nfxp, cost_result_nfxp = estimate(
                    init_dict_nfxp, data
                )
                results.loc[(*indexer, run), (slice("RC", "theta_13"))][
                    : len(cost_result_nfxp["x"])
                ] = cost_result_nfxp["x"]
                results.loc[(*indexer, run), (slice("theta_30", "theta_310"))][
                    : len(transition_result_nfxp["x"])
                ] = transition_result_nfxp["x"]
                results.loc[(*indexer, run), column_slicer_nfxp] = process_result(
                    specification["Approach"], cost_result_nfxp, alg_nfxp
                )
                results.loc[(*indexer, run), "Demand"] = get_qoi(
                    init_dict_nfxp,
                    np.concatenate(
                        (transition_result_nfxp["x"], cost_result_nfxp["x"])
                    ),
                )
            except TypeError:
                results.loc[(*indexer, run), :] = results.shape[1] * np.nan
                results.loc[(*indexer, run), "Converged"] = 0

            results.to_pickle(
                "data/sensitivity/sensitivity_specification_"
                + alg_nfxp
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
                specification["Approach"], cost_result_mpec, alg_nfxp
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


def partial_sensitivity(sensitivity_results, axis, axis_name):
    """
    creates a table with mean and standard deviation of the statistics in the
    sensititvity_results table when changing only one part of the specification,
    namely the axis.

    Parameters
    ----------
    sensitivity_results : pd.DataFrame
        table with all runs of the sensitivity simulation.
    axis : list
        the values of one ingredient of the specification.
    axis_name : string
        the name of the specification part that is supposed to be changed.

    Returns
    -------
    table : pd.DataFrame
        table that contains the mean and standard deviation of some variables
        across NFXP and MPEC when changing one part of the specifications.

    """
    table_temp = (
        sensitivity_results.loc[sensitivity_results["Converged"] == 1]
        .astype(float)
        .groupby(level=[axis_name, "Approach"])
    )
    approaches = ["NFXP", "MPEC"]
    statistics = ["Mean", "Standard Deviation"]
    index = pd.MultiIndex.from_product(
        [axis, approaches, statistics], names=[axis_name, "Approach", "Statistic"]
    )
    table = pd.DataFrame(index=index, columns=sensitivity_results.columns)
    table.loc(axis=0)[:, :, "Mean"] = table_temp.mean()
    table.loc(axis=0)[:, :, "Standard Deviation"] = table_temp.std()
    table_temp = (
        sensitivity_results["Converged"]
        .astype(float)
        .groupby(level=[axis_name, "Approach"])
    )
    table.loc[(slice(None), slice(None), "Mean"), "Converged"] = table_temp.mean()
    table.loc[
        (slice(None), slice(None), "Standard Deviation"), "Converged"
    ] = table_temp.std()
    table = table.astype(float)

    return table


def get_difference_approach(sensitivity_results):
    """
    obtain the averages and standard deviations across all specifications and runs
    for MPEC and NFXP, respectively.

    Parameters
    ----------
    sensitivity_results : pd.DataFrame
        table with all runs of the sensitivity simulation.

    Returns
    -------
    table : pd.DataFrame
        table that contains the means and standard deviations.

    """
    table_temp = (
        sensitivity_results.loc[sensitivity_results["Converged"] == 1]
        .astype(float)
        .groupby(level=["Approach"])
    )
    approaches = ["NFXP", "MPEC"]
    statistics = ["Mean", "Standard Deviation"]
    index = pd.MultiIndex.from_product(
        [approaches, statistics], names=["Approach", "Statistic"]
    )
    table = pd.DataFrame(index=index, columns=sensitivity_results.columns)
    table.loc(axis=0)["MPEC", "Mean"] = table_temp.mean().loc["MPEC"]
    table.loc(axis=0)["NFXP", "Mean"] = table_temp.mean().loc["NFXP"]
    table.loc(axis=0)["MPEC", "Standard Deviation"] = table_temp.std().loc["MPEC"]
    table.loc(axis=0)["NFXP", "Standard Deviation"] = table_temp.std().loc["NFXP"]
    table_temp = (
        sensitivity_results["Converged"].astype(float).groupby(level=["Approach"])
    )
    table.loc[("NFXP", "Mean"), "Converged"] = table_temp.mean().loc["NFXP"]
    table.loc[("NFXP", "Standard Deviation"), "Converged"] = table_temp.std().loc[
        "NFXP"
    ]
    table.loc[("MPEC", "Mean"), "Converged"] = table_temp.mean().loc["MPEC"]
    table.loc[("MPEC", "Standard Deviation"), "Converged"] = table_temp.std().loc[
        "MPEC"
    ]
    table = table.astype(float)

    return table


def get_specific_sensitivity(sensitivity_results, specifications):
    """
    get mean and standard deviations for small pertubations in model
    specification and numerical approach.

    Parameters
    ----------
    sensitivity_results : pd.DataFrame
        table with all runs of the sensitivity simulation.
    specifications : list
        contains the model specifications for which the means and standard
        deviations are calculated.

    Returns
    -------
    sensitivity_results_new : pd.DataFrame
        contains the results per run for the given specifications.
    table : pd.DataFrame
        contains the means and standard devaitions across different variables.

    """
    indexes = []
    original_index = sensitivity_results.index
    for index in original_index:
        if list(index[:4]) in specifications:
            indexes.append(index)

    sensitivity_results_new = sensitivity_results.loc[indexes, :]

    index_table = []
    for spec in np.arange(int(len(indexes) / 250)):
        temp_index = list(indexes[250 * spec][:5])
        for statistic in [["Mean"], ["Standard Deviation"]]:
            temp = temp_index.copy()
            temp.extend(statistic)
            index_table.append(tuple(temp))

    index_table = pd.MultiIndex.from_tuples(index_table)
    table = pd.DataFrame(index=index_table, columns=sensitivity_results.columns)

    for spec in np.arange(int(len(indexes) / 250)):
        index = indexes[250 * spec : 250 * (spec + 1)]
        index_table = index[0][:5]
        temp_results = sensitivity_results.reindex(index)
        temp_results = temp_results.loc[temp_results["Converged"] == 1]
        table.loc[(*index_table, "Mean"), :] = temp_results.mean()
        table.loc[(*index_table, "Standard Deviation"), :] = temp_results.std()

    table = table.astype(float)

    return sensitivity_results_new, table


def get_difference_approach_per_run(sensitivity_results):
    """
    obtain the averages and standard deviations across all specifications per run
    for MPEC and NFXP, respectively.

    Parameters
    ----------
    sensitivity_results : pd.DataFrame
        table with all runs of the sensitivity simulation.

    Returns
    -------
    table : pd.DataFrame
        table that contains the means and standard deviations.

    """
    table_temp = (
        sensitivity_results.loc[sensitivity_results["Converged"] == 1]
        .astype(float)
        .groupby(level=["Run", "Approach"])
    )
    runs = np.arange(250)
    approaches = ["NFXP", "MPEC"]
    statistics = ["Mean", "Standard Deviation"]
    index = pd.MultiIndex.from_product(
        [runs, approaches, statistics], names=["Run", "Approach", "Statistic"]
    )
    table = pd.DataFrame(index=index, columns=sensitivity_results.columns)
    table.loc(axis=0)[:, :, "Mean"] = table_temp.mean()
    table.loc(axis=0)[:, :, "Standard Deviation"] = table_temp.std()
    table_temp = (
        sensitivity_results["Converged"].astype(float).groupby(level=["Approach"])
    )
    table.loc[(slice(None), slice(None), "Mean"), "Converged"] = table_temp.mean()
    table.loc[
        (slice(None), slice(None), "Standard Deviation"), "Converged"
    ] = table_temp.std()
    table = table.astype(float)

    return table


def get_extensive_specific_sensitivity(sensitivity_results, specifications):
    """
    get mean, standard deviations and confidence interval for pertubations in model
    specification and numerical approach.

    Parameters
    ----------
    sensitivity_results : pd.DataFrame
        table with all runs of the sensitivity simulation.
    specifications : list
        contains the model specifications for which the means and standard
        deviations are calculated.

    Returns
    -------
    sensitivity_results_new : pd.DataFrame
        the simulation results of the specificatiosn supplied.
    table : pd.DataFrame
        contains mean, standard deviation and confidence interval per specification.

    """
    original_index = sensitivity_results.index
    ordered_indexes = [[] for _ in range(len(specifications))]
    for index in original_index:
        for order, spec in enumerate(specifications):
            if list(index[:4]) == spec:
                ordered_indexes[order].append(index)

    indexes = []
    for index in ordered_indexes:
        for order in np.arange(len(ordered_indexes[0])):
            indexes.append(index[order])

    sensitivity_results_new = sensitivity_results.loc[indexes, :]

    index_table = []
    for spec in np.arange(int(len(indexes) / 250)):
        temp_index = list(indexes[250 * spec][:5])
        for statistic in [
            ["Mean"],
            ["Upper SD"],
            ["Lower SD"],
            ["Upper Percentile"],
            ["Lower Percentile"],
            ["MSE"],
        ]:
            temp = temp_index.copy()
            temp.extend(statistic)
            index_table.append(tuple(temp))

    names = list(sensitivity_results.index.names)
    names[-1] = "Statistic"
    index_table = pd.MultiIndex.from_tuples(index_table, names=names)
    table = pd.DataFrame(index=index_table, columns=sensitivity_results.columns)

    for spec in np.arange(int(len(indexes) / 250)):
        index = indexes[250 * spec : 250 * (spec + 1)]
        index_table = index[0][:5]
        temp_results_0 = sensitivity_results.reindex(index)
        temp_results = temp_results_0.loc[temp_results_0["Converged"] == 1]
        table.loc[(*index_table, "Mean"), :] = temp_results.mean()
        table.loc[(*index_table, "Upper SD"), :] = (
            temp_results.mean() + temp_results.std()
        )
        table.loc[(*index_table, "Lower SD"), :] = (
            temp_results.mean() - temp_results.std()
        )
        table.loc[(*index_table, "Upper Percentile"), :] = temp_results.apply(
            lambda x: np.percentile(x, 97.5, axis=0)
        )
        table.loc[(*index_table, "Lower Percentile"), :] = temp_results.apply(
            lambda x: np.percentile(x, 2.25, axis=0)
        )
        table.loc[(*index_table, "MSE"), "Demand"] = (
            temp_results["Demand"]
            .to_frame()
            .apply(lambda x: ((x - 11.095184215630066) ** 2).mean())
            .to_numpy()[0]
        )
        table.loc[(*index_table, "Mean"), "Converged"] = temp_results_0[
            "Converged"
        ].mean()

    table = table.astype(float)

    return sensitivity_results_new, table
