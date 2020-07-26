"""
Contains all functions that are needed in intermediary steps in order to obtain
certain tables and figures of the thesis.
"""
import numpy as np
import pandas as pd
import scipy.io
from ruspy.estimation.estimation import estimate
from ruspy.model_code.demand_function import get_demand


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
        (np.full(number_states, 500.0), np.full(number_cost_params, np.inf))
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
                results.loc[factor, run, start, "NFXP"] = process_result(
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
                ] = process_result(
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


def process_result(approach, transition_result, cost_result, number_states):
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
