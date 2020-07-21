import numpy as np
import pandas as pd
import scipy.io

from ruspy.estimation.estimation import estimate


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
    
    # Initialize the set up for the nested fixed point algorithm
    stopping_crit_fixed_point = 1e-13
    switch_tolerance_fixed_point = 1e-2

    # Initialize the set up for MPEC
    lower_bound = np.concatenate((np.full(number_states, -np.inf), np.full(number_cost_params, 0.0)))
    upper_bound = np.concatenate((np.full(number_states, 500.0), np.full(number_cost_params, np.inf)))
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
            # implies that we use analytical first order derivatives as opposed to numerical ones
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
            # implies that we use analytical first order derivatives as opposed to numerical ones
            "gradient": "Yes",
            "tol": rel_ipopt_stopping_tolerance,
            "set_lower_bounds": lower_bound,
            "set_upper_bounds": upper_bound,
        },
    }

    # Initialize DataFrame to store the results of each run of the Monte Carlo simulation
    index = pd.MultiIndex.from_product([discount_factor,
                                        range(number_runs),
                                        range(starting_cost_params.shape[1]),
                                        approach],
                                       names=["Discount Factor", "Run", "Start", "Approach"])

    columns=["RC", "theta_11", "theta_30", "theta_31", "theta_32", "theta_33",
             "CPU Time", "Converged", "# of Major Iter.", "# of Func. Eval.",
             "# of Bellm. Iter.", "# of N-K Iter."]

    results = pd.DataFrame(index=index, columns=columns)

    # Main loop to calculate the results for each run
    for factor in discount_factor:
        # load simulated data
        mat = scipy.io.loadmat("data/RustBusTableXSimDataMC250_beta" + str(int(100000*factor)))

        for run in range(number_runs):
            data = process_data(mat, run, number_buses, number_periods)

            for start in range(starting_cost_params.shape[1]):
                # Adapt the Initiation Dictionairy of NFXP for this run
                init_dict_nfxp["model_specifications"]["discount_factor"] = factor
                init_dict_nfxp["optimizer"]["params"] = pd.DataFrame(starting_cost_params[:, start], columns=["value"])

                # Run NFXP using ruspy
                transition_result_nfxp, cost_result_nfxp = estimate(init_dict_nfxp, data)

                # store the results of this run
                results.loc[factor, run, start, "NFXP"] = process_result(
                    "NFXP", transition_result_nfxp, cost_result_nfxp, number_states)

                # Adapt the Initiation Dictionairy of MPEC for this run
                init_dict_mpec["model_specifications"]["discount_factor"] = factor
                init_dict_mpec["optimizer"]["params"] = np.concatenate((
                    starting_expected_value_fun, starting_cost_params[:, start]))

                # Run MPEC using ruspy
                transition_result_mpec, cost_result_mpec = estimate(init_dict_mpec, data)

                # store the results of this run
                results.loc[
                    factor, run, start, "MPEC"].loc[
                    ~results.columns.isin(["# of Bellm. Iter.", "# of N-K Iter."])] = process_result(
                            "MPEC", transition_result_mpec, cost_result_mpec, number_states)
    
    return results


def process_data(df, run, number_buses, number_periods):
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
