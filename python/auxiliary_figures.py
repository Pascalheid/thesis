"""
Contains all the functions needed to obtain the figures in my thesis.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_figure_2(results, beta):
    """
    Creates and saves Figure 2 of my thesis.

    Parameters
    ----------
    results : pd.DataFrame
        the results of the Monte Carlo simulation in Iskhakov et al. (2016)
        created by ``get_iskhakov_results``.
    beta : float
        indicates for which of the beta the demand function is supposed to be
        derived.

    Returns
    -------
    None.

    """
    results_beta = results.loc[
        (beta, slice(None), slice(None), "MPEC"), ("RC", "theta_11")
    ]
    figure_2 = sns.jointplot("theta_11", "RC", results_beta, kind="hex")
    figure_2.set_axis_labels(r"$\hat\theta_{11}$", r"$\hat{RC}$")
    figure_2.savefig("figures/figure_2.png", dpi=1000)


def get_figure_3_and_4(demand, rc_range, true_demand):
    """
    Creates and saves Figures 3 and 4.

    Parameters
    ----------
    demand : pd.DataFrame
        contains the demand over a range of replacement costs depending on
        the estimated structural parameters of a Monte Carlo run.
    rc_range : np.array
        range over which the demand is calculated.
    true_demand : pd.DataFrame
        contains the demand derived from true structural parameters.

    Returns
    -------
    None.

    """
    # transform demand data into plot
    data = demand.astype(float).to_numpy()
    mean = data.mean(axis=1)
    std = data.std(axis=1)
    std_upper_bound = mean + std
    std_lower_bound = mean - std
    lower_percentile = np.percentile(data, 2.5, axis=1)
    upper_percentile = np.percentile(data, 97.5, axis=1)
    ci_lower_bound = 2 * mean - upper_percentile
    ci_upper_bound = 2 * mean - lower_percentile

    # plot demand function and its uncertainty, Figure 4
    fig, axis = plt.subplots()
    axis.plot(rc_range, true_demand.to_numpy(), color="black", label="True Demand")
    axis.plot(rc_range, mean, color=sns.color_palette("Blues")[5], label="Mean")
    axis.plot(rc_range, std_lower_bound, color=sns.color_palette("Blues")[3])
    axis.plot(
        rc_range,
        std_upper_bound,
        color=sns.color_palette("Blues")[3],
        label=r"$\sigma$-Band",
    )
    axis.plot(rc_range, ci_lower_bound, color=sns.color_palette("Blues")[1])
    axis.plot(
        rc_range,
        ci_upper_bound,
        color=sns.color_palette("Blues")[1],
        label="95% Confidence Band",
    )
    axis.set_xlim([4, 13])
    axis.set_ylim([0, 30])
    axis.set_xlabel("Replacement Cost in Thousands")
    axis.set_ylabel("Expected Annual Engine Replacement")
    axis.legend()
    # axis.fill_between(rc_range, upper_bound, lower_bound, color="0.5")
    plt.savefig("figures/figure_4.png", dpi=1000)

    # plot distribution of QoI, Figure 3
    data = demand.loc[11.0, :].astype(float).to_numpy()
    true_data = true_demand.loc[11.0]
    mean = data.mean()
    std = data.std()
    lower_percentile = np.percentile(data, 2.5)
    upper_percentile = np.percentile(data, 97.5)

    fig, ax = plt.subplots()
    sns.distplot(data, ax=ax)
    ax.set_xlabel(r"$Y$")
    ax.set_ylabel(r"$f_Y$")
    ylim = ax.get_ylim()
    ax.vlines(
        true_data, *ylim, colors=sns.light_palette("black")[4], label="True Value"
    )
    ax.vlines(mean, *ylim, colors=sns.light_palette("black")[3], label="Mean")
    ax.vlines(
        mean + std,
        *ylim,
        colors=sns.light_palette("black")[2],
        label=r"$\sigma$-interval"
    )
    ax.vlines(mean - std, *ylim, colors=sns.light_palette("black")[2])
    ax.vlines(
        upper_percentile,
        *ylim,
        colors=sns.light_palette("black")[1],
        label="95% Confidence \nInterval"
    )
    ax.vlines(lower_percentile, *ylim, colors=sns.light_palette("black")[1])

    ax.legend()
    plt.savefig("figures/figure_3.png", dpi=1000)
