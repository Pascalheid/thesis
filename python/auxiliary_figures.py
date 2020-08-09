"""
Contains all the functions needed to obtain the figures in my thesis.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


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
    axis.plot(rc_range, lower_percentile, color=sns.color_palette("Blues")[1])
    axis.plot(
        rc_range,
        upper_percentile,
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

    ax.set_yticklabels([])
    ax.legend()
    plt.savefig("figures/figure_3.png", dpi=1000)


def get_sensitivity_figure(
    sensitivity_table, specifications, labels, figure_name, legend=False
):
    """
    creates Figures x to y in the paper.

    Parameters
    ----------
    sensitivity_table : pd.DataFrame
        the sensitivity_results_new of the ``get_extensive_specific_sensitivity``
        function for a given specification.
    specifications : list
        list of lists that consist the specifications of interest.
    labels : list
        list of strings that are the labels for the x axis.
    figure_name : string
        how to name the figure when saving it.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={"wspace": 0.1})
    for axis, approach in enumerate(["NFXP", "MPEC"]):
        ax[axis].scatter(
            np.arange(len(specifications)),
            sensitivity_table.loc[
                (slice(None), slice(None), slice(None), slice(None), approach, "Mean"),
                "Demand",
            ],
            color=sns.color_palette("Blues")[5],
            marker="_",
            s=150,
            label="Mean",
            zorder=1,
        )
        for statistic in [
            ("Lower SD", "Upper SD", 3, 0.05, r"$\sigma$-Interval"),
            ("Lower Percentile", "Upper Percentile", 1, -0.05, "95% CI"),
        ]:
            ax[axis].scatter(
                np.arange(len(specifications)) + statistic[3],
                sensitivity_table.loc[
                    (
                        slice(None),
                        slice(None),
                        slice(None),
                        slice(None),
                        approach,
                        statistic[0],
                    ),
                    "Demand",
                ],
                color=sns.color_palette("Blues")[statistic[2]],
                marker="_",
                zorder=0,
            )
            ax[axis].scatter(
                np.arange(len(specifications)) + statistic[3],
                sensitivity_table.loc[
                    (
                        slice(None),
                        slice(None),
                        slice(None),
                        slice(None),
                        approach,
                        statistic[1],
                    ),
                    "Demand",
                ],
                color=sns.color_palette("Blues")[statistic[2]],
                marker="_",
                zorder=0,
            )

            ax[axis].vlines(
                np.arange(len(specifications)) + statistic[3],
                sensitivity_table.loc[
                    (
                        slice(None),
                        slice(None),
                        slice(None),
                        slice(None),
                        approach,
                        statistic[1],
                    ),
                    "Demand",
                ],
                sensitivity_table.loc[
                    (
                        slice(None),
                        slice(None),
                        slice(None),
                        slice(None),
                        approach,
                        statistic[0],
                    ),
                    "Demand",
                ],
                color=sns.color_palette("Blues")[statistic[2]],
                zorder=0,
                label=statistic[4],
            )
        ax[axis].set_title(approach)
        ax[axis].axhline(
            y=11.0952, c="black", linewidth=0.5, linestyle="--", label="True Demand"
        )
        ax[axis].set_xticks(np.arange(len(specifications)))
        ax[axis].set_xticklabels(labels)
        ax[0].set_ylabel("Demand")
        if legend is True:
            ax[1].legend()
        if figure_name == "9":
            ax[0].set_xlabel("Specification")
            ax[0].set_xticklabels(["4", "22", "28", "34", "35", "33", "31"])

    plt.savefig("figures/figure_" + figure_name + ".png", dpi=1000)


def get_sensitivity_figure_single(
    sensitivity_table, specifications, labels, figure_name, approach, legend=False
):
    """
    creates Figures x to y in the paper.

    Parameters
    ----------
    sensitivity_table : pd.DataFrame
        the sensitivity_results_new of the ``get_extensive_specific_sensitivity``
        function for a given specification.
    specifications : list
        list of lists that consist the specifications of interest.
    labels : list
        list of strings that are the labels for the x axis.
    figure_name : string
        how to name the figure when saving it.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots()
    ax.scatter(
        np.arange(len(specifications)),
        sensitivity_table.loc[
            (slice(None), slice(None), slice(None), slice(None), approach, "Mean"),
            "Demand",
        ],
        color=sns.color_palette("Blues")[5],
        marker="_",
        s=150,
        label="Mean",
        zorder=1,
    )
    for statistic in [
        ("Lower SD", "Upper SD", 3, 0.05, r"$\sigma$-Interval"),
        ("Lower Percentile", "Upper Percentile", 1, -0.05, "95% CI"),
    ]:
        ax.scatter(
            np.arange(len(specifications)) + statistic[3],
            sensitivity_table.loc[
                (
                    slice(None),
                    slice(None),
                    slice(None),
                    slice(None),
                    approach,
                    statistic[0],
                ),
                "Demand",
            ],
            color=sns.color_palette("Blues")[statistic[2]],
            marker="_",
            zorder=0,
        )
        ax.scatter(
            np.arange(len(specifications)) + statistic[3],
            sensitivity_table.loc[
                (
                    slice(None),
                    slice(None),
                    slice(None),
                    slice(None),
                    approach,
                    statistic[1],
                ),
                "Demand",
            ],
            color=sns.color_palette("Blues")[statistic[2]],
            marker="_",
            zorder=0,
        )

        ax.vlines(
            np.arange(len(specifications)) + statistic[3],
            sensitivity_table.loc[
                (
                    slice(None),
                    slice(None),
                    slice(None),
                    slice(None),
                    approach,
                    statistic[1],
                ),
                "Demand",
            ],
            sensitivity_table.loc[
                (
                    slice(None),
                    slice(None),
                    slice(None),
                    slice(None),
                    approach,
                    statistic[0],
                ),
                "Demand",
            ],
            color=sns.color_palette("Blues")[statistic[2]],
            zorder=0,
            label=statistic[4],
        )
    ax.set_title(approach)
    ax.axhline(y=11.0952, c="black", linewidth=0.5, linestyle="--", label="True Demand")
    ax.set_xticks(np.arange(len(specifications)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Demand")
    if legend is True:
        ax.legend()
    plt.savefig("figures/figure_" + figure_name + ".png", dpi=1000)


def get_sensitivity_density(
    sensitivity_results, approach, cumulative, figure_name, specification_range, mark=()
):
    """
    plots the density function or cdf of the quantity of interest across
    a range of specifications.

    Parameters
    ----------
    sensitivity_results : pd.DataFrame
        table with all runs of the sensitivity simulation.
    approach : list
        list of strings for which approach(es) the distribution is shown.
    cumulative : bool
        indicates whether the cdf instead of the densitiy is plotted.
    mark : list
        list of integers that describes the specification for which a darker
        color should be used.
    figure_name : str
        name of the figure for saving.
    specification_range : list
        numbers of specifications for which the distribution is shown.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots()
    for spec in specification_range:
        temp = sensitivity_results[250 * spec : 250 * (spec + 1)]
        if temp.index[0][-2] in approach:
            if list(temp.index[0][:4]) == [0.975, "linear", 400, "Yes"]:
                color = sns.color_palette("Blues")[5]
                zorder = 5
            elif spec in mark:
                color = sns.color_palette("Blues")[2]
                zorder = 5
            else:
                color = sns.color_palette("Blues")[0]
                zorder = 1
            sns.kdeplot(
                temp["Demand"], color=color, zorder=zorder, cumulative=cumulative
            )
    ax.set_xlabel(r"$Y$")
    if cumulative is True:
        ax.set_ylabel(r"$F_Y$")
        ax.set
    else:
        ax.set_ylabel(r"$f_Y$")
        ax.set_yticklabels([])

    legend = [Line2D([0], [0], color=sns.color_palette("Blues")[5], lw=2)]
    ax.legend(legend, ["correctly specified"])
    plt.savefig("figures/" + figure_name + ".png", dpi=1000)


def get_sensitivity_density_both(
    sensitivity_results, approach, figure_name, specification_range, mark=()
):
    """
    plots the density function and cdf of the quantity of interest across
    a range of specifications.

    Parameters
    ----------
    sensitivity_results : pd.DataFrame
        table with all runs of the sensitivity simulation.
    approach : list
        list of strings for which approach(es) the distribution is shown.
    mark : list
        list of integers that describes the specification for which a darker
        color should be used.
    figure_name : str
        name of the figure for saving.
    specification_range : list
        numbers of specifications for which the distribution is shown.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(2, 1)
    for number, cumulative in enumerate([False, True]):
        for spec in specification_range:
            temp = sensitivity_results[250 * spec : 250 * (spec + 1)]
            if temp.index[0][-2] in approach:
                if list(temp.index[0][:4]) == [0.975, "linear", 400, "Yes"]:
                    color = sns.color_palette("Blues")[5]
                    zorder = 5
                elif spec in mark:
                    color = sns.color_palette("Blues")[2]
                    zorder = 5
                else:
                    color = sns.color_palette("Blues")[0]
                    zorder = 1
                sns.kdeplot(
                    temp["Demand"],
                    color=color,
                    zorder=zorder,
                    cumulative=cumulative,
                    ax=ax[number],
                    legend=False,
                )
        ax[number].set_xlabel(r"$Y$")
        if cumulative is True:
            ax[number].set_ylabel(r"$F_Y$")
        else:
            ax[number].set_ylabel(r"$f_Y$")
            ax[number].set_yticklabels([])

        legend = [
            Line2D([0], [0], color=sns.color_palette("Blues")[5], lw=2),
            Line2D([0], [0], color="black", lw=0.5, linestyle="--"),
        ]
        ax[number].axvline(
            x=11.0952, c="black", linewidth=0.5, linestyle="--", label="True Demand"
        )
        ax[0].legend(legend, ["Correctly specified", "True Demand"])

    plt.savefig("figures/" + figure_name + ".png", dpi=1000)


def get_mse_figure(sensitivity_table, specifications, figure_name):
    """
    plots the mean squared error of the demand QoI on the specification
    of the simulation run per approach.

    Parameters
    ----------
    sensitivity_table : pd.DataFrame
        the sensitivity_results_new of the ``get_extensive_specific_sensitivity``
        function for a given specification.
    specifications : list
        list of lists that consist the specifications of interest.
    figure_name : string
        how to name the figure when saving it.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots()
    for approach in [("MPEC", 5), ("NFXP", 3)]:
        ax.scatter(
            np.arange(len(specifications)),
            sensitivity_table.loc[
                (
                    slice(None),
                    slice(None),
                    slice(None),
                    slice(None),
                    approach[0],
                    "MSE",
                ),
                "Demand",
            ],
            color=sns.color_palette("Blues")[approach[1]],
            marker=".",
            # s=150,
            label=approach[0],
            zorder=approach[1],
        )
    ax.set_ylabel("Mean Squared Error of QoI")
    ax.set_xlabel("Specification")
    for spec in np.arange(0, 36, 5):
        ax.axvline(spec, color="black", linestyle="--", linewidth=0.5)
    ax.legend()
    plt.savefig("figures/" + figure_name + ".png", dpi=1000)
