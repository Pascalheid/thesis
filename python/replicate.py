"""
Running this module yields all the tables and figures that can be found in
my thesis.
"""
import numpy as np
import pandas as pd

from python.auxiliary import get_iskhakov_results
from python.auxiliary import simulate_figure_3_and_4
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
# if you want to run the simulation
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

# simulate the demand function over the Monte Carlo runs and
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
