import numpy as np

from python.auxiliary import get_iskhakov_results
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

# create table 1
table_1 = get_table_1(results, number_runs, starting_cost_params)

# get table 2
my_results_table_2, iskhakov_table_2 = get_table_2(results, discount_factor, approach)
