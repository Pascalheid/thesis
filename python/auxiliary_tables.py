"""This module contains functions to create the tables in the thesis."""

import numpy as np
import pandas as pd

import scipy.io

def get_table_1(results, number_runs, starting_cost_params):
    columns_table_1 = ["CPU Time", "Converged", "# of Major Iter.", "# of Func. Eval.", "# of Bellm. Iter.", "# of N-K Iter."]
    table_1 = results[columns_table_1].astype(float).groupby(["Discount Factor", "Approach"]).mean()
    table_1["Converged"] = (table_1["Converged"]*number_runs*starting_cost_params.shape[1]).astype(int)
    table_1.astype(float).round(3)
    
    return table_1


def get_table_2(results, discount_factor, approach):
    # Create Table I from Su & Judd (2012) with the simulated values from Iskahkov et al. (2016)
    columns_table = ["RC", "theta_11"]
    table_temp = results.loc[results["Converged"] == 1, columns_table].astype(float).groupby(
        level=["Discount Factor", "Approach"])
    
    statistic = ["Mean", "Standard Deviation"]
    index = pd.MultiIndex.from_product([discount_factor, approach, statistic],
                                       names=["Discount Factor", "Approach", "Statistic"])
    table = pd.DataFrame(index=index, columns=columns_table)
    table.loc(axis=0)[:,:,"Mean"] = table_temp.mean()
    table.loc(axis=0)[:,:,"Standard Deviation"] = table_temp.std()
    table.astype(float).round(3)
    
    # Process the results of Iskhakov et al. (2016) created by their original 
    # matlab code
    index = pd.MultiIndex.from_product([discount_factor, statistic],
                                       names=["Discount Factor", "Statistic"])
    NFXP_Iskhakov = pd.DataFrame(index=index, columns=columns_table)
    for factor in discount_factor:
        NFXP_Iskhakov_temp = scipy.io.loadmat(
            "data/solution_iskhakov_beta_" + str(int(100000*factor)))[
            "result_jr87_" + str(int(100000*factor))]
        NFXP_Iskhakov.loc[factor, "Mean"] = NFXP_Iskhakov_temp.mean(axis=0)
        NFXP_Iskhakov.loc[factor, "Standard Deviation"] = NFXP_Iskhakov_temp.std(axis=0)    
    NFXP_Iskhakov.astype(float).round(3)
    
    return table, NFXP_Iskhakov