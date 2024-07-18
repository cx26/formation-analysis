import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
import random
import math
import kneebow 
from kneebow.rotor import Rotor
    
def legend_without_duplicate_labels(ax):
    """
    Update the legend of a matplotlib Axes object to only display unique labels.
    
    This function modifies the legend of the provided Axes `ax` by removing any duplicate
    labels, ensuring that each label appears only once in the legend, which helps in
    maintaining clarity and reducing visual clutter in plots with overlapping legend entries.
    """
    
    # Get current legend handles and labels.
    handles, labels = ax.get_legend_handles_labels()
    
    # Remove duplicate labels while preserving order.
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    
    # Apply the unique handles and labels to the legend.
    ax.legend(*zip(*unique))

def get_r_cycle(hppc_data, hppc_cycle):
    """
    Extracts resistance-related features from a specific HPPC cycle.

    Parameters:
    hppc_data (DataFrame): DataFrame containing HPPC test data.
    hppc_cycle (int): The specific HPPC cycle from which to extract data.

    Returns:
    DataFrame: DataFrame containing resistance calculations for different steps and test times within the HPPC cycle.
    """

    df_r_cycle = pd.DataFrame()
    df_r_cycle['cycle_index'] = [hppc_cycle]
    
    # Select data for the specified HPPC cycle
    chosen_cycle = hppc_data[hppc_data.cycle_index == hppc_cycle]
    steps = hppc_data.step_index.unique()[1:6]
    counters = []
    
    # Collect unique step counters for relevant steps
    for step in steps:
        idx_lst = chosen_cycle[chosen_cycle.step_index == step].step_index_counter.unique()
        idx_lst = idx_lst[idx_lst > 2]  
        counters.append(idx_lst.tolist())
    
    # Define different steps (charge, discharge and rest) in the cycle
    long_rest = chosen_cycle[chosen_cycle.step_index == steps[0]]
    time = long_rest.test_time.iloc[0]
    chosen_cycle = chosen_cycle[chosen_cycle.test_time >= time]
    long_rest = chosen_cycle[chosen_cycle.step_index == steps[0]]
    short_rest = chosen_cycle[chosen_cycle.step_index == steps[2]]
    charge = chosen_cycle[chosen_cycle.step_index == steps[3]]
    discharge = chosen_cycle[chosen_cycle.step_index == steps[1]]
    
    # Calculate resistance values at various time scales for each state of charge
    for i in range(len(counters[1])):
        soc = i
        long_rest_end = long_rest[long_rest.step_index_counter == counters[0][soc]]['voltage'].iloc[-1]
        short_rest_end = short_rest[short_rest.step_index_counter == counters[2][soc]]['voltage'].iloc[-1]
        charge_soc = charge[charge.step_index_counter == counters[3][soc]]
        discharge_soc = discharge[discharge.step_index_counter == counters[1][soc]]

        for idx in [0, 3, 10, 30]:
            charge_chosen = charge_soc[charge_soc.test_time - charge_soc.test_time.min() <= idx]
            discharge_chosen = discharge_soc[discharge_soc.test_time - discharge_soc.test_time.min() <= idx]
            charge_end = charge_chosen.iloc[-1]
            discharge_end = discharge_chosen.iloc[-1]
            r_ch = (charge_end.voltage - short_rest_end) / charge.current.mean()
            r_disch = (discharge_end.voltage - long_rest_end) / discharge.current.mean()
            
            # Check if the pulse duration was fully executed
            if charge_end.test_time - charge_chosen.test_time.min() < idx - 1:
                r_ch = None
            if discharge_end.test_time - discharge_chosen.test_time.min() < idx - 1:
                r_disch = None

            df_r_cycle['r_c_' + str(i) + '_' + str(idx) + 's'] = [r_ch]
            df_r_cycle['r_d_' + str(i) + '_' + str(idx) + 's'] = [r_disch]

    return df_r_cycle

def get_df_r(diag_data):
    """
    Aggregates resistance features from multiple HPPC cycles into a single DataFrame.

    Parameters:
    diag_data (DataFrame): DataFrame containing diagnostic cycling data.

    Returns:
    DataFrame: Aggregated DataFrame of resistance features across multiple cycles.
    """
    hppc_data = diag_data[diag_data.cycle_type == 'hppc']
    hppc_cycles = hppc_data.cycle_index.unique()
    df_r_0 = get_r_cycle(hppc_data, 0)
    results = [df_r_0]  # Initialize the list with the first cycle's data

    for hppc_cycle in hppc_cycles[1:]:
        df_r_cycle = get_r_cycle(hppc_data, hppc_cycle)
        df_r_diff = df_r_cycle.subtract(df_r_0, fill_value=None)
        results.append(df_r_diff)  # Append each DataFrame to the list

    df_r = pd.concat(results, ignore_index=True)  # Perform a single concatenation
    return df_r
    
def get_rpt_summary(diag_summary, regu_summary, diag_pos, seq_num):
    """
    Constructs a summary DataFrame for various energy, capacity, and coulombic efficiency metrics from diagnostic and regular cycling data.

    Parameters:
    diag_summary (DataFrame): Contains diagnostic cycle data including energy and capacity metrics for different rates.
    regu_summary (DataFrame): Contains regular cycling data including energy, capacity, and coulombic efficiency metrics.
    diag_pos (str or int): Diagnostic position, used to select specific cycles or conditions.
    seq_num (int): Sequence number, typically used for tracking or identifying data batches.

    Returns:
    DataFrame: A single-row DataFrame with summary metrics based on the provided diagnostic position.
    """
    rpt_low = diag_summary[diag_summary.cycle_type == 'rpt_0.05C']
    rpt_med = diag_summary[diag_summary.cycle_type == 'rpt_0.2C']
    rpt_low_cycles = rpt_low.cycle_index.unique()
    
    # Initialize DataFrame with specified metrics and one row.
    df = pd.DataFrame(columns=['rpt_low_energy', 'rpt_med_energy', 'regu_energy', 'rpt_low_cap', 
                               'rpt_med_cap', 'regu_cap', 'rpt_low_cv_t', 'rpt_med_cv_t', 'regu_cv_t', 
                               'rpt_low_ce', 'rpt_med_ce', 'regu_ce', 'seq_num', 'diag_pos', 'cycle_index'], 
                      index=range(1))
    
    if diag_pos == 'hppc_1':
        df['seq_num'] = seq_num
        df['diag_pos'] = diag_pos
        regu_cycle = 9  # Hardcoded cycle index
        df['cycle_index'] = 8  # Hardcoded cycle index for consistency
        df['regu_energy'] = regu_summary[regu_summary.cycle_index == regu_cycle]['discharge_energy'].iloc[0]
        df['regu_cap'] = regu_summary[regu_summary.cycle_index == regu_cycle]['discharge_capacity'].iloc[0]
        df['regu_cv_t'] = regu_summary[regu_summary.cycle_index == regu_cycle]['CV_time'].iloc[0]
        df['regu_ce'] = regu_summary[regu_summary.cycle_index == regu_cycle]['discharge_capacity'].iloc[0] / \
                        regu_summary[regu_summary.cycle_index == regu_cycle]['charge_capacity'].iloc[0]
        return df
    
    else:
        if diag_pos == 0:
            regu_cycle = rpt_low_cycles[diag_pos] + 2  # Compute regular cycle based on position
        else:
            regu_cycle = rpt_low_cycles[diag_pos] - 2

        # Fetch metrics from diagnostic summaries and regular summaries based on computed or selected cycle.
        df['rpt_low_energy'] = rpt_low['discharge_energy'].iloc[diag_pos]
        df['rpt_med_energy'] = rpt_med['discharge_energy'].iloc[diag_pos]
        df['regu_energy'] = regu_summary[regu_summary.cycle_index == regu_cycle]['discharge_energy'].iloc[0]
        df['rpt_low_cap'] = rpt_low['discharge_capacity'].iloc[diag_pos]
        df['rpt_med_cap'] = rpt_med['discharge_capacity'].iloc[diag_pos]
        df['regu_cap'] = regu_summary[regu_summary.cycle_index == regu_cycle]['discharge_capacity'].iloc[0]
        df['rpt_low_cv_t'] = rpt_low['CV_time'].iloc[diag_pos]
        df['rpt_med_cv_t'] = rpt_med['CV_time'].iloc[diag_pos]
        df['regu_cv_t'] = regu_summary[regu_summary.cycle_index == regu_cycle]['CV_time'].iloc[0]
        df['regu_ce'] = regu_summary[regu_summary.cycle_index == regu_cycle]['discharge_capacity'].iloc[0] / \
                        regu_summary[regu_summary.cycle_index == regu_cycle]['charge_capacity'].iloc[0]
        df['rpt_low_ce'] = rpt_low['coulombic_efficiency'].iloc[diag_pos]
        df['rpt_med_ce'] = rpt_med['coulombic_efficiency'].iloc[diag_pos]
        df['seq_num'] = seq_num
        df['diag_pos'] = diag_pos
        df['cycle_index'] = rpt_low_cycles[diag_pos] - 1  # Adjust the cycle index for consistency

        return df

def find_knee(diag_summary, regu_summary, regu_life, thresh):
    """
    Identifies the 'knee' point in the discharge capacity curve of battery data, which indicates a significant
    change in battery performance over its life cycle. This function filters the data based on discharge capacity
    and uses interpolation to find the knee point where the degradation accelerates.

    Parameters:
    diag_summary (DataFrame): DataFrame containing diagnostic cycle information.
    regu_summary (DataFrame): DataFrame containing regular cycling data.
    regu_life (int): The last cycle index considered to be within the healthy life span of the battery.
    thresh (float): Threshold for filtering the cycles based on the percentage of previous discharge capacity.

    Returns:
    int or None: The cycle index of the knee point if found, otherwise None.
    """
    new_regu_summary = regu_summary[regu_summary.cycle_index <= regu_life]
    med_cycles = diag_summary[(diag_summary.cycle_type == 'rpt_0.2C') & (diag_summary.cycle_index <= regu_life)].cycle_index.unique()

    for i in range(1, len(med_cycles)):
        pre = regu_summary[(regu_summary.cycle_index < med_cycles[i])].discharge_capacity.iloc[-1]
        if i != len(med_cycles) - 1:
            next_cycle = med_cycles[i + 1]
        else:
            next_cycle = regu_life
        chosen = regu_summary[(regu_summary.cycle_index > med_cycles[i]) & (regu_summary.cycle_index < next_cycle) & (regu_summary.discharge_capacity > (thresh * pre))]
        new_regu_summary = new_regu_summary.drop(labels=chosen.index, axis=0)

    x = new_regu_summary.cycle_index
    y = new_regu_summary.discharge_capacity
    f = scipy.interpolate.interp1d(x, y, fill_value='extrapolate')

    data = pd.DataFrame()
    x_new = range(new_regu_summary.cycle_index.min(), regu_life)
    data['x'] = x_new
    data['y'] = f(x_new)
    data = data.to_numpy()
    rotor = Rotor()
    rotor.fit_rotate(data)
    # rotor.plot_knee()  # Plotting is commented out but can be used for visual verification
    knee_idx = rotor.get_knee_index()

    return x_new[knee_idx]


def get_cycle_life(diag_summary, regu_summary, ax):
    """
    Determines the cycle life and other metrics of a battery based on its discharge capacity and diagnostic summaries.

    Parameters:
    diag_summary (DataFrame): DataFrame containing diagnostic summaries with cycle type details.
    regu_summary (DataFrame): DataFrame containing regular cycling data with discharge capacity, charge throughput, and energy throughput.
    ax (matplotlib.axes.Axes): Axes object for plotting results.

    Returns:
    tuple: Contains various lifecycle metrics such as RPT low life, RPT medium life, regular life, knee point in the cycle life, and throughput metrics.
    """
    
    # Check if minimum discharge capacity is above 80% of maximum, indicating no significant degradation.
    if regu_summary.discharge_capacity.min() > regu_summary.discharge_capacity.max() * 0.8:
        return None, None, None, None, None, None

    else:
        # Identify cycles where discharge capacity is at least 80% of the maximum.
        healthy = regu_summary[regu_summary.discharge_capacity >= regu_summary.discharge_capacity.max() * 0.8]
        regu_life = healthy.cycle_index.iloc[-1]  # Last cycle index considered healthy
        q_throughput = healthy.charge_throughput.iloc[-1]  # Last recorded charge throughput
        e_throughput = healthy.energy_throughput.iloc[-1]  # Last recorded energy throughput
        
        # Plot healthy cycle indices against normalized discharge capacity
        ax.plot(healthy.cycle_index, healthy.discharge_capacity / healthy.discharge_capacity.max() * 100)
        thresh = 0.995
        
        # Find the 'knee' in the cycle life curve
        regu_knee = find_knee(diag_summary, regu_summary, regu_life, thresh)
    
        rpt_low = diag_summary[diag_summary.cycle_type == 'rpt_0.05C']
        rpt_med = diag_summary[diag_summary.cycle_type == 'rpt_0.2C']

        # Check if minimum discharge capacity in medium RPT data is above 80% of its maximum
        if rpt_med.discharge_capacity.min() > rpt_med.discharge_capacity.max() * 0.8:
            return None, None, regu_life, regu_knee, q_throughput, e_throughput
        else:
            eol_med = rpt_med.discharge_capacity.max() * 0.8
            f_med = interp1d(rpt_med.discharge_capacity, rpt_med.cycle_index)
            rpt_med_life = f_med(eol_med)
            if rpt_low.discharge_capacity.min() > rpt_low.discharge_capacity.max() * 0.8:
                return None, rpt_med_life, regu_life, regu_knee, q_throughput, e_throughput
            else:
                eol_low = rpt_low.discharge_capacity.max() * 0.8
                f_low = interp1d(rpt_low.discharge_capacity, rpt_low.cycle_index)
                rpt_low_life = f_low(eol_low)
                return rpt_low_life, rpt_med_life, regu_life, regu_knee, q_throughput, e_throughput

            
            
def combine_one_time_features(diag_data, diag_summary, regu_summary, ax, seq_num):
    """
    Combines various one-time features related to the cycle life and operational characteristics of a battery into a single DataFrame.

    Parameters:
    diag_data (DataFrame): DataFrame containing detailed diagnostic data.
    diag_summary (DataFrame): DataFrame summarizing diagnostic cycle results.
    regu_summary (DataFrame): DataFrame summarizing regular cycle results.
    ax (matplotlib.axes.Axes): Axes object for plotting, if needed within called functions.
    seq_num (int): A sequence number identifying the specific batch or set of battery test data.

    Returns:
    DataFrame: A DataFrame containing various computed features including life metrics and throughput for the specified sequence of battery data.
    """
    # Initialize DataFrame with specified columns and a single row initialized.
    result = pd.DataFrame(columns=[
        'seq_num', 'rpt_low_life', 'rpt_med_life', 'regu_life', 'regu_knee', 'q_throughput', 'e_throughput'
    ], index=range(1))
    result['seq_num'] = seq_num  # Assign the sequence number to the DataFrame

    # Uncomment the line below to include Open Circuit Voltage (OCV) information if function available
    # result['ocv_ini'], result['ocv_end'], result['v_decay_rate'] = get_ocv_info(diag_data)

    # Populate the DataFrame with cycle life metrics and throughput data by calling another function
    result['rpt_low_life'], result['rpt_med_life'], result['regu_life'], result['regu_knee'], result['q_throughput'], result['e_throughput'] = get_cycle_life(diag_summary, regu_summary, ax)

    # Return the populated DataFrame containing all the features
    return result


def get_seq_num(file):
    """
    This function helps retrive the sequence number from a file name (string)
    """
    if len(file) == 32:
        return 'Ref_' + file[file.find('.')-3:file.find('.')]
    else:
        return int(file[file.find('.')-3:file.find('.')])