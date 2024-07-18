import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import interpolate

def smooth(df, name, window):
    """
    Smooths the data in a specified column of a DataFrame using the Savitzky-Golay filter.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data to be smoothed. 
    name (str): The name of the voltage column containing the data to be smoothed.
    window (int): The length of the window to be used in the filter, which determines the
                  number of data points used for smoothing each point.
    
    Returns:
    DataFrame: The modified DataFrame with an additional column 'Voltage_sm' containing
               the smoothed data.
    """
    # Apply the Savitzky-Golay filter to smooth the data
    df['Voltage_sm'] = savgol_filter(df[name], window, 1)
    
    # Return the updated DataFrame
    return df

def process_full_test(full_file_name):
    """
    Processes a full test dataset by reading data from a CSV file, aligning voltages, 
    and applying smoothing to them. The data is interpolated using cubic interpolation 
    and then smoothed using a Savitzky-Golay filter.
    
    Parameters:
    full_file_name (str): The name of the CSV file containing test data.

    Returns:
    DataFrame: A DataFrame with columns for aligned voltage, charge (Q), and smoothed voltage.
    """
    # Construct the file path and load the data from CSV
    halfcell_path = 'data/dvf_data' # -> change this to where you save the data 
    full_cell = os.path.join(halfcell_path, full_file_name)
    full = pd.read_csv(full_cell) # unit mAh

    # Prepare a DataFrame for the processed data
    df_full = pd.DataFrame(columns=['Voltage_aligned', 'Q', 'Voltage_sm'], index=np.arange(1001))

    # Normalize and scale the capacity data
    capacity = 'discharge_capacity'
    voltage = 'voltage'
    x = (full[capacity] - full[capacity].min()) * 1000
    y = full[voltage]

    # Create cubic interpolation of the voltage as a function of normalized capacity
    f = interpolate.interp1d(x, y, kind='cubic', fill_value="extrapolate")
    z = np.linspace(x.min(), x.max(), 1001)
    df_full['Q'] = x.max() - z
    df_full['Voltage_aligned'] = f(z)

    # Smooth the aligned voltage data
    window = 5
    df_full = smooth(df_full, 'Voltage_aligned', window)
    
    # Return the processed DataFrame
    return df_full


def smooth_pe(pe, window):
    ### smooth the cathode data and structure the data to the desired format
    df_pe = pd.DataFrame()
    df_pe['SOC_aligned'] = 100 - pe['SOC_aligned']
    df_pe['Voltage_aligned'] = pe['Voltage_aligned']
    pe = smooth(pe, 'Voltage_aligned', window)
    df_pe['Voltage_sm'] = pe['Voltage_sm']
    return df_pe 

def smooth_ne(ne, window):
    ### smooth the anode data and structure the data to the desired format
    df_ne = pd.DataFrame()
    df_ne['SOC_aligned'] = ne['SOC_aligned']
    df_ne['Voltage_aligned'] = ne['Voltage_aligned']
    ne = smooth(ne, 'Voltage_aligned', window)
    df_ne['Voltage_sm'] = ne['Voltage_sm']
    return df_ne
 
def plot_dqdv(df, linestyle, color, voltage):\
    ### for full cell dqdv
    chosen = np.divide(np.gradient(df['SOC_aligned']),np.gradient(df[voltage]))
    plt.plot(df[voltage], chosen, color=color, linestyle = linestyle) 
    
def plot_dvdq_ele(df, linestyle, color, voltage, ax):
    ### for half cell dvdq
    chosen = np.divide(np.gradient(df[voltage]),np.gradient(df['SOC_aligned']))
    ax.plot(df['SOC_aligned'], -chosen, color=color, linestyle = linestyle) 
    
def simulation_error(x, df_pe, df_ne, full_cell, voltage):
    
    """
    Comapre the simulation with real data and calculate an error.  
    
    Parameters:
    x (np.array): An array of the fitting parameters
    df_pe (DataFrame): cathode data 
    df_ne (DataFrame): anode data 
    full_cell (DataFrame): full cell data that we want to simulate 
    voltage (string): smoothed or raw voltages 

    Returns:
    error (float): the defined error 
    """
    
    lbd = 0.1 # weight of the simulated voltage in the error 
    Q_ne = x[0]
    Q_pe = x[1]
    SOC_ne_0 = x[2]
    SOC_pe_0 = x[3]
    Q_ex = x[4]
    
    SOC_pe = df_pe['SOC_aligned']
    SOC_ne = df_ne['SOC_aligned']
    pe_voltage_smoothed = df_pe[voltage]
    ne_voltage_smoothed = df_ne[voltage]
    
    Q = full_cell.Q
    V_oc = full_cell[voltage]
    
    ne_input = SOC_ne_0 + Q/Q_ne*100
    pe_input = SOC_pe_0 - Q/Q_pe*100
    
    f_pe = interpolate.interp1d(SOC_pe, pe_voltage_smoothed, assume_sorted = False, fill_value = 'extrapolate')
    f_ne = interpolate.interp1d(SOC_ne, ne_voltage_smoothed, assume_sorted = False, fill_value = 'extrapolate')

    V_oc_simu = f_pe(pe_input) - f_ne(ne_input)
    v_err = np.square(V_oc_simu - V_oc).sum() # squared voltage error 
    
    dvdq_fc = np.divide(np.gradient(V_oc),np.gradient(Q))
    dvdq_fc_sim = np.divide(np.gradient(V_oc_simu),np.gradient(Q))
    
    dvdq_inds = np.argwhere((dvdq_fc_sim<0.009))
    dvdq_err = np.square(dvdq_fc[dvdq_inds] - dvdq_fc_sim[dvdq_inds]).sum() # sqaured dvdq error 
    
    SOC_ne_100 = SOC_ne_0 + (Q.max()+Q_ex)/Q_ne*100
    SOC_pe_100 = SOC_pe_0 - (Q.max()+Q_ex)/Q_pe*100
    
    # deviation between the top of charge voltages 
    cutoff_err = abs(f_pe(SOC_pe_100) - f_ne(SOC_ne_100) - V_oc.max())
    
    return lbd*v_err + (1-lbd)*dvdq_err + cutoff_err**2


def plot_dvdq(v, q, color, ax, label):
    # for full cell dvdq
    chosen = abs(np.divide(np.gradient(v),np.gradient(q)))
    ax.plot(q, chosen, color=color, label = label) 
    
    
def process_fitting_results(res, full_cell, df_ne, df_pe, plot, voltage):
    """
    Process the fitting results and perform calculations and plotting.

    Parameters:
    res (object): The fitting results object.
    full_cell (DataFrame): The DataFrame containing the full cell data.
    df_ne (DataFrame): The DataFrame containing the negative electrode data.
    df_pe (DataFrame): The DataFrame containing the positive electrode data.
    plot (bool): Flag indicating whether to plot the results.
    voltage (str): The voltage column to use for calculations and plotting.

    Returns:
    tuple: A tuple containing the error, SOC_ne_100, SOC_pe_100, Q.max(), and Q_li.
    """

    # Extract the fitting parameters
    x = res.x
    Q_ne = x[0]
    Q_pe = x[1]
    SOC_ne_0 = x[2]
    SOC_pe_0 = x[3]
    Q_ex = x[4]

    # Extract the relevant data from the DataFrames
    SOC_pe = df_pe['SOC_aligned']
    SOC_ne = df_ne['SOC_aligned']
    pe_voltage_smoothed = df_pe[voltage]
    ne_voltage_smoothed = df_ne[voltage]

    # Calculate the input electrode SOC values for voltage interpolation
    Q = full_cell.Q
    V_oc = full_cell['Voltage_sm']
    pe_input = SOC_pe_0 - Q/Q_pe*100
    ne_input = SOC_ne_0 + Q/Q_ne*100

    # Perform interpolation
    f_pe = interpolate.interp1d(SOC_pe, pe_voltage_smoothed, assume_sorted=False, fill_value='extrapolate')
    f_ne = interpolate.interp1d(SOC_ne, ne_voltage_smoothed, assume_sorted=False, fill_value='extrapolate')

    # Calculate the simulated open circuit voltage
    V_oc_simu = f_pe(pe_input) - f_ne(ne_input)

    # Calculate the error
    error = np.sqrt(np.square(V_oc_simu - V_oc).sum()/len(V_oc))

    # Calculate SOC_ne_100 and SOC_pe_100
    SOC_ne_100 = SOC_ne_0 + (Q.max()+Q_ex)/Q_ne*100
    SOC_pe_100 = SOC_pe_0 - (Q.max()+Q_ex)/Q_pe*100

    # Calculate Q_li
    Q_li = SOC_pe_0/100*Q_pe + SOC_ne_0/100*Q_ne

    if plot:
        # Calculate full cell real and simulated dVdQ and dQdV
        dvdq_fc = np.divide(np.gradient(V_oc), np.gradient(Q))
        Q_new = np.linspace(0, Q.max()+Q_ex, 1001)
        V_oc_simu = f_pe(SOC_pe_0 - Q_new/Q_pe*100) - f_ne(SOC_ne_0 + Q_new/Q_ne*100)
        dvdq_fc_sim = np.divide(np.gradient(V_oc_simu), np.gradient(Q_new))
        dqdv_fc = np.divide(np.gradient(Q), np.gradient(full_cell['Voltage_sm']))
        dqdv_fc_sim = np.divide(np.gradient(Q_new), np.gradient(abs(V_oc_simu)))


        num_rows = 2
        num_cols = 3
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*4.5, num_rows*2.5), facecolor='w', edgecolor='k', dpi=150)
        fig.subplots_adjust(hspace=.4, wspace=.3)
        axs = axs.ravel()

        # Plot dV/dQ
        axs[0].plot(Q, dvdq_fc, label='Real', color='grey')
        axs[0].plot(Q_new, dvdq_fc_sim, label='Fitted', color='skyblue')
        axs[0].legend(loc='best')
        axs[0].set_ylabel('dVdQ')
        axs[0].set_ylim([0, 0.008])
        axs[0].set_xlabel('Capacity (mAh)')

        # Plot dQ/dV
        axs[1].plot(V_oc, dqdv_fc, label='Real', color='grey')
        axs[1].plot(V_oc_simu, dqdv_fc_sim, label='Fitted', color='skyblue')
        axs[1].legend(loc='best')
        axs[1].set_ylabel('dQdV')
        axs[1].set_xlabel('Voltage (V)')

        # Plot V_Q
        axs[2].plot(Q, V_oc, label='Real', color='grey')
        axs[2].plot(Q_new, V_oc_simu, label='Fitted', color='skyblue')
        axs[2].legend(loc='best')
        axs[2].set_ylabel('Voltage/V')
        axs[2].set_xlabel('Capacity (mAh)')

        # Plot simulated electrodes and full cell curves
        y_vec = np.linspace(0, 100, 1000)
        x_vec = np.linspace(0, 100, 1000)
        qpos = Q_pe*(SOC_pe_0 - y_vec)/100
        vpos = f_pe(y_vec)
        axs[3].plot(qpos, vpos, color='royalblue', label='PE')
        axs[3].plot(Q, V_oc, label='Real', color='grey')
        axs[3].plot(Q, f_pe(pe_input) - f_ne(ne_input), label='Fitted', color='grey', linestyle='--')
        qneg = Q_ne*(x_vec - SOC_ne_0)/100
        vneg = f_ne(x_vec)
        axs[3].plot(qneg, vneg, color='salmon', label='NE')
        axs[3].legend(loc='center', fontsize=8)
        axs[3].set_ylabel('Voltage (V)')
        axs[3].set_xlabel('Q (mAh)')
        axs[3].set_xlim([-25, 300])
        axs[3].vlines(x=[0, np.max(Q)], ymin=0, ymax=5, color=[0.4, 0.4, 0.4], linestyle=':')

        # Plot dVdQ plot for individual electrodes
        q = (100 - pe_input)*Q_pe/100
        v = f_pe(pe_input)
        plot_dvdq(vpos, qpos, 'royalblue', axs[4], 'PE')
        q = ne_input*Q_ne/100
        v = f_ne(ne_input)
        plot_dvdq(vneg, qneg, 'salmon', axs[4], 'NE')

        axs[4].plot(Q, dvdq_fc, label='Real', color='grey')
        axs[4].plot(Q_new, dvdq_fc_sim, label='Fitted', color='grey', ls='--')
        axs[4].axvline(x=0, color='cyan', ls='--')
        axs[4].legend(loc='best', fontsize=8)
        axs[4].set_ylabel('dVdQ')
        axs[4].set_xlabel('Q (mAh)')
        axs[4].set_ylim([0, 0.008])

    return error, SOC_ne_100, SOC_pe_100, Q.max(), Q_li

def process_full(full):
    """
    Process the full cell data by interpolating the voltage values and smoothing the aligned voltage.

    Parameters:
    full (DataFrame): The DataFrame containing full cell discharge capacity and voltage columns.

    Returns:
    DataFrame: The processed full cell DataFrame.

    """

    df_full = pd.DataFrame(columns=['Voltage_aligned', 'Q', 'Voltage_sm'], index=np.arange(1001))
    capacity = 'discharge_capacity'
    voltage = 'voltage'

    # change the Q unit to be mAh instead of Ah
    x = (full[capacity] - full[capacity].min()) * 1000
    y = full[voltage]

    f = interpolate.interp1d(x, y, kind='cubic', fill_value="extrapolate")

    z = np.linspace(x.min(), x.max(), 1001)

    df_full['Q'] = x.max() - z
    
    df_full['Voltage_aligned'] = f(z)
    
    df_full = smooth(df_full, 'Voltage_aligned', 3)
    
    return df_full