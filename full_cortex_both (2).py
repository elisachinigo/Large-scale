##main function

import sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import scipy.io
import os
from scipy.signal import find_peaks
from scipy.stats import ks_2samp
import argparse

# Parse task ID (from Slurm array)
def parse_args():
    parser = argparse.ArgumentParser(description = "Run simulation with the model parameters")
    parser.add_argument('task_id', type = int, help = "Slurm array task ID")
    parser.add_argument('gen_file', type=str, help='Path to the generation mat file')
    return parser.parse_args()

def main():
    args = parse_args()
    task_id = args.task_id # slurm task ID
    gen_file = args.gen_file # path to generation_n.mat file
    #load parameters from the generation file
    data = scipy.io.loadmat(gen_file)
    model_params = data['params'] #All parameter sets
    print("task_id",task_id)
    current_parms = model_params[task_id,:] # select the parameters for this specific job
    # simulate using current_params
    print(f"Running simulation for task ID {task_id} with parameters: {current_parms}")

    ## run the full function here
    full_cortex(task_id,current_parms,gen_filename = gen_file)


def full_cortex(task_id,current_parms,gen_filename):
    # load the long-range connections
    print("Starting full_cortex function")
    conn_cxcx = np.loadtxt('/scratch/ec4513/large_scale/conn_cxcx.csv', delimiter=',')
    if not os.path.exists('/scratch/ec4513/large_scale/conn_cxcx.csv'):
        print("Error: conn_cxcx.csv not found!")
        return
    area_list = list(np.loadtxt('/scratch/ec4513/large_scale/area_list.csv', delimiter=',', dtype='str'))
    # load hierarchy
    hierarchy_df = pd.read_csv('/scratch/ec4513/large_scale/dfHier.csv', header=0, index_col=0)
    hierarchy = np.array(hierarchy_df['hierarchy index'])
    # we want to normalise and scale the ctx-ctx connectivity matrix. To do this we define a squish factor k to rescale and then normalise by the max. 
    k = current_parms[0] ## need to change this should be a parameter!!!!
    print('K', k)
    W_scale = conn_cxcx**k
    W_norm = W_scale/np.max(W_scale)
    # now using this scaling we can change the long range projections to E and I cells respectively. 
    m_matrix = generate_m_scaling(hierarchy,1)
    W_E = W_norm * m_matrix
    W_I = W_norm * (1-m_matrix)
    # Initialize the new 86x86 matrix with zeros
    new_matrix = np.zeros((86, 86))

    # Populate the new_matrix based on the mapping rules
    for i in range(W_E.shape[0]):
        for j in range(W_E.shape[1]):
            new_matrix[2*i, 2*j] = 0.1*W_E[i, j]
            new_matrix[2*i+1,2*j] = 0.1*W_I[i,j]
    mu_LR = current_parms[1]
    new_matrix = new_matrix * mu_LR
    # now that we've set up the long-range circuit let's set up all the local connections (they lie on the diagonal)
    # to save time and effort let's just set these to be the same as in the RSC only model (get back to this when running parameter simulations)
    W_EE = 3.04
    W_IE = 3.24
    W_EI = -1.5
    W_II = -0.5
    
    # populate with fixed local weights
    for i in range(W_E.shape[0]):
        new_matrix[2*i, 2*i] = W_EE
        new_matrix[2*i,2*i+1] = W_EI
        new_matrix[2*i+1, 2*i] = W_IE
        new_matrix[2*i+1, 2*i+1] = W_II
    
        # Populate the new_matrix based on the mapping rules
        #for i in range(W_E.shape[0]):
         #   new_matrix[2*i, 2*i] = current_parms[0][task_id]['W_EE']
          #  new_matrix[2*i,2*i+1] = current_parms[0][task_id]['W_EI']
           # new_matrix[2*i+1, 2*i] = current_parms[0][task_id]['W_IE']
            #new_matrix[2*i+1, 2*i+1] = current_parms[0][task_id]['W_II']

    cluster_func(task_id,new_matrix,current_parms,gen_filename)
    




def generate_m_scaling(harrishierarchy,beta_scaling):
    h = harrishierarchy
    n = np.size(h)
    m = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            m[i,j] = 1/(1+np.exp(-beta_scaling * (h[i]-h[i])))
    return m

def random_parms(n=50):
    parms_list = []
    parms_array = np.zeros((n,8))
    for i in range(n):
        parms_rand = {
            'k_scaling' : 0.3,
            'mu_LR': np.random.rand() * 0.8,
            'I_E' : np.random.rand() * 5,
            'I_I' : np.random.rand() * 5,
            'beta_a' : np.random.rand(), 
            'A0' : np.random.rand() * 10,
            'beta_h' : - np.random.rand(),
            'h0' : np.random.rand() * 10,
        }
        # Populate the row for this individual
        parms_array[i] = [
            parms_rand['k_scaling'],
            parms_rand['mu_LR'],
            parms_rand['I_E'],
            parms_rand['I_I'],
            parms_rand['beta_a'],
            parms_rand['A0'],
            parms_rand['beta_h'],
            parms_rand['h0']
        ]
    return parms_array

# Main function
def cluster_func(task_id,new_matrix,opt_parms=None,gen_filename=None):
    total_simtime = 20000  # Total simulation time
    initial_simtime = 1000 # simulation time before doing initial check for crossing
    dt = 1
    parms = {
            'N_neurons': 86,
            'I_in': np.tile(np.array([opt_parms[2],opt_parms[3]]),(43,1)).reshape(86,1),
            'W': new_matrix,
            'beta_h': np.tile(np.array([opt_parms[6], opt_parms[6]]),(43,1)).reshape(86,1),
            'beta_a': np.tile(np.array([opt_parms[4], opt_parms[4]]),(43,1)).reshape(86,1),
            'tau_r': np.ones((86,1)),
            'tau_h': np.ones((86,1))*100,
            'h0': np.ones((86,1))* opt_parms[7],
            'hk': np.ones((86,1))*(-20),
            'tau_a': np.ones((86,1))*100,
            'A0': np.ones((86,1)) * opt_parms[5],
            'Ak': np.ones((86,1))*(20),
            'k': np.tile(np.array([0.02, 0.05]),(43,1)).reshape(86,1),
            'h': np.tile(np.array([0, 12]),(43,1)).reshape(86,1),
            'n': 2,
            'noiseamp': 0.37,
            'noisefreq': 0.05,
            'k_scaling' : opt_parms[0],
            'mu_LR' :opt_parms[1]
            }
    
    Y_sol_initial = SSAdapt_run(initial_simtime, dt, parms)
    Y_sol_total = crossing_check(Y_sol_initial, total_simtime, initial_simtime,parms,task_id,gen_filename)
    return Y_sol_total.shape


# System solver function
def SSAdapt_run(simtime, dt, parms):
    N_neurons = parms['N_neurons']
    W = parms['W']
    I_in = parms['I_in']
    beta_h = parms['beta_h'] # strength of Ih
    beta_a = parms['beta_a'] # strength of adaptation
    tau_r = parms['tau_r'] # timescale for neuron firing
    tau_h = parms['tau_h'] # timescale of Ih
    hk = parms['hk'] # steepness activation function of Ih
    h0 = parms['h0'] # threshold of activation of Ih
    tau_a = parms['tau_a'] # timescale of adaptation
    Ak = parms['Ak'] # steepness of activation function of adaptation
    A0 = parms['A0'] # threshold of activation of adaptation
    k = parms['k'] # thresholds of activation for the two neuron populations E and I
    h = parms['h'] # steepness of the activation function for the two neuron populations
    n = parms['n']
    mu_LR = parms['mu_LR'] # scales the LR projections strengths relative to local circuits
    #LR = parms['LR']
    noiseamp = parms['noiseamp']
    noisefreq = parms['noisefreq']

    # Initial conditions
    r_init = np.random.rand(N_neurons)
    h_init = np.random.rand(N_neurons)
    a_init = np.random.rand(N_neurons)
    y0 = np.concatenate([r_init, h_init,a_init])

    # Time span
    tspan = np.arange(0, simtime + dt, dt)

    # Generate noise
    Inoise, noiseT = OUNoise(noisefreq, noiseamp, simtime, dt / 10, dt, N_neurons)

    # Solve the system using solve_ivp (non-delay ODE solver)
    sol = solve_ivp(lambda t, y: SSadapt_eqs(t, y, N_neurons, W, I_in, beta_h, beta_a, tau_r, tau_h, hk, h0, tau_a, Ak, A0, k, h, n, noiseT, Inoise), 
                    [tspan[0], tspan[-1]], y0, t_eval=tspan, rtol=1e-12, atol=1e-12)

    return sol.y.T

# ODE system of equations (non-delay)
def SSadapt_eqs(t, y, N_neurons, W, I_in, beta_h, beta_a, tau_r, tau_h, hk, h0, tau_a, Ak, A0, k, h, n, noiseT, Inoise):
    r = y[:N_neurons].reshape(-1,1)  # Neuronal activity
    Ih = y[N_neurons:2*N_neurons].reshape(-1,1) #I_h current activity
    a = y[2*N_neurons:].reshape(-1,1)  # Adaptation activity

    # Interpolate noise for the current time step
    interp_func = interp1d(noiseT, Inoise, axis=0, fill_value='extrapolate')
    noise_vals = interp_func(t).reshape(-1,1)

    # Calculate total input
    I_tot = np.dot(W, r) - Ih - a + I_in + noise_vals

    # Neuronal dynamics
    F_I = k * np.heaviside(I_tot - h , 0) * (I_tot - h ) ** n
    
    # I_h dynamic
    hinf = np.zeros((N_neurons,1))
    #hinf[:43] = beta_h[:43] / (1 + np.exp(-hk[:43] * (r[:43] - h0[:43])))
    #hinf[43:] = beta_h[43:] / (1 + np.exp(-hk[43:] * (r[43:] - h0[43:])))
    hinf[:43] = beta_h[:43] / (1 + np.exp(-np.clip(hk[:43] * (r[:43] - h0[:43]), -500, 500)))
    hinf[43:] = beta_h[43:] / (1 + np.exp(-np.clip(hk[43:] * (r[43:] - h0[43:]), -500, 500)))


    # Adaptation dynamics
    Ainf = np.zeros((N_neurons,1))
    #Ainf[:43] = beta_a[:43] / (1 + np.exp(-Ak[:43] * (r[:43] - A0[:43])))
    #Ainf[43:] = beta_a[43:] / (1 + np.exp(-Ak[43:] * (r[43:] - A0[43:])))
    Ainf[:43] = beta_a[:43] / (1 + np.exp(-np.clip(Ak[:43] * (r[:43] - A0[:43]), -500, 500)))
    Ainf[43:] = beta_a[43:] / (1 + np.exp(-np.clip(Ak[43:] * (r[43:] - A0[43:]), -500, 500)))

    # Rate of change for r and a
    dr = (-r + F_I) / tau_r
    dh = (-Ih + hinf) / tau_h
    da = (-a + Ainf) / tau_a

    # Combine into a single vector for ODE solver
    return np.concatenate([dr,dh, da]).flatten()

# Ornstein-Uhlenbeck noise generation function
def OUNoise(theta, sigma, duration, dt, save_dt, numsignals):
    simtimevector = np.arange(0, duration + dt, dt)
    SimTimeLength = len(simtimevector)
    randnums = np.random.randn(numsignals, SimTimeLength)
    X_t = sigma * np.random.randn(numsignals, 1)  # Start at random value

    X = []
    T = []
    for tt in range(SimTimeLength):
        dX = -theta * X_t * dt + np.sqrt(2 * theta) * sigma * randnums[:, tt].reshape(-1, 1) * np.sqrt(dt)
        X_t += dX
        if tt % int(save_dt / dt) == 0:
            X.append(X_t.copy())
            T.append(simtimevector[tt])

    return np.hstack(X).T, np.array(T)


import numpy as np
from scipy.signal import find_peaks
from diptest import diptest as diptestfunction

def bimodal_thresh(bimodaldata, maxthresh=np.inf, schmidt=True, maxhistbins=40, startbins=10, setthresh=None):
    """
    Calculate the threshold between bimodal data modes (e.g., UP vs DOWN states) and return crossing times.
    
    Args:
    - bimodaldata: A 1D array of bimodal time-series data.
    - maxthresh: Optional, sets a maximum threshold.
    - schmidt: Optional, uses Schmidt trigger (halfway points between peaks).
    - maxhistbins: Maximum number of histogram bins to try.
    - startbins: Minimum number of histogram bins to start with.
    - setthresh: Manually set the threshold.
    
    Returns:
    - thresh: The calculated threshold between the modes.
    - cross: A dictionary with 'upints' and 'downints' (onsets and offsets of UP/DOWN states).
    - bihist: Bimodal histogram data.
    - diptest: Dip test result.
    - crossup, crossdown: Crossings for UP and DOWN states.
    """
    
    # Optional parameter handling
    thresh = setthresh if setthresh is not None else None
    
    # Initialize crossings
    crossup, crossdown = np.nan, np.nan

    # Handle NaN and inf values in bimodaldata
    bimodaldata = np.nan_to_num(bimodaldata, nan=0.0, posinf=maxthresh, neginf=-maxthresh)
    
    # Hartigan's Dip Test for bimodality
    dip, p_value = diptestfunction(bimodaldata)
    diptest_result = {"dip": dip, "p_value": p_value}
    
    if p_value > 0.05:  # Not bimodal, exit early
        print("Dip test indicates data is not bimodal.")
        return np.nan, {'upints': [], 'downints': []}, None, diptest_result, None, None

    # Histogram-based thresholding to find exactly two peaks
    if thresh is None:
        numbins = startbins
        while numbins <= maxhistbins:
            hist, bin_edges = np.histogram(bimodaldata, bins=numbins)
            print("number of bins",numbins)
            peaks, _ = find_peaks(hist, prominence =0.3)
            if len(peaks) == 2:
                peak1, peak2 = peaks[:2]
                print('peak1',peak1,'peak2',peak2)
                betweenpeaks = bin_edges[peak1:peak2+1]  # Values between peaks
                dip, diploc = find_peaks(-hist[peak1:peak2+1], height=None)
                if dip.size > 0:
                    thresh = betweenpeaks[dip]  # Use first dip as threshold
                    print("threshold",thresh)
                break
            numbins += 1
        
        # If no two peaks are found, set thresh to NaN
        if thresh is None:
            thresh = np.nan
            print("Unable to find valid threshold.")
            return thresh, {'upints': [], 'downints': []}, hist, diptest_result, None, None

    # Schmidt Trigger for UP/DOWN thresholds
    if schmidt:
        # Calculate UP and DOWN thresholds using betweenpeaks
        if not np.isnan(thresh):
            threshUP = thresh + 0.5 * (betweenpeaks[-1] - thresh)
            threshDOWN = thresh + 0.5 * (betweenpeaks[0] - thresh)
            
            # Find UP and DOWN crossings
            overUP = bimodaldata > threshUP
            overDOWN = bimodaldata > threshDOWN
            
            crossup = np.where(np.diff(overUP.astype(int)) == 1)[0]
            crossdown = np.where(np.diff(overDOWN.astype(int)) == -1)[0]
            
            # Remove incomplete crossings
            allcrossings = np.concatenate([np.column_stack((crossup, np.ones(len(crossup)))), 
                                           np.column_stack((crossdown, np.zeros(len(crossdown))))])
            allcrossings = allcrossings[np.argsort(allcrossings[:, 0])]
            updownswitch = np.diff(allcrossings[:, 1])
            allcrossings = np.delete(allcrossings, np.where(updownswitch == 0)[0] + 1, axis=0)
            
            # Update crossup and crossdown based on filtered crossings
            crossup = allcrossings[allcrossings[:, 1] == 1][:, 0]
            crossdown = allcrossings[allcrossings[:, 1] == 0][:, 0]
            
            # Ensure non-empty crossings
            crossup = crossup if len(crossup) > 0 else np.array([0])
            crossdown = crossdown if len(crossdown) > 0 else np.array([0])
            
            # If still no valid crossings, return None
            if len(crossup) == 0 or len(crossdown) == 0:
                return np.nan, {'upints': [], 'downints': []}, hist, diptest_result, None, None
    else:
        overind = bimodaldata > thresh
        crossup = np.where(np.diff(overind.astype(int)) == 1)[0]
        crossdown = np.where(np.diff(overind.astype(int)) == -1)[0]
        
        if len(crossup) == 0 or len(crossdown) == 0:
            return np.nan, {'upints': [], 'downints': []}, hist, diptest_result, None, None

    # Initialize separate arrays for MATLAB-like up/down intervals
    upforup = crossup.copy()
    upfordown = crossup.copy()
    downforup = crossdown.copy()
    downfordown = crossdown.copy()
    
    # Adjust intervals similar to MATLAB logic
    if crossup[0] < crossdown[0]:
        upfordown = upfordown[1:]  # Remove the first element from upfordown
    if crossdown[-1] > crossup[-1]:
        downfordown = downfordown[:-1]  # Remove the last element from downfordown
    if crossdown[0] < crossup[0]:
        downforup = downforup[1:]  # Remove the first element from downforup
    if crossup[-1] > crossdown[-1]:
        upforup = upforup[:-1]  # Remove the last element from upforup
    
    # Combine results for UP and DOWN intervals
    cross = {
        'upints': np.column_stack((upforup, downforup)) if len(upforup) and len(downforup) else [],
        'downints': np.column_stack((downfordown, upfordown)) if len(downfordown) and len(upfordown) else []
    }
    
    # If empty, set to empty list as in MATLAB code
    if len(cross['upints']) == 0:
        cross['upints'] = []
    if len(cross['downints']) == 0:
        cross['downints'] = []
    if len(crossup) == 0:
        crossup = np.nan
    if len(crossdown) == 0:
        crossdown = np.nan

    # Bimodal histogram output for inspection
    bihist = {'hist': hist, 'bins': bin_edges}

    return thresh, cross, bihist, diptest_result, crossup, crossdown


## compares the similarity of the simulated output with rachel's data. Returns a vector (value for RSC) of the fitness value of the simulation
def fit(sim_sol):
    #find the fitness for the RSC module only. Can average across RSC areas. Hard coded -- change later. 
    area_list = list(np.loadtxt('/scratch/ec4513/large_scale/area_list.csv', delimiter=',', dtype='str'))
    RSC = ['RSPv','RSPd','RSPagl']
    indices = [area_list.index(name) for name in RSC if name in area_list]
    print('indices',indices)
    average_fit = []
    for i in indices:
        thresh, cross, bihist, diptest, crossup, crossdown = bimodal_thresh(sim_sol[:,2*i])
        if len(cross['upints'])==0 or len(cross['downints'])==0:
            print("No valid crossings found")
            average_fit = np.append(average_fit, 0)
        else:
            up_durations = cross['upints'][:,1] - cross['upints'][:,0]
            down_durations = cross['downints'][:,1] - cross['downints'][:,0]
            # here load the RSC data from rachel
            datas = scipy.io.loadmat('/scratch/ec4513/evo_algo/U_D_samples.mat')
            data_up = datas['U_durations']
            data_down = datas['D_durations']
            print('data_up shape',data_up.shape)
            print('data_down shape', data_down.shape)
            # perform the KS tests for up and down distributions
            stat_up, p_up = ks_2samp(up_durations, data_up.ravel())
            stat_down, p_down = ks_2samp(down_durations, data_down.ravel())
            fitness = (1-stat_up)*(1-stat_down)
            average_fit = np.append(average_fit, fitness)
            print('stat_up',stat_up)
            print('stat_down',stat_down)
            print(np.histogram(data_up))
        print('average fit',average_fit, i)
    #average_fit = np.mean(average_fit)
    return average_fit
    
def generate_new_population(old_params, old_fitness):
    # Step 1: Average fitness across 3 values to get a single score per individual
    avg_fitness = np.mean(old_fitness,axis = 1)
    
    # Step 2: Sort by fitness in descending order and get indices
    sorted_indices = np.argsort(avg_fitness)[::-1]
    top_indices = sorted_indices[:20] # take top 20 fittest individuals
    
    # Step 3: Select top parameters for reproduction
    top_params = old_params[top_indices]
    num_individuals = 50
    num_params = old_params.shape[1]
    
    # Step 4: Generate new population
    new_params = np.zeros((num_individuals, num_params))
    for i in range(num_individuals):
        # randomly select two parents
        parent1, parent2 = top_params[np.random.choice(20,2,replace= False)]
        
        # Generate each parameter by sampling from a uniform distribution around the parents' values
        for j in range(num_params):
            param_min = min(parent1[j], parent2[j]) - abs(parent1[j] - parent2[j]) / 2
            param_max = max(parent1[j], parent2[j]) + abs(parent1[j] - parent2[j]) / 2
            new_params[i,j] = np.random.uniform(param_min, param_max)
    return new_params

def crossing_check(Y_sol, total_simtime, initial_simtime,parms,task_id,gen_filename):
    thresh, cross, bihist, diptest, crossup, crossdown = bimodal_thresh(Y_sol[:,48])
    print("Ran threshold calc")
    print('cross[upints]',cross['upints'])
    print('cross[downints]',cross['downints'])
    if len(cross['upints']) == 0 and len(cross['downints']) == 0:
        # No crossings detected, terminate early with fitness  score of 0
        print("No crossings detected within initial timesteps. Terminating early with fitness score 0")
        model_fitness = np.array(np.zeros((1,3)))
        data = scipy.io.loadmat(gen_filename)
        print('fitness',model_fitness.shape)
        #update the fitness for the current model (indexed by 'id')
        data['fitness'][task_id,:] = model_fitness.flatten()
        scipy.io.savemat(gen_filename,data)
    else:
        print("Crossing detected!  Continuing simulation")
        # Crossing detected so continue simulation from where it left off
        last_state = Y_sol[-1,:]
        print("last_state shape",last_state.shape)
        # Define the remaining time span
        remaining_steps = total_simtime - initial_simtime
        remaining_tspan = np.linspace(initial_simtime, total_simtime, remaining_steps)
        dt = 1
        N_neurons = parms['N_neurons']
        W = parms['W']
        I_in = parms['I_in']
        beta_h = parms['beta_h'] # strength of Ih
        beta_a = parms['beta_a'] # strength of adaptation
        tau_r = parms['tau_r'] # timescale for neuron firing
        tau_h = parms['tau_h'] # timescale of Ih
        hk = parms['hk'] # steepness activation function of Ih
        h0 = parms['h0'] # threshold of activation of Ih
        tau_a = parms['tau_a'] # timescale of adaptation
        Ak = parms['Ak'] # steepness of activation function of adaptation
        A0 = parms['A0'] # threshold of activation of adaptation
        k = parms['k'] # thresholds of activation for the two neuron populations E and I
        h = parms['h'] # steepness of the activation function for the two neuron populations
        n = parms['n']
        mu_LR = parms['mu_LR'] # scales the LR projections strengths relative to local circuits
        #LR = parms['LR']
        noiseamp = parms['noiseamp']
        noisefreq = parms['noisefreq']
        Inoise, noiseT = OUNoise(noisefreq, noiseamp, remaining_steps, dt / 10, dt, N_neurons)
        print(Inoise.shape)
        sol_full = solve_ivp(lambda t, y: SSadapt_eqs(t, y, N_neurons, W, I_in, beta_h, beta_a, tau_r, tau_h, hk, h0, tau_a, Ak, A0, k, h, n, noiseT, Inoise), 
                    [remaining_tspan[0], remaining_tspan[-1]], last_state, t_eval=remaining_tspan, rtol=1e-12, atol=1e-12)
                    

        full_solution_vals = np.concatenate((Y_sol, sol_full.y.T))
        model_fitness = fit(full_solution_vals)
        #scipy.io.savemat(f'/scratch/ec4513/large_scale/simulation_{id}.mat', {'Y_sol': Y_sol, 'parms': parms})
        #model_fitness = fit(Y_sol) # assuming this is a  (3x1) vector
        data = scipy.io.loadmat(gen_filename)
        print('fitness',model_fitness.shape)
        print(data['fitness'])
        #update the fitness for the current model (indexed by 'id')
        data['fitness'][task_id,:] = model_fitness.flatten()
        scipy.io.savemat(gen_filename,data)
    return model_fitness

# This will only run if the script is executed directly (not imported)
if __name__ == "__main__":
    main()