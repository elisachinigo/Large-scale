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
from diptest_funcs import *

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
    beta_scaling = current_parms[9]
    m_matrix = generate_m_scaling(hierarchy,beta_scaling)
    W_E = W_norm * m_matrix
    W_I = W_norm * (1-m_matrix)
    # Initialize the new 86x86 matrix with zeros
    new_matrix = np.zeros((86, 86))
    mu_EE = current_parms[1]
    mu_EI = current_parms[8]
    # Populate the new_matrix based on the mapping rules
    for i in range(W_E.shape[0]):
        for j in range(W_E.shape[1]):
            new_matrix[2*i, 2*j] = mu_EE * W_E[i, j]
            new_matrix[2*i+1,2*j] = mu_EI * W_I[i,j]
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

def random_parms(n=100):
    parms_list = []
    parms_array = np.zeros((n,10))
    for i in range(n):
        parms_rand = {
            'k_scaling' : np.random.rand(),
            'mu_LR_EE': np.random.rand()*0.8 ,
            'I_E' : np.random.rand() * 5,
            'I_I' : np.random.rand() * 5,
            'beta_a' : np.random.rand(), 
            'A0' : np.random.rand() * 10,
            'beta_h' : - np.random.rand(),
            'h0' : np.random.rand() * 10,
            'mu_LR_EI': np.random.rand()*0.8,
            'beta_scaling':np.random.rand()*5,
        }
        # Populate the row for this individual
        parms_array[i] = [
            parms_rand['k_scaling'],
            parms_rand['mu_LR_EE'],
            parms_rand['I_E'],
            parms_rand['I_I'],
            parms_rand['beta_a'],
            parms_rand['A0'],
            parms_rand['beta_h'],
            parms_rand['h0'],
            parms_rand['mu_LR_EI'],
            parms_rand['beta_scaling']
        ]
    return parms_array

# Main function
def cluster_func(task_id,new_matrix,opt_parms=None,gen_filename=None):
    total_simtime = 12000  # Total simulation time
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
            'mu_LR_EE' : opt_parms[1],
            'mu_LR_EI' : opt_parms[8],
            'beta_scaling':opt_parms[9]
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
    mu_LR_EE = parms['mu_LR_EE'] # scales the LR projections strengths relative to local circuits
    mu_LR_EI = parms['mu_LR_EI'] 
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


from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import os

def find_best_scale_factor(data_up, data_down, up_durations, down_durations, sf_range, logtransform=True, nboot=300):
    """
    Find the best scaling factor for matching simulated dwell times with experimental dwell times.

    Parameters:
    - data_up: Vector of experimental UP state dwell times.
    - data_down: Vector of experimental DOWN state dwell times.
    - up_durations: Vector of simulated UP state dwell times.
    - down_durations: Vector of simulated DOWN state dwell times.
    - sf_range: Tuple indicating range of scale factors (e.g., (0.0001, 0.02)).
    - output_dir: Directory to save the output plot.
    - id: Generation ID to specify in the file path.
    - k: Simulation ID to specify in the file path.
    - logtransform: Boolean to apply log transform to dwell times.
    - nboot: Number of scale factors to test.

    Returns:
    - bestsf: Best scaling factor.
    - dwelltimes_sim_sf: Dictionary with scaled simulated UP and DOWN dwell times.
    - KSSTAT: Dictionary with final K-S statistics for UP and DOWN states.
    - selectionparm: Final value of similarity metric at bestsf.
    """

    # Initialize variables
    scalefactors = np.linspace(sf_range[0], sf_range[1], nboot)
    KSSTAT = {'UP': [], 'DOWN': []}
    dwelltimes_sim_sf = {'UP': np.nan, 'DOWN': np.nan}
    bestsf = np.nan
    selectionparm = np.nan

    # Check for NaN in simulated data
    if np.any(np.isnan(up_durations)) or np.any(np.isnan(down_durations)):
        return bestsf, dwelltimes_sim_sf, KSSTAT, selectionparm

    # Log transform if specified
    if logtransform:
        data_up = np.log10(data_up)
        data_down = np.log10(data_down)

    # Loop through scale factors and perform K-S test
    for sf_test in scalefactors:
        # Scale simulated dwell times
        dwelltimes_sim_test = {
            'UP': up_durations * sf_test,
            'DOWN': down_durations * sf_test
        }

        # Log transform scaled data if specified
        if logtransform:
            dwelltimes_sim_test['UP'] = np.log10(dwelltimes_sim_test['UP'])
            dwelltimes_sim_test['DOWN'] = np.log10(dwelltimes_sim_test['DOWN'])

        # Perform K-S test between experimental and scaled simulated data
        KSSTAT['DOWN'].append(ks_2samp(dwelltimes_sim_test['DOWN'], data_down).statistic)
        KSSTAT['UP'].append(ks_2samp(dwelltimes_sim_test['UP'], data_up).statistic)

    # Calculate similarity metric
    KSSTAT['UP'] = np.array(KSSTAT['UP'])
    KSSTAT['DOWN'] = np.array(KSSTAT['DOWN'])
    selectionparm = (1 - KSSTAT['UP']) * (1 - KSSTAT['DOWN'])

    # Find the best scale factor based on the similarity metric
    bestsfidx = np.argmax(selectionparm)
    bestsf = scalefactors[bestsfidx]
    dwelltimes_sim_sf = {
        'UP': up_durations * bestsf,
        'DOWN': down_durations * bestsf
    }

    # Create plot of K-S statistics and similarity metric
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(scalefactors, KSSTAT['DOWN'], 'r', label='K-S: DOWN')
    ax.plot(scalefactors, KSSTAT['UP'], 'g', label='K-S: UP')
    ax.plot(scalefactors, selectionparm, 'k', linewidth=2, label='Similarity')
    ax.legend(loc='best')
    ax.set_xlabel('Time Scale Factor (s/tau_r)')
    ax.set_ylabel('K-S Statistic / Similarity Metric')
    ax.set_title('Finding Time Scale Factor')

    # Save the plot to the specified directory
    #output_path = os.path.join(output_dir, f'gen_{id}/sim_{k}.png')
    #os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #plt.savefig(output_path)
    #plt.close(fig)

    # Final outputs
    KSSTAT['DOWN'] = KSSTAT['DOWN'][bestsfidx]
    KSSTAT['UP'] = KSSTAT['UP'][bestsfidx]
    selectionparm = selectionparm[bestsfidx]

    return bestsf, dwelltimes_sim_sf, KSSTAT, selectionparm
    
def fit(full_solution_vals):
    #find the fitness for the RSC module only. Can average across RSC areas. Hard coded -- change later. 
    area_list = list(np.loadtxt('/scratch/ec4513/large_scale/area_list.csv', delimiter=',', dtype='str'))
    RSC = ['RSPv','RSPd','RSPagl']
    indices = [area_list.index(name) for name in RSC if name in area_list]
    print('indices',indices)
    average_fit = []
    average_KS_UP = []
    average_KS_DOWN = []
    for i in indices:
        thresh, cross, bihist, diptest, crossup, crossdown = bimodal_thresh(full_solution_vals[:,2*i])
        if len(cross['upints'])==0 or len(cross['downints'])==0:
            print("No valid crossings found")
            average_fit = np.append(average_fit, 0)
            average_KS_UP =np.append(average_KS_UP,1)
            average_KS_DOWN= np.append(average_KS_DOWN,1)
        else:
            up_durations = cross['upints'][:,1] - cross['upints'][:,0]
            down_durations = cross['downints'][:,1] - cross['downints'][:,0]
            # here load the RSC data from rachel
            datas = scipy.io.loadmat('/scratch/ec4513/evo_algo/U_D_samples.mat')
            data_up = datas['U_durations']
            data_down = datas['D_durations']
            print('data_up shape',data_up.shape)
            print('data_down shape', data_down.shape)
            # perform the KS tests for up and down distributions & find the best timescale factor tau
            bestsf, dwelltimes_sim_sf, KSSTAT, selectionparm=find_best_scale_factor(data_up.ravel(), data_down.ravel(), up_durations, down_durations, (0.0001,1))
            average_fit = np.append(average_fit, selectionparm)
            average_KS_UP = np.append(average_KS_UP,KSSTAT['UP'])
            average_KS_DOWN = np.append(average_KS_DOWN,KSSTAT['DOWN'])
        print('average fit',average_fit, i)
    return average_fit,average_KS_UP,average_KS_DOWN



def generate_new_population(old_params, old_fitness):
    # Step 1: Average fitness across 3 values to get a single score per individual
    avg_fitness = np.mean(old_fitness,axis = 1)
    
    # Step 2: Sort by fitness in descending order and get indices
    sorted_indices = np.argsort(avg_fitness)[::-1]
    top_indices = sorted_indices[:20] # take top 20 fittest individuals
    
    # Step 3: Select top parameters for reproduction
    top_params = old_params[top_indices]
    top_fitness = avg_fitness[top_indices]
    num_individuals = 100
    num_params = old_params.shape[1]
    
    # Step 4: Generate new population
    new_params = np.zeros((num_individuals, num_params))
    for i in range(num_individuals):
        # randomly select two parents
        idx1, idx2 = np.random.choice(20, 2, replace=False)
        parent1, parent2 = top_params[idx1], top_params[idx2]
        fitness1, fitness2 = top_fitness[idx1], top_fitness[idx2]
        
        # Calculate selection probabilities for each parent based on their fitness
        prob1 = fitness1 / (fitness1 + fitness2)
        prob2 = 1 - prob1  # Complementary probability for parent2

        for j in range(num_params):
            # Choose parameter from parent1 or parent2 based on probability
            chosen_param = parent1[j] if np.random.rand() < prob1 else parent2[j]
            
            # Add white noise to the chosen parameter
            new_params[i, j] = chosen_param + np.random.normal(0, 0.1)  # Adjust 0.1 to set noise level
    return new_params



def crossing_check(Y_sol, total_simtime, initial_simtime,parms,task_id,gen_filename):
    thresh, cross, bihist, diptest, crossup, crossdown = bimodal_thresh(Y_sol[:,22])
    print("Ran threshold calc")
    print('cross[upints]',cross['upints'])
    print('cross[downints]',cross['downints'])
    if len(cross['upints']) == 0 and len(cross['downints']) == 0:
        # No crossings detected, terminate early with fitness  score of 0
        print("No crossings detected within initial timesteps. Terminating early with fitness score 0")
        model_fitness = np.array(np.zeros((1,3)))-1
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
        mu_LR_EE = parms['mu_LR_EE'] # scales the LR projections strengths relative to local circuits
        mu_LR_EI = parms['mu_LR_EI']
        #LR = parms['LR']
        noiseamp = parms['noiseamp']
        noisefreq = parms['noisefreq']
        Inoise, noiseT = OUNoise(noisefreq, noiseamp, remaining_steps, dt / 10, dt, N_neurons)
        print(Inoise.shape)
        sol_full = solve_ivp(lambda t, y: SSadapt_eqs(t, y, N_neurons, W, I_in, beta_h, beta_a, tau_r, tau_h, hk, h0, tau_a, Ak, A0, k, h, n, noiseT, Inoise), 
                    [remaining_tspan[0], remaining_tspan[-1]], last_state, t_eval=remaining_tspan, rtol=1e-12, atol=1e-12)
                    

        full_solution_vals = np.concatenate((Y_sol, sol_full.y.T))
        model_fitness, average_KS_UP,average_KS_DOWN = fit(full_solution_vals)
        #scipy.io.savemat(f'/scratch/ec4513/large_scale/simulation_{id}.mat', {'Y_sol': Y_sol, 'parms': parms})
        #model_fitness = fit(Y_sol) # assuming this is a  (3x1) vector
        data = scipy.io.loadmat(gen_filename)
        print('fitness',model_fitness.shape)
        print(data['fitness'])
        #update the fitness for the current model (indexed by 'id')
        data['fitness'][task_id,:] = model_fitness.flatten()
        data['KS_stats_up'][task_id,:] = average_KS_UP.flatten()
        data['KS_stats_down'][task_id,:] = average_KS_DOWN.flatten()
        scipy.io.savemat(gen_filename,data)
    return model_fitness

# This will only run if the script is executed directly (not imported)
if __name__ == "__main__":
    main()