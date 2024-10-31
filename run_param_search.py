import subprocess
import numpy as np 
import scipy.io as sio
from full_cortex_both import *
import time
import os





# create empty .mat file to store future simulation outputs and parameters
def create_mat_file(filename):
    # create empty placeholders for parameters  and fitness
    empty_params = np.zeros((50,10)) # 50 models with 10 params varying
    empty_fitness = np.zeros((50,3)) # 50 models with 2 fitness values each
    sio.savemat(filename,{'params': empty_params, 'fitness': empty_fitness})




# generates the sbatch files to paralellise simulations at each generation of the evo algorithm
def create_and_submit_sbatch(filename, job_name, num_tasks, output_dir, error_dir, command):
    os.makedirs(output_dir, exist_ok= True)
    os.makedirs(error_dir, exist_ok = True)
    # Create .sbatch file content
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array=0-{num_tasks-1} 
##SBATCH --ntasks={num_tasks}
#SBATCH --output={output_dir}/output_%A_%a.txt
#SBATCH --error={error_dir}/error_%A_%a.txt
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G   


# Load any necessary modules here
# module load python/3.8

# Execute the command, passing the task ID ($SLURM_ARRAY_TASK_ID) to the script
{command} $SLURM_ARRAY_TASK_ID {job_name}.mat
"""
    #write sbatch file
    with open(filename, 'w') as f:
        f.write(sbatch_content)
    
    print(f"Created {filename} with job: {job_name}")
    
    # Submit the job
    result = subprocess.run(['sbatch', filename], stdout=subprocess.PIPE, text=True)
    
    if result.returncode == 0:
        # Parse the output to extract the jobID
        output = result.stdout.strip()
        job_id = output.split()[-1]  # The job ID is typically the last part of the output
        print(f"Submitted job array with job ID: {job_id}")
        return job_id
    else:
        # If there's an error submitting the job, print the error and return None
        print(f"Error submitting job: {result.stderr.decode('utf-8')}")
        return None


def wait_for_jobs_completion(job_id):
    # Poll squeue to check ifthe job is still in the queue
    print(f"Waiting for 1 minutes before the first status check for job array {job_id}...")
    time.sleep(600) # 30 mins = 1800 seconds
    while True:
        #Use squeue to check the status of the job
        result = subprocess.run(['squeue', '--job', job_id], shell=True, capture_output = True, text = True)
        if result.returncode != 0:
            print(f"Error checking job status: {result.stderr}")
            return False # Exit on error
             
        # If the job ID is no longer in the squeue output, assume it's completed
        if job_id not in result.stdout:
            print(f"Job {job_id} completed.")
            break
    
        # Wait for a bit before checking again
        time.sleep(600)


generation = 0
max_generations = 5 # change this to max n. generations -- to do


while generation < max_generations:
    #define generation-specific folder paths
    output_dir = f"/scratch/ec4513/evo_algo/gen_{generation}/output"
    error_dir = f"/scratch/ec4513/evo_algo/gen_{generation}/error"
    if generation == 0:
        # create gen_0.mat file to store simulation outputs
        gen_filename = f"gen_{generation}.mat"
        model_params =  random_parms()
        # Save initial parameters to the mat file
        sio.savemat(gen_filename, {'params': model_params, 'fitness': np.zeros((50, 3))})  # Assuming fitness is 2D
        #simulate models and save the outputs
        job_id = create_and_submit_sbatch(f"generation_{generation}.sbatch", f"gen_{generation}", 50, output_dir, error_dir, "python full_cortex_both.py")
    else:
        # load previous generation's mat file
        prev_gen_filename =f"gen_{generation -1}.mat"
        prev_gen_data = sio.loadmat(prev_gen_filename)
        old_model_params = prev_gen_data['params']
        old_fitness = prev_gen_data['fitness']
        # generate new parameters with some function (to do)
        model_params = generate_new_population(old_model_params,old_fitness)
        
        gen_filename = f"gen_{generation}.mat"
        # Save initial parameters to the mat file
        sio.savemat(gen_filename, {'params': model_params, 'fitness': np.zeros((50, 3))})  # Assuming fitness is 2D
        
        #simulate models and save outputs
        job_id =  create_and_submit_sbatch(f"generation_{generation}.sbatch", f"gen_{generation}", 50, output_dir, error_dir, "python full_cortex_both.py")
    
    # Wait for the current generation's jobs to complete before moving to the next generation
    if job_id is not None: # to do
        wait_for_jobs_completion(job_id)
        
    generation +=1
