from mpi4py import MPI
import numpy as np
import math
from matplotlib import pyplot as plt
from time import time

# MPI Initialization
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Constants for the problem
rows = 1000
col = 1000
rows_per_pe = rows // size  # Each PE gets 250 rows for 4 PEs
max_temp_error= 0.01

# check if number of pe equal to 4
if size != 4:
    if rank == 0:
        print("This program requires exactly 4 processes.")
    MPI.Finalize()
    exit()

# Function to initialize temperature with boundary conditions
def initialize_temperature(temp, r_local):
    temp[:, :] = 0
    # Set up Top boundary condition
    if rank == 0:  
        temp[0, :] = 0  # Top value is set to 0
    # Set up bottom boundary condition ,
    # This condition only applies for the last pe which is 3 in our case
    if rank == size - 1:  
        for i in range(col + 2):
            temp[r_local + 1, i] = 100 * math.sin(((math.pi / 2) / col) * i)
    # All pe executes right boundary condition for their portion individually
    for i in range(r_local + 2):
        global_i = i + rank * rows_per_pe
        temp[i, col + 1] = 100 * math.sin(((math.pi / 2) / rows) * global_i)

# Function to output data
def output(data):
    plt.imshow(data)
    plt.colorbar()
    #the output is saved as a png file for visualization
    plt.savefig("plate_parallel2.png")
    data.tofile("plate_parallel2.out")

# Initializing local temperature arrays with ghost rows
r_local = rows_per_pe
temperature = np.zeros((r_local + 2, col + 2))  # Local grid including ghost rows
temperature_last = np.zeros_like(temperature)

# Initialize boundary conditions for each process
initialize_temperature(temperature_last, r_local)
temperature = temperature_last.copy()
#.copy() belongs to numpy library it helps in optimizing.
# The other method is to assign it element by element in a loop which will slow down the process
# Maximum iterations
max_iterations = 10000 # can be anything greater than 4000 since the code converges at 3578

if rank == 0:
    max_iterations = int(input("Maximum iterations: "))#Getting the max interation input using only pe 0 
max_iterations = comm.bcast(max_iterations, root=0)#bcast makes it available for other pe 

# Timer-is done to calculate total run time of the code
start_time = time()

# Main iteration loop
dt_global = 100.0
iteration = 0

while dt_global > max_temp_error and iteration < max_iterations:
    # Main calculation: averaging the four neighbors
    for i in range(1, r_local + 1):
        for j in range(1, col + 1):
            temperature[i, j] = 0.25 * (
                temperature_last[i + 1, j] + temperature_last[i - 1, j] +
                temperature_last[i, j + 1] + temperature_last[i, j - 1]
            )
    
    # COMMUNICATION PHASE: exchange ghost rows with neighbors 
    if rank < size - 1:#for pe-0,pe-1,pe-2
        comm.Send(temperature[r_local, :], dest=rank + 1) #pe0 sends its last row to pe 1,pe1 to pe2, pe2 to pe3
        comm.Recv(temperature[r_local + 1, :], source=rank + 1)#pe 0 receives the first row of pe1 similarly for other pe
    if rank > 0:#for pe 1,pe2,pe3
        comm.Send(temperature[1, :], dest=rank - 1)#pe 1 sends its first row to pe0 similarly for other pe
        comm.Recv(temperature[0, :], source=rank - 1)#pe 1 receives pe0's last row similarly for other pe

    # Calculate local maximum temperature difference (dt)
    dt_local = np.max(np.abs(temperature - temperature_last))
    #np.max and np.abs are optimized function in numpy ,
    # this step computes the local temperature difference across the entire grid in one operation, 
    # this process can be done manually  wiht the below syntax in loop but when we run the code it takes longer time 
    #it took 30 mins aprox to run the code with the loop and 17 mins aprox when np.max and np.abs were used
    '''
    dt_local=0
    for i in range(1, local_rows + 1):
        for j in range(1, COLUMNS + 1):
           dt_local= max(dt_local,abs(temperature[i,j]-temperature_last[i,j]))
           temperature_last[i,j] - temperature[i,j]
    
    '''
    # Reduce the local dt to get the global dt
    dt_global = comm.allreduce(dt_local, op=MPI.MAX)

    # Update temperature_last with new values
    temperature_last = temperature.copy()

    # Print for every 100 iteration progress on rank 0
    if rank == 0 and iteration % 100 == 0:
        print(f"Iteration {iteration}: dt = {dt_global:.6f}",flush= True)
    #print(iteration)
    iteration += 1

# Output the final result after the last iteration
if rank == 0:
    end_time = time()
    print(f"Completed in {iteration} iterations.")
    print(f"Total time: {end_time - start_time:.2f} seconds.")

# Gather results from all pe 
finalop= None
if rank == 0:
    finalop = np.empty((rows, col + 2), dtype=np.float64)

senddata = temperature[1:-1, :]
comm.Gather(senddata, finalop, root=0)

# Displaying output
if rank == 0:
    output(finalop[:, 1:])  # Removing ghost columns
    print("Output saved as plate_parallel2.png and plate_parallel2.out")

MPI.Finalize()