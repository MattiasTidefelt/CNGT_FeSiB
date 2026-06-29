#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 14:16:50 2025

@author: ag0406

Parallel crystallization modeling for the Fe-Si-B system

"""

from mpi4py import MPI
import numpy as np
from os import makedirs
from os.path import exists
import h5py
import pickle

# from MPI_NucFeSiB import MPI_Nuc_FeSiB
from MPI_NucFeSiB_HeatQuench import MPI_Nuc_FeSiB_HQ 
from MPI_AM_FeSiB import MPI_Nuc_FeSiB_AM
from MPI_Iso_FeSiB import MPI_Nuc_FeSiB_Iso

def master(n_workers, CompsAndHR):
    comm = MPI.COMM_WORLD
    status = MPI.Status()
    
    nComps = CompsAndHR.shape[0]
    nit = np.zeros(nComps)
    trackers = [None]*nComps
    for i in range(nComps):
        trackers[i]=f'{i:03}'  

    data_sent = 0
    data_recv = 0

    # Keep track of the indices of the next data element for each worker
    next_indices = {i: i for i in range(1, n_workers + 1)}
    
    # Keep track of available workers and their status
    available_workers = set(range(1, n_workers + 1))
    worker_status = {worker: "idle" for worker in available_workers}
        

        ################################### One file
    with h5py.File(f'{DirPathOut}/CrystallizationData.hdf5', 'w') as fh:
        failedRuns = 0
        while data_recv < nComps:           

            # Check if there is available worker                
            if comm.Iprobe(source=MPI.ANY_SOURCE, tag=2):  
                # print('//////////////////// TAG 2 //////////////////////')
                msg = np.array([1])
                comm.Recv([msg, MPI.INT], source=MPI.ANY_SOURCE, tag=2, status=status) 
                worker_rank = status.Get_source()
            
                sendFlag = True
                while sendFlag:
                    # Check if any worker has sent data size
                    if comm.Iprobe(source=worker_rank, tag=3): 
                        # print('//////////////////// TAG 3 //////////////////////')
                        data_size = np.array([1])
                        comm.Recv([data_size, MPI.INT], source=worker_rank, tag=3, status=status)
                        
                        
                        recvFlag = True
                        while recvFlag:
                            # Check if any worker has sent data
                            if comm.Iprobe(source=worker_rank, tag=4):
                                # print('//////////////////// TAG 4 //////////////////////')
                                serialized_data = bytearray(data_size[0])
                                comm.Recv([serialized_data, MPI.BYTE], source=worker_rank, tag=4)
                                
                                result_data = pickle.loads(serialized_data)
                                nit[result_data["id"]] = result_data["nit"]
                                data_recv += 1                         
                                
                                # Send new data to worker
                                if data_sent < nComps:                              
                                    data_set = np.array(int(trackers[data_sent]))
                                     
                                    comm.Send([data_set,MPI.INT], dest=worker_rank, tag=1)
                                    
                                    next_indices[worker_rank] += 1
                                    data_sent += 1
        
                                    print(f"# Root has sent pattern {data_sent}:{nComps} to rank {worker_rank}")
                                else:                         
                                    print(f'Free worker, rank {worker_rank}')
                                    worker_status[worker_rank] = "idle"
                                    available_workers.add(worker_rank)
                                
                                # if result_data["nit"] != 0:
                                # Store data in a HDF5 file
                                group_id = str(result_data["id"])
                                grp = fh.create_group(group_id)
                                grp.create_dataset("t", data=result_data["t"])
                                grp.create_dataset("T", data=result_data["T"])
                                grp.create_dataset("ftot", data=result_data["ftot"])
                                grp.create_dataset("f", data=result_data["f"])
                                grp.create_dataset("rm", data=result_data["rm"])
                                grp.create_dataset("nit", data=result_data["nit"])
                            
                                print(f'# Data received and saved: {data_recv}:{nComps}') 
                                
                                if result_data["nit"] == 0:
                                    failedRuns += 1
                                    
                                recvFlag = False
                                sendFlag = False      
                                ###################################
                                ###################################
                                ###################################
                
            # Check if there is data to send to a worker
            ################################### One file
            if  available_workers and data_sent < nComps:
            ################################### Many files
        
                # Find an available worker
                worker_rank = available_workers.pop()
                  
                data_set = np.array(int(trackers[data_sent]))
                 
                comm.Send([data_set,MPI.INT], dest=worker_rank, tag=1)
                
                next_indices[worker_rank] += 1
                data_sent += 1
                
                worker_status[worker_rank] = "busy"
                print(f"# Root has sent data to rank {worker_rank}")
                
        
        print('------------------------------------------------------') 
        print(f'iterations: min {np.min(nit)}, max {np.max(nit)}, mean {np.mean(nit):1.1f}')
        print(f'Failed runs: {failedRuns}')
        print('------------------------------------------------------') 
        for i in range(1,n_workers+1):
            comm.Send([np.array([-1]),MPI.INT], dest=i, tag=1)
             
        
def worker(CompsAndHR,Htype):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    while True:
        recv_data = np.array([1])
        comm.Recv([recv_data, MPI.INT], source=0, tag=1)
        
        if recv_data == -1:
            print(f' #######################  Stop worker {rank}  ####################### ')
            break
        
        else:
            print(f"# Pattern id {recv_data[0]+1}:{len(CompsAndHR)} received to rank {rank}")
            SimData = CompsAndHR[recv_data[0]]
            x0 = SimData[:3]
            HR = SimData[3]

            # if Htype == "AM":
            #     [Time,T,ftot,f,Rmean,nit] = MPI_Nuc_FeSiB_AM(x0,HR)
            # elif Htype == "Heating" or Htype == "Quenching":
            #     [Time,T,ftot,f,Rmean,nit] = MPI_Nuc_FeSiB_HQ(x0,HR)
            # elif Htype == "Iso":
            #     [Time,T,ftot,f,Rmean,nit] = MPI_Nuc_FeSiB_Iso(x0,HR)    # Hr is Iso T in this case

            try:
                if Htype == "AM":
                    [Time,T,ftot,f,Rmean,nit] = MPI_Nuc_FeSiB_AM(x0,HR)
                elif Htype == "Heating" or Htype == "Quenching":
                    [Time,T,ftot,f,Rmean,nit] = MPI_Nuc_FeSiB_HQ(x0,HR)
                elif Htype == "Iso":
                    [Time,T,ftot,f,Rmean,nit] = MPI_Nuc_FeSiB_Iso(x0,HR)    # Hr is Iso T in this case
            except:
                print("----------------------------------")
                print(f"Failed to run {x0}, storing zeros")
                print("----------------------------------")
                Time = 0
                T = 0
                ftot = 0
                f = np.zeros(11)
                Rmean = f.copy()
                nit = 0
                
    
            # #--------------------------- request a new job -------------------------------            
            msg = np.array([1])
            comm.Send([msg,MPI.INT], dest=0, tag=2)
            
            ################################### One file
            result_data = {
                "t": Time,
                "T": T,
                "ftot": ftot,
                "f": f,
                "rm": Rmean,
                "nit": nit,
                "id": recv_data[0],
            }
            
            serialized_data = pickle.dumps(result_data)
            data_size = np.array([len(serialized_data)])
            
            comm.Send([data_size, MPI.INT], dest=0, tag=3)
            comm.Send([bytearray(serialized_data), MPI.BYTE], dest=0, tag=4)
            print(f"# Rank {rank} fitted pattern {recv_data[0]+1}:{len(CompsAndHR)}")
            
            ################################### 
            
if __name__ == '__main__':        

    # Htype = "Heating"   #55 min, 86 now, 68 now!, large on atom6: 15 min
    # Htype = "Quenching"   # 10.3 h, 12 now, 165 min now!!! (small 365......), large on atom6: 324 cpuh on atom6 
    # Htype = "AM"      # 49 h (19 h for small now....), 960 cpuh on atom6
    Htype = "Iso"      # 5 h
    # --------------------  Get data sets ------------------------------  
    DirPathIn = "/Users/ag0406/Documents/Mattias/PhD/CNT/FeBSi/MPI_Sampling/"

    with h5py.File(f'{DirPathIn}/CompData_{Htype}.hdf5', 'r') as fh:
        CompsAndHR = np.array([fh["CompData"]])[0]
        
    DirPathOut = f"{DirPathIn}{Htype}/"
    if not exists(DirPathOut):
        makedirs(DirPathOut)   
    
    simt=MPI.Wtime()
    #--------------------  Initiate MPI ------------------------------
    comm= MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    n_workers = size-1   
    
    #-----------------------------------------------------------------       
    if rank == 0:
        master(n_workers, CompsAndHR)
    else:
        worker(CompsAndHR,Htype)

    print(f' #######################  Thread {rank} finished ####################')
    
    if rank == 0:
        print('------------------------------------------------------') 
        simtime=str(round((MPI.Wtime()-simt)/60,3))
        print(f'Computation time: {simtime} min')
        print('------------------------------------------------------')  
    MPI.Finalize()