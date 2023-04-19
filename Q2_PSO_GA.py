import Q2_GA
import Q2_PSO
import numpy as np
import time

def evaluate(
    nodeNum, 
    blockNum, 
    storageLimit, 
    alloNum, 
    blockBackups, 
    n_iterations, 
    n_particles, 
    node_cost_ratio = 1):
    start_time = time.time()
    fileSaveName = 'PSO_GA_B{0}_N{1}_BU{2}_L{3}_AN{4}_E{5}'.format(blockNum,nodeNum,blockBackups,storageLimit,alloNum,n_iterations)
    PSO_epoch_result,best_solution = Q2_PSO.evaluate(
    start_time=start_time,
    nodeNum=nodeNum,
    blockNum=blockNum,
    storageLimit=storageLimit,
    n_iterations=int(n_iterations*0.05),
    n_particles=n_particles,
    blockBackups=blockBackups,
    node_cost_ratio=node_cost_ratio,)
    GA_epoch_result,best_solution = Q2_GA.evaluate(
        start_time=start_time,
        nodeNum=nodeNum,
        blockNum=blockNum,
        storageLimit=storageLimit, 
        alloNum=alloNum, 
        epochNum=int(n_iterations * 0.95), 
        blockBackups=blockBackups, 
        node_cost_ratio=node_cost_ratio,
        best_solution=best_solution)
    
    result = np.concatenate([PSO_epoch_result, GA_epoch_result], axis=0)
    np.savez('./IterationChart/Q2/'+fileSaveName+'-alloEpoch.npz', alloEpoch=result)
    return  result,best_solution

def getResult(blockNum,nodeNum,blockBackups,storageLimit,alloNum,epochNum):
    fileSaveName = 'PSO_GA_B{0}_N{1}_BU{2}_L{3}_AN{4}_E{5}'.format(blockNum,nodeNum,blockBackups,storageLimit,alloNum,epochNum)
    result = np.load('./IterationChart/Q2/'+fileSaveName+'-alloEpoch.npz')
    return result['alloEpoch']