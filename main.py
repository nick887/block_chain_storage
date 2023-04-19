import Q2_GA
import Q2_CA
import Q2_PSO
import matplotlib.pyplot as plt
import numpy as np
import Q2_PSO_GA
import time

iteration_num = 10000
nodeNum = 30
blockNum = 100

if __name__=="__main__":
    GA_epoch_result,_ = Q2_GA.evaluate(
        time.time(),
        nodeNum=nodeNum,
        blockNum=blockNum,
        storageLimit=0.65, 
        alloNum=10, 
        epochNum=iteration_num, 
        blockBackups=4, 
        node_cost_ratio=1)

    CA_epoch_result,_ = Q2_CA.evaluate(
        nodeNum=nodeNum,
        blockNum=blockNum,
        storageLimit=0.65,
        epochNum=iteration_num,
        blockBackups=4,
        initial_temp=0.1,
        final_temp=0.0001,
        cooling_rate=0.999999,
        node_cost_ratio=1,)

    PSO_epoch_result,_ = Q2_PSO.evaluate(
        time.time(),
        nodeNum=nodeNum,
        blockNum=blockNum,
        storageLimit=0.65,
        n_iterations=iteration_num,
        n_particles=10,
        blockBackups=4,
        node_cost_ratio=1,)
    PSO_GA_epoch_result,_ = Q2_PSO_GA.evaluate(
        nodeNum=nodeNum,
        blockNum=blockNum,
        storageLimit=0.65,
        n_iterations=iteration_num,
        n_particles=10,
        blockBackups=4,
        alloNum=10,
        node_cost_ratio=1,
    )


    #%%
    PSO_result = Q2_PSO.getResult(
        blockNum=100,
        nodeNum=30,
        blockBackups=4,
        storageLimit=0.65,
        n_particles=10,
        n_iterations=iteration_num,
    )

    CA_result = Q2_CA.getResult(
        blockNum=100,
        nodeNum=30,
        blockBackups=4,
        storageLimit=0.65,
        epochNum=iteration_num,
    )

    GA_result = Q2_GA.getResult(
        blockNum=100,
        nodeNum=30,
        blockBackups=4,
        storageLimit=0.65,
        alloNum=10,
        epochNum=iteration_num,
    )

    PSO_GA_result = Q2_PSO_GA.getResult(
        blockNum=100,
        nodeNum=30,
        blockBackups=4,
        storageLimit=0.65,
        alloNum=10,
        epochNum=iteration_num,
    )

    GA_worker1_result = Q2_GA.getResultWorker1(
        blockNum=100,
        nodeNum=30,
        blockBackups=4,
        storageLimit=0.65,
        alloNum=10,
        epochNum=iteration_num,
    )
    GA_worker10_result = Q2_GA.getResultWorker10(
        blockNum=100,
        nodeNum=30,
        blockBackups=4,
        storageLimit=0.65,
        alloNum=10,
        epochNum=iteration_num,
    )


    fig_target_value,ax = plt.subplots(figsize=(6,4), dpi= 1080)
    fig_communication_cost,bx = plt.subplots(figsize=(6,4), dpi= 1080)
    fig_node_storage_proportion,cx = plt.subplots(figsize=(6,4), dpi= 1080)
    fig_target_value_t,axt = plt.subplots(figsize=(6,4), dpi= 1080)
    fig_communication_cost_t,bxt = plt.subplots(figsize=(6,4), dpi= 1080)
    fig_node_storage_proportion_t,cxt = plt.subplots(figsize=(6,4), dpi= 1080)
    fig_target_compare_with_workers,axc = plt.subplots(figsize=(6,4), dpi= 1080)
    fig_target_compare_with_pso_pso_ga,axb = plt.subplots(figsize=(6,4), dpi= 1080)

    ax.plot(range(iteration_num+1), PSO_result[:,0],'-',label='PSO target')
    ax.plot(range(iteration_num+1), GA_result[:,0],'-',label='GA target')
    ax.plot(range(iteration_num+1), CA_result[:,0],'-',label='CA target')
    ax.legend()

    bx.plot(range(iteration_num+1), PSO_result[:,1],'-',label='PSO communication cost')
    bx.plot(range(iteration_num+1), GA_result[:,1],'-',label='GA communication cost')
    bx.plot(range(iteration_num+1), CA_result[:,1],'-',label='CA communication cost')
    bx.legend()


    cx.plot(range(iteration_num+1), PSO_result[:,2],'-',label='PSO node storage proportion')
    cx.plot(range(iteration_num+1), GA_result[:,2],'-',label='GA node storage proportion')
    cx.plot(range(iteration_num+1), CA_result[:,2],'-',label='CA node storage proportion')
    cx.legend()

    stacked = np.vstack((PSO_result, GA_result, CA_result))
    max_values = np.min(stacked[:,3][-3:])
    print(max_values)

    stacked1 = np.vstack((GA_worker1_result, GA_worker10_result))
    max_values1 = np.min(stacked1[:,3][-2:])
    print(max_values1)

    stacked2 = np.vstack((PSO_GA_result, GA_result))
    max_values2 = np.max(stacked2[:,3][-2:])
    print(max_values2)


    axt.plot(PSO_result[PSO_result[:,3]<max_values,3], PSO_result[PSO_result[:,3]<max_values,0],'-',label='PSO target value')
    axt.plot(GA_result[GA_result[:,3]<max_values,3], GA_result[GA_result[:,3]<max_values,0],'-',label='GA target value')
    axt.plot(CA_result[CA_result[:,3]<max_values,3], CA_result[CA_result[:,3]<max_values,0],'-',label='CA target value')
    axt.legend()

    bxt.plot(PSO_result[PSO_result[:,3]<max_values,3], PSO_result[PSO_result[:,3]<max_values,1],'-',label='PSO communication cost')
    bxt.plot(GA_result[GA_result[:,3]<max_values,3], GA_result[GA_result[:,3]<max_values,1],'-',label='GA communication cost')
    bxt.plot(CA_result[CA_result[:,3]<max_values,3], CA_result[CA_result[:,3]<max_values,1],'-',label='CA communication cost')
    bxt.legend()

    cxt.plot(PSO_result[PSO_result[:,3]<max_values,3], PSO_result[PSO_result[:,3]<max_values,2],'-',label='PSO node storage proportion')
    cxt.plot(GA_result[GA_result[:,3]<max_values,3], GA_result[GA_result[:,3]<max_values,2],'-',label='GA node storage proportion')
    cxt.plot(CA_result[CA_result[:,3]<max_values,3], CA_result[CA_result[:,3]<max_values,2],'-',label='CA node storage proportion')
    cxt.legend()

    axc.plot(GA_worker10_result[GA_worker10_result[:,3]<max_values1,3], GA_worker10_result[GA_worker10_result[:,3]<max_values1,0],'-',label='GA workers 10 target value')
    axc.plot(GA_worker1_result[GA_worker1_result[:,3]<max_values1,3], GA_worker1_result[GA_worker1_result[:,3]<max_values1,0],'-',label='GA workers 1 target value')
    axc.legend()

    axb.plot(GA_result[GA_result[:,3]<max_values2,3], GA_result[GA_result[:,3]<max_values2,0],'-',label='GA target value')
    axb.plot(PSO_GA_result[PSO_GA_result[:,3]<max_values2,3], PSO_GA_result[PSO_GA_result[:,3]<max_values2,0],'-',label='PSO-GA target value')
    axb.legend()

    fileSaveName = f"Q2-iteration-plot-{nodeNum}-{blockNum}-{iteration_num}"

    fig_target_value.savefig('./IterationChart/Q2/'+fileSaveName+'-target_value.svg',dpi=300,format='svg',bbox_inches = 'tight')
    fig_communication_cost.savefig('./IterationChart/Q2/'+fileSaveName+'-communication_cost.svg',dpi=300,format='svg',bbox_inches = 'tight')
    fig_node_storage_proportion.savefig('./IterationChart/Q2/'+fileSaveName+'-node_storage_proportion.svg',dpi=300,format='svg',bbox_inches = 'tight')
    fig_communication_cost_t.savefig('./IterationChart/Q2/'+fileSaveName+'-communication_cost_t.svg',dpi=300,format='svg',bbox_inches = 'tight')
    fig_target_value_t.savefig('./IterationChart/Q2/'+fileSaveName+'-target_value_t.svg',dpi=300,format='svg',bbox_inches = 'tight')
    fig_node_storage_proportion_t.savefig('./IterationChart/Q2/'+fileSaveName+'-node_storage_proportion_t.svg',dpi=300,format='svg',bbox_inches = 'tight')
    fig_target_compare_with_workers.savefig('./IterationChart/Q2/'+fileSaveName+'-target_compare_with_worker.svg',dpi=300,format='svg',bbox_inches = 'tight')
    fig_target_compare_with_pso_pso_ga.savefig('./IterationChart/Q2/'+fileSaveName+'-target_compare_with_pso_pso_ga.svg',dpi=300,format='svg',bbox_inches = 'tight')

    # TODO 不同节点和区块数量下算法性能对比
    # TODO 不同迭代次数下对比
    # TODO 不同种群数量对比
    # TODO 相同算法在不同节点和区块数量下对比
    # GPU加速对比