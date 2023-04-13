import numpy as np
import time
import matplotlib.pyplot as plt
import random
import torch
import math
from concurrent.futures import ThreadPoolExecutor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Particle:
    def __init__(self, position, velocity, cost_func):
        self.position = position
        self.velocity = velocity
        self.cost_func = cost_func
        self.cost = cost_func(position)
        if device == "cuda":
            self.best_position = torch.clone(position).cuda()
        else:
            self.best_position = torch.clone(position)
        self.best_cost = self.cost

    def update_velocity(self, global_best_position,current_iteration, max_iterations, w_max=1.8, w_min=0.4,c1=2, c2=2):
        w = w_max - (w_max - w_min) * (current_iteration / max_iterations)
        r1, r2 = random.random(), random.random()
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        # print("diff: ",cognitive + social)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, global_best_position, current_iteration, max_iterations):
        self.position = self.position + self.velocity
        self.position = 1 / (1 + torch.exp(-self.position))  # 使用 sigmoid 函数将 position 转换为 0 到 1 之间的值
        self.position = (self.position > 0.5).float()  # 将 position 转换为布尔值（0 或 1）
        for i in range(self.position.shape[1]):
            block_distribution = self.position[:,i]
            while torch.sum(block_distribution) > 4:
                block_distribution[torch.argmax(block_distribution)] = 0
        # print(self.position)
        while(np.all(constraint(self.position)) == False):
            # print("不满足约束条件")
            self.update_velocity(global_best_position, current_iteration, max_iterations)
            self.position = self.position + self.velocity
            self.position = 1 / (1 + torch.exp(-self.position))  # 使用 sigmoid 函数将 position 转换为 0 到 1 之间的值
            self.position = (self.position > 0.5).float()  # 将 position 转换为布尔值（0 或 1）


    def update_cost(self):
        self.cost = self.cost_func(self.position)
        if self.cost < self.best_cost:
            self.best_position = torch.clone(self.position)
            self.best_cost = self.cost

def constraint(blockAllo, condition = -1): #-1代表所有条件,0-2代表条件1-3
    if(condition == -1):
        constraint_result = np.full(3, False, dtype=bool) #三个约束的结果 默认全False
        #约束条件1 每个节点的空间限制
        # print(torch.all(torch.matmul(blockAllo,blockInfo[:,1]) <= nodeLimit))
        # print(torch.all(torch.sum(blockAllo,dim=0) >= blockBackups))
        # print((torch.sum(blockAllo.sum(axis=0) * blockInfo[:, 1]) / torch.sum(nodeLimit)) <= storageLimit)
        constraint_result[0] = torch.all(torch.matmul(blockAllo,blockInfo[:,1]) <= nodeLimit) #为True即为节点均满足要求
        #约束条件2 每个区块至少有blockBackups备份
        constraint_result[1] = torch.all(torch.sum(blockAllo,dim=0) >= blockBackups) #为True即为满足备份数量要求
        #约束条件3 系统总的存储空间占用率不超过限定值
        constraint_result[2] = (torch.sum(blockAllo.sum(axis=0) * blockInfo[:, 1]) / torch.sum(nodeLimit)) <= storageLimit
        return constraint_result
    
    elif(condition == 0):
        return torch.all(torch.matmul(blockAllo, blockInfo[:, 1]) <= nodeLimit)
    
    elif(condition == 1):
        return torch.all(blockAllo.sum(axis=0) >= blockBackups)
    
    elif(condition == 2):
        return torch.sum(blockAllo.sum(axis=0) * blockInfo[:, 1]) / torch.sum(nodeLimit) <= storageLimit

def initial_solution():
    if device == "cuda":
        blockAllo = torch.zeros((nodeNum, blockNum), dtype=torch.float32).cuda()
    else:
        blockAllo = torch.zeros((nodeNum, blockNum), dtype=torch.float32)

    while np.all(constraint(blockAllo)) == False:
        if device == "cuda":
            blockAllo = torch.zeros((nodeNum, blockNum), dtype=torch.float32).cuda()
        else:
            blockAllo = torch.zeros((nodeNum, blockNum), dtype=torch.float32)
        for block in range(blockNum):
            nodeChoice = random.sample(range(nodeNum), np.random.randint(blockBackups, blockBackups + 1))
            blockAllo[nodeChoice, block] = 1
    return blockAllo

def fitness(solution):
    nodeBlockCost = torch.zeros((nodeNum, blockNum),dtype=torch.float32)

    for node in range(nodeNum):
        for block in range(blockNum):
            nodeBlockCost[node, block] = torch.min(costAll[block, solution[:, block] == 1, node])

    nodeRatio = torch.matmul(solution, blockInfo[:, 1]) / nodeLimit
    nodeProportion = (torch.exp(5 * nodeRatio) - 1) / (math.exp(5) - 1)

    nodeBlockCostAvg = torch.sum(nodeBlockCost) / (nodeNum * blockNum)
    nodeProportionAvg = torch.sum(nodeProportion) / nodeNum * 50

    return nodeBlockCostAvg + nodeProportionAvg * node_cost_ratio

def objective(solution):
    nodeBlockCost = torch.zeros((nodeNum, blockNum),dtype=torch.float32)

    for node in range(nodeNum):
        for block in range(blockNum):
            nodeBlockCost[node, block] = torch.min(costAll[block, solution[:, block] == 1, node])

    nodeRatio = torch.matmul(solution, blockInfo[:, 1]) / nodeLimit
    nodeProportion = (torch.exp(5 * nodeRatio) - 1) / (math.exp(5) - 1)

    nodeBlockCostAvg = torch.sum(nodeBlockCost) / (nodeNum * blockNum)
    nodeProportionAvg = torch.sum(nodeProportion) / nodeNum * 50
    result = np.array([nodeBlockCostAvg+nodeProportionAvg*node_cost_ratio, nodeBlockCostAvg, nodeProportionAvg*node_cost_ratio])
    return result

def pso(n_particles, n_iterations, fitness):
    particles = []
    for _ in range(n_particles):
        position = initial_solution()
        if device == "cuda":
            velocity = (torch.rand(nodeNum, blockNum, dtype=torch.float32) - 0.5).cuda()
        else:
            velocity = (torch.rand(nodeNum, blockNum, dtype=torch.float32) - 0.5)
        particles.append(Particle(position, velocity, fitness))

    global_best_position = torch.clone(particles[0].position)
    global_best_cost = particles[0].cost

    for particle in particles[1:]:
        if particle.cost < global_best_cost:
            global_best_position = torch.clone(particle.position)
            global_best_cost = particle.cost

    print(global_best_cost)
    def update_particle(particle):
        particle.update_velocity(global_best_position, iteration, n_iterations)
        particle.update_position(global_best_position, iteration, n_iterations)
        particle.update_cost()
        return particle

    no_improvement_count = 0
    with ThreadPoolExecutor() as executor:
        for iteration in range(n_iterations):
            particles = list(executor.map(update_particle, particles))
            for particle in particles:
                # particle.update_velocity(global_best_position, iteration, n_iterations)
                # particle.update_position(global_best_position, iteration, n_iterations)
                # particle.update_cost()

                if particle.cost < global_best_cost:
                    global_best_position = torch.clone(particle.position)
                    global_best_cost = particle.cost
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

            if no_improvement_count >= n_particles:
                print("shuffle")
                for particle in particles:
                    position = initial_solution()
                    position = (position > 0.5).float()
                    if device == "cuda":
                        velocity = torch.rand(nodeNum, blockNum, dtype=torch.float32).cuda() - 0.5
                    else:
                        velocity = torch.rand(nodeNum, blockNum) - 0.5
                    particle.position = position
                    particle.velocity = velocity
                    particle.cost = fitness(position)
                no_improvement_count = 0
            
            r = objective(global_best_position)
            # print(iteration,r)
            alloEpoch[iteration] = r
        return global_best_position



if __name__=="__main__":

    start_time = time.time()
    blockNum = 100 #区块数量
    nodeNum = 30 #节点数量
    blockBackups = 4 #区块备份数量 至少为1
    storageLimit = 0.65 #优化后系统总存储空间占比
    storageLimitLoad = 0.5 #从Q1加载的限制区块数据量
    saveResult = True #是否保存数据 True False
    expfromQ1 = False
    n_particles = 10
    n_iterations = 100
    node_cost_ratio = 1

    #区块信息
    blockInfo = np.loadtxt('./Data/blockInfo{}.csv'.format(blockNum), delimiter=',')
    
    #节点信息
    nodeLimit = np.loadtxt('./Data/NodeInfo/nodeLimit-{}.csv'.format(nodeNum), delimiter=',')
    nodeOptDis = np.load('./Data/NodeInfo/nodeRouteInfo-{}-30-100-100.npz'.format(nodeNum))['arr_0']
    nodeStorage = np.sum(nodeLimit) #CU本地存储的空间和
    storageLimitcondition = np.sum(nodeLimit)*storageLimit #CU所有节点的空间限制    

    print('Start...') 

    fileSaveName = 'PSO_B{0}_N{1}_BU{2}_L{3}_AN{4}_E{5}'.format(blockNum,nodeNum,blockBackups,storageLimit,n_particles,n_iterations)

    print('Allocate {0} blocks to {1} peers.'.format(blockNum,nodeNum))
    print('Block backup is {}.'.format(blockBackups))
    print('Total block storage size is {}'.format(np.sum(blockInfo[:,1])))
    print('Total node storage limit is {}'.format(np.sum(nodeLimit)))
    #节点允许的空间限制 与 区块总大小 的比值
    nodeBlockRatio = np.sum(nodeLimit)/np.sum(blockInfo[:,1])*storageLimit
    print('Ratio of node limit to block size is {}'.format(nodeBlockRatio))

    if (nodeBlockRatio<=blockBackups*1.2):
        print("storageLimit is too small, there's not enough space")    
    else:
        alloEpoch = np.zeros((n_iterations,3)) #每轮训练的最优结果 总目标 通信成本 存储平衡度 
        print('Storage optimization target is {}.'.format(storageLimit))
            #所有节点需要的花销shape=(blockNum,nodeNum,nodeNum)

        costAll = (blockInfo[:,1]*blockInfo[:,2]).reshape((blockNum,1,1)) * nodeOptDis.reshape((1,nodeNum,nodeNum))

        if device == "cuda":
            costAll = torch.from_numpy(costAll.astype(np.float32)).cuda()
            blockInfo = torch.from_numpy(blockInfo.astype(np.float32)).cuda()
            nodeOptDis = torch.from_numpy(nodeOptDis.astype(np.float32)).cuda()
            nodeLimit = torch.from_numpy(nodeLimit.astype(np.float32)).cuda()
        else:
            costAll = torch.from_numpy(costAll.astype(np.float32))
            blockInfo = torch.from_numpy(blockInfo.astype(np.float32))
            nodeOptDis = torch.from_numpy(nodeOptDis.astype(np.float32))
            nodeLimit = torch.from_numpy(nodeLimit.astype(np.float32))
            

        final_solution = pso(n_particles=n_particles, n_iterations=n_iterations, fitness=fitness)

        fig = plt.figure(figsize=(6,4), dpi= 300)
        plt.plot(range(n_iterations), alloEpoch[:,0],'-',label='target value')
        # plt.plot(range(n_iterations+1), alloEpoch[:,1],'-',label='communication cost')
        # plt.plot(range(n_iterations+1), alloEpoch[:,2],'-',label='node storage proportion')
        plt.legend()
        print("Final state: ", final_solution)
        # 计算最优状态的能量
        final_energy = fitness(final_solution)
        print("Final energy: ", final_energy)
        end_time = time.time()
        run_time = end_time - start_time
        print("程序运行时间：", run_time, "秒") 
    if saveResult:
        # np.save('./Data/Result/Q2/{}'.format(fileSaveName),final_solution)
        fig.savefig('./IterationChart/Q2/'+fileSaveName+'.svg',dpi=300,format='svg',bbox_inches = 'tight')

