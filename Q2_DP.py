import numpy as np

def dynamic_allocation(block_sizes, node_capacities, communication_costs):
    num_blocks = len(block_sizes)
    num_nodes = len(node_capacities)

    # 初始化动态规划表
    dp = np.full((num_blocks + 1, num_nodes + 1), float('inf'))
    dp[0, :] = 0

    # 存储分配策略
    allocation = np.zeros((num_blocks, num_nodes), dtype=int)

    # 动态规划
    for i in range(1, num_blocks + 1):
        for j in range(1, num_nodes + 1):
            for k in range(num_nodes):
                # 当前节点容量是否足够存放当前区块
                if block_sizes[i - 1] <= node_capacities[k]:
                    # 更新动态规划表和分配策略
                    cost = dp[i - 1, j - 1] + communication_costs[i - 1, k, :].sum()
                    if cost < dp[i, j]:
                        dp[i, j] = cost
                        allocation[i - 1, :] = 0
                        allocation[i - 1, k] = 1

    # 寻找最优解
    min_cost = float('inf')
    best_allocation = None
    for j in range(1, num_nodes + 1):
        if dp[-1, j] < min_cost:
            min_cost = dp[-1, j]
            best_allocation = allocation

    return best_allocation

# 区块大小
block_sizes = np.array([5, 8, 3])
# 节点容量
node_capacities = np.array([10, 8, 10])
# 通信成本
communication_costs = np.random.rand(3, 3, 3)

allocation = dynamic_allocation(block_sizes, node_capacities, communication_costs)
print(allocation)
