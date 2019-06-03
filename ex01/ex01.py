
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path =  'ex01\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()


#%%
data.describe()

#%% [markdown]
# 看下数据长什么样子

#%%
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()

#%% [markdown]
# 现在让我们使用梯度下降来实现线性回归，以最小化成本函数。 
#%% [markdown]
# <font color=red size=3> #############请在下面写出线性回归的代价函数（备注：函数名字为computeCost(X, y, theta)）#################### </font>
def computeCost(X, y, theta):
    # print(X)
    # print(theta)
    temp = np.dot(X, theta.T) - y
    # 把temp每个结果都平方生成一个数组后求和，除以2m
    result = [[temp[i][j]**2 for j in range(len(temp[i]))] for i in range(len(temp))]
    return np.sum(result)/(2*len(temp))

#%% [markdown]
# 让我们在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度。

#%%
data.insert(0, 'Ones', 1)
# 在第0列 'population'前加入一列名为'Ones' 值为全1的列

#%% [markdown]
# 现在我们来做一些变量初始化。
# 将源数据表格拆分为输入x和真实值y
cols = data.shape[1]
#X是所有行，去掉最后一列
X = data.iloc[:,0:cols-1]
#X是所有行，最后一列
y = data.iloc[:,cols-1:cols]

#%% [markdown]
# 观察下 X (训练集) and y (目标变量)是否正确.

#%%
X.head()#head()是观察前5行


#%%
y.head()

#%% [markdown]
# 代价函数是应该是numpy矩阵，所以我们需要转换X和Y，然后才能使用它们。 我们还需要初始化theta。

#%%
#x为两列的矩阵，y为一列
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

#%% [markdown]
# theta 是一个(1,2)矩阵
#%%
X
#%%
y
#%%
theta

#%% [markdown]
# 看下维度

#%%
X.shape, theta.shape, y.shape

#%% [markdown]
# 计算代价函数 (theta初始值为0).

#%%

# 初始化 theta为(0,0)
computeCost(X, y, theta)

#%% [markdown]
# # batch gradient decent（批量梯度下降）
# $${{\theta }_{j}}:={{\theta }_{j}}-\alpha \frac{\partial }{\partial {{\theta }_{j}}}J\left( \theta  \right)$$
#%% [markdown]
# <font color=red size=3> #############请在下面写出线性回归的批量梯度下降代码（备注：函数名字为gradientDescent(X, y, theta, alpha, iters) 要求返回模型参数theta 以及 每次迭代过程中的代价函数 cost）#################### </font>

#%%

def gradientDescent(X, y, theta, alpha, iters):
    # temp = [[0. 0.]]
    # 用于存放临时的theta
    temp = np.matrix(np.zeros(theta.shape))
    #.ravel():将多维数组转化为一维数组
    # int(theta.ravel().shape[1]) = 2
    parameters = int(theta.ravel().shape[1])
    # np.zeros -> matrix
    # cost -> (1,100) matrix
    cost = np.zeros(iters)
    
    for i in range(iters):
        # theta0 公式
        part_1 = np.dot(X, theta.T) - y
        
        for j in range(parameters):
            # theta1 公式
            part_2 = np.multiply(part_1, X[:, j])
            temp[0, j] -= alpha * np.sum(part_2)/len(X)
            print(j)
            print(temp[0, j])
        theta = temp
        print(theta)
        # 用每一次迭代计算出来的theta分别计算对应的cost用于画图
        cost[i] = computeCost(X, y, theta)
    return theta, cost

#%% [markdown]
# 初始化一些附加变量 - 学习速率α和要执行的迭代次数。
# 注意：在此处更改学习率，看看对模型的训练有什么影响

#%%
alpha = 0.001
iters = 1000

#%% [markdown]
# 现在让我们运行梯度下降算法来将我们的参数θ适合于训练集。

#%%
g, cost = gradientDescent(X, y, theta, alpha, iters)

#%% [markdown]
# 最后，我们可以使用我们拟合的参数计算训练模型的代价函数（误差）。

#%%
computeCost(X, y, g)



#%%

# 改变学习速率生成比较图的函数 draw_compare(X,y,theta,alpha1,alpha2,iters)
alpha1 = 0.001
alpha2 = 0.01
iters = 1000

# ====================================
# ====================================
# 函数重构部分：
def draw_prepare(X,y,theta,alpha,iters):
    g, cost = gradientDescent(X, y, theta, alpha, iters)
    computeCost(X, y, g)
    # 设置x取值
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    # 设置y取值 ： J(theta) 公式
    f = g[0, 0] + (g[0, 1] * x)
    return g,cost,x,f
    

def draw_predicted(x1,f1,alpha1,x2,f2,alpha2):
    fig, ax = plt.subplots(figsize = (12,8))
    ax.plot(x1, f1, 'r', label='Prediction while alpha = '+ str(alpha1))
    ax.plot(x2, f2, 'purple', label='Prediction while alpha = '+ str(alpha2))
    ax.scatter(data.Population, data.Profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.show()

def draw_cost(cost1,cost2,iters):
    fig, ax = plt.subplots(figsize = (12,8))
    #横轴：迭代次数 ；纵轴： 误差
    ax.plot(np.arange(iters), cost1, 'r',label = 'alpha1 cost')
    ax.plot(np.arange(iters), cost2, 'blue',label = 'alpha2 cost')
    ax.legend(loc=1)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()

def draw_compare(X,y,theta,alpha1,alpha2,iters):
    g1,cost1,x1,f1 = draw_prepare(X,y,theta,alpha1,iters)
    g2,cost2,x2,f2 = draw_prepare(X,y,theta,alpha2,iters)
    draw_predicted(x1,f1,alpha1,x2,f2,alpha2)
    draw_cost(cost1,cost2,iters)
# ====================================
# ====================================
# ====================================

draw_compare(X,y,theta,alpha1,alpha2,iters)


