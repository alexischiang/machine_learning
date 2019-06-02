#%% [markdown]
# # 机器学习练习 2 - 逻辑回归
#%% [markdown]
# ## 逻辑回归
#%% [markdown]
# 在训练的初始阶段，我们将要构建一个逻辑回归模型来预测，某个学生是否被大学录取。设想你是大学相关部分的管理者，想通过申请学生两次测试的评分，来决定他们是否被录取。现在你拥有之前申请学生的可以用于训练逻辑回归的训练样本集。对于每一个训练样本，你有他们两次测试的评分和最后是被录取的结果。为了完成这个预测任务，我们准备构建一个可以基于两次测试评分来评估录取可能性的分类模型。

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.matlib


#%%
path = 'ex02\ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.head()

#%% [markdown]
# 让我们创建两个分数的散点图，并使用颜色编码来可视化，如果样本是正的（被接纳）或负的（未被接纳）。

#%%
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

#%% [markdown]
# 看起来在两类间，有一个清晰的决策边界。现在我们需要实现逻辑回归，那样就可以训练一个模型来预测结果。
#%% [markdown]
# # sigmoid 函数
# g 代表一个常用的逻辑函数（logistic function）为S形函数（Sigmoid function），公式为： \\[g\left( z \right)=\frac{1}{1+{{e}^{-z}}}\\] 
# 合起来，我们得到逻辑回归模型的假设函数： 
# 	\\[{{h}_{\theta }}\left( x \right)=\frac{1}{1+{{e}^{-{{\theta }^{T}}X}}}\\] 
#%% [markdown]
# <font color=red size=3> #############请在下面写出sigmoid函数sigmoid(z)#################### </font>

#%%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#%% [markdown]
# 让我们做一个快速的检查，来确保它可以工作。

#%%
nums = np.arange(-10, 10, step=1)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(nums, sigmoid(nums), 'r')
plt.show()

#%% [markdown]
# 棒极了！现在，我们需要编写代价函数来评估结果。
# 代价函数：
# $J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)]}$
#%% [markdown]
# 现在，我们要做一些设置，和我们在练习1在线性回归的练习很相似。

#%%
# add a ones column - this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

#%% [markdown]
# 让我们来检查矩阵的维度来确保一切良好。

#%%
theta
#%%
X
#%%
y

#%%
X.shape, theta.shape, y.shape

#%% [markdown]
# <font color=red size=3> #############请在下面写出逻辑斯蒂回归的代价函数cost(theta, X, y)，函数返回计算的代价函数值#################### </font>
#%%
# test 全1矩阵
# all_ones = np.matlib.ones((100,1))
# # print(all_ones)
# print(all_ones.shape)
# print('------')
# # print(all_ones.T)
# print(all_ones.T.shape)





#%%
def cost(theta, X, y):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

#%% [markdown]
# 让我们计算初始化参数的代价函数(theta为0)。

#%%
cost(theta, X, y)

#%% [markdown]
# 看起来不错，接下来，我们需要一个函数来计算我们的训练数据、标签和一些参数thata的梯度。
#%% [markdown]
# # gradient descent(梯度下降)
# * 这是批量梯度下降（batch gradient descent）  
# * 转化为向量化计算： $\frac{1}{m} X^T( Sigmoid(X\theta) - y )$
# $$\frac{\partial J\left( \theta  \right)}{\partial {{\theta }_{j}}}=\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}})x_{_{j}}^{(i)}}$$
#%% [markdown]
# <font color=red size=3> #############请在下面写出逻辑斯蒂回归的梯度函数gradient(theta, X, y)，函数返回计算的梯度,注意这里不是写出梯度下降算法，而是要求写出求梯度的函数，目的是在下面使用scipy.optimize优化的时候使用#################### </font>

#%%
def gradient(theta, X, y):
    
    
    
    
    
    
    
    

#%% [markdown]
# 注意，我们实际上没有在这个函数中执行梯度下降，我们仅仅在计算一个梯度步长。我们可以用SciPy的“optimize”命名空间来做同样的事情。
#%% [markdown]
# 我们看看用我们的数据和初始参数为0的梯度下降法的结果。

#%%
gradient(theta, X, y)

#%% [markdown]
# 现在可以用SciPy's truncated newton（TNC）实现寻找最优参数。
#%% [markdown]
# 调用：
# 
# scipy.optimize.fmin_tnc(func, x0, fprime=None, args=(), approx_grad=0, bounds=None, epsilon=1e-08, scale=None, offset=None, messages=15, maxCGit=-1, maxfun=None, eta=-1, stepmx=0, accuracy=0, fmin=0, ftol=-1, xtol=-1, pgtol=-1, rescale=-1, disp=None, callback=None)
# 
# 最常使用的参数：
# 
# func：优化的目标函数
# 
# x0：初值
# 
# fprime：提供优化函数func的梯度函数，不然优化函数func必须返回函数值和梯度，或者设置approx_grad=True
# 
# approx_grad :如果设置为True，会给出近似梯度
# 
# args：元组，是传递给优化函数的参数

#%%
import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
result

#%% [markdown]
# 让我们看看在这个结论下代价函数计算结果是什么个样子~

#%%
cost(result[0], X, y)

#%% [markdown]
# 接下来，我们需要编写一个函数，用我们所学的参数theta来为数据集X输出预测。然后，我们可以使用这个函数来给我们的分类器的训练精度打分。
# 逻辑回归模型的假设函数： 
# 	\\[{{h}_{\theta }}\left( x \right)=\frac{1}{1+{{e}^{-{{\theta }^{T}}X}}}\\] 
# 当${{h}_{\theta }}$大于等于0.5时，预测 y=1
# 
# 当${{h}_{\theta }}$小于0.5时，预测 y=0 。

#%%
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


#%%
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))

#%% [markdown]
# 我们的逻辑回归分类器预测正确，如果一个学生被录取或没有录取，达到89%的精确度。不坏！记住，这是训练集的准确性。我们没有保持住了设置或使用交叉验证得到的真实逼近，所以这个数字有可能高于其真实值（这个话题将在以后说明）。

#%%



