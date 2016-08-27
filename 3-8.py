# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 20:04:28 2016

@author: zhangweijian
"""

import numpy as np
import numpy
import matplotlib.pyplot as plt
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体    
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题  

class Bayes(object):
    
    # 初始化定义一些常量，根据书上的内容，为简化模型，认为先验分布是各向同性的高斯分布
    def __init__(self):
        self.alpha = 0.5       # 先验分布的协方差矩阵是alpha**(-1)*I，alpha是精度
        self.beta = 25.0         # 似然函数的精度
        self.features = 20          # 设计矩阵所提取的特征总数
        self.low,self.high = -1,1   # 函数的定义域
        self.M = np.matrix(np.array([0]*self.features).reshape(-1,1))  #根据已有参数设定先验分布的均值
        self.S = np.matrix(self.alpha**(-1)*np.identity(self.features))  #                     和协方差
        self.samples = []          # 每一次估计所用到的样本点和
        self.targets = []          #                  样本值
    
    # 定义一个根据均值和协方差矩阵生成相应概率密度函数的函数
    def gaussian(self,loc,scale):
        def prob_g(x):
            D = loc.shape[0]
            scale_det = np.linalg.det(scale)
            return 1/((2*np.pi)**(D/2)*scale_det**0.5)*\
                    np.exp(-0.5*(x-loc).T*scale.I*(x-loc))
        return prob_g
    
    # 计算先验分布
    def prior(self):
        return self.gaussian(self.M,self.S)
    
    # 生成模拟数据的原函数
    def source_function(self,x):
        return np.sin(2*np.pi*x)
        
    # 产生样本数据，并混入噪音
    def sampling(self,nums):
        samples = np.random.rand(nums)*(self.high-self.low)+self.low
        targets = self.source_function(samples) + np.random.randn(nums)/np.sqrt(self.beta)
        targets = np.matrix(targets.reshape(-1,1))
        return samples,targets
        
    # 高斯核函数
    def gaussian_kernel(self,center,s=0.1):
        def parkage(x):
            return np.exp(-((x-center)**2)/(2*s**2))
        return parkage
        
    # 基函数列生成函数
    def kernels(self):
        k_points = np.linspace(self.low,self.high,self.features)
        kernel_functions = []
        for center in k_points:
            kernel_functions.append(self.gaussian_kernel(center))
        return kernel_functions
        
    # 生成设计矩阵Q
    def degin(self,samples):
        kernels = self.kernels()
        if not isinstance(samples,(list,numpy.ndarray)):
            Q = [[kernel(samples) for kernel in kernels]]
        else:
            Q = [[kernel(sample) for kernel in kernels] for sample in samples]
        Q = np.matrix(Q)
        return Q
    
    # 似然函数
    def likelihood(self,x,y):
        def parkage(W):
            loc = x*W
            scale = np.matrix(1.0/self.beta)
            return self.gaussian(loc,scale)(y)
        return parkage
        
    # 观测数据后由先验分布生成后验分布
    def estimate(self,nums):
        samples,targets = self.sampling(nums)
        self.samples += list(samples)
        for item in targets.flatten().tolist()[0]:
            self.targets.append(item)
        Q = self.degin(samples)
        S_I = self.S.I + self.beta*Q.T*Q
        self.M = S_I.I*(self.S.I*self.M + self.beta*Q.T*targets)
        self.S = S_I.I
        
    # 绘制先验分布，只能绘制两个特征的分布
    def plot_prior(self):
        if self.features != 2:
            print u'特征数量过多，无法可视化！'
            return
        prior_pdf = self.prior()
        x_0,x_1 = np.mgrid[-1:1:100j,-1:1:100j]
        z = []
        for point in zip(x_0.flatten(),x_1.flatten()):
            point = np.matrix(np.array(point).reshape(-1,1))
            z.append(prior_pdf(point))
        z = np.array(z).reshape(x_0.shape)
        plt.contourf(x_0,x_1,z)
        
    # 绘制似然函数的分布,只能绘制两个特征的分布
    def plot_likelihood(self):
        if self.features != 2:
            print u'特征数量过多，无法可视化！'
            return
        sample,target = self.sampling(1)
        sample = self.degin(sample)
        likelihood_pdf = self.likelihood(sample,target)
        x_0,x_1 = np.mgrid[-1:1:100j,-1:1:100j]
        z = []
        for point in zip(x_0.flatten(),x_1.flatten()):
            point = np.matrix(np.array(point).reshape(-1,1))
            z.append(likelihood_pdf(point))
        z = np.array(z).reshape(x_0.shape)
        plt.contourf(x_0,x_1,z)
    
    # 预测分布
    def predict(self,x):
        Q = self.degin(x)
        scale = 1/self.beta + Q*self.S*Q.T
        loc = Q*self.M
        return loc,scale
    
    # 绘制预测分布图像
    def plot_predict(self):
        data = []
        l = np.linspace(self.low,self.high,200)
        for x in l:
            loc,scale = self.predict(x)
            data.append([x,loc,scale])
        data = np.array(data)
        plt.plot(data[:,0],data[:,1],c='r',label=u'预测曲线')
        plt.fill_between(data[:,0],data[:,1]+data[:,2],data[:,1]-data[:,2],alpha=0.35,label=u'预测曲线的方差')
        plt.scatter(self.samples,self.targets,label=u'样本点')
        plt.scatter(self.samples,[-2]*len(self.samples),marker='+',c='k',alpha=0.8,s=30)
        plt.plot(l,self.source_function(l),c='k',alpha=0.7,lw=2,label=u'真实原函数')
        plt.ylim(-4,6)
        plt.legend(loc='best')
        plt.title(u'基于%d个样本点的估计'%len(self.samples))
        
if __name__=='__main__':
    b = Bayes()
    plt.figure()
    plt.subplot(221)
    b.estimate(1)
    b.plot_predict()
    plt.subplot(222)
    b.estimate(1)
    b.plot_predict()
    plt.subplot(223)
    b.estimate(2)
    b.plot_predict()
    plt.subplot(224)
    b.estimate(30)
    b.plot_predict()
	plt.show()
	
    
    