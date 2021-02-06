# FROM_ML_TO_NLP
# 基础知识
* [广义线性模型](https://zhuanlan.zhihu.com/p/22876460)
  
# 机器学习  
## 监督学习  
* 最大熵模型
* 神经网络  
  * 损失函数(loss function)
    * 均方差 
    * [信息熵，交叉熵(cross-entropy)，相对熵(KL散度)](https://charlesliuyx.github.io/2017/09/11/%E4%BB%80%E4%B9%88%E6%98%AF%E4%BF%A1%E6%81%AF%E7%86%B5%E3%80%81%E4%BA%A4%E5%8F%89%E7%86%B5%E5%92%8C%E7%9B%B8%E5%AF%B9%E7%86%B5/)   
      <details>  
      <summary>笔记</summary>  
  
      **信息熵**: 信息的不确定性或信息量的期望值。  
      公式为```sum([p[i] * f[p[i]] for i in range(n)])```。    
      其中```p[i]```为第i件事发生的概率，```f[p[i]]```为```p[i]```中的信息不确定性，一般有```f[p[i]]=log(1/p[i])```，即概率越大，不确定性越低。  
      备注：这里```f[p[i]]```也可以理解为第i件事所需要的存储空间，则```f```可以理解为一整个存储策略。  
      **交叉熵**：在给定的真实分布的情况下，使用现有存储策略所需要消耗的空间大小。在训练模型的过程中，我们总希望信息熵越来越小，也即通过调整参数来最优化我们的存储策略。也可以理解成不断的拟合真实的信息量分布。    
      **相对熵**：用于衡量信息熵和交叉熵之间的差异。训练过程中总希望相对熵趋近0。  
      </details>    
  * 激活函数      
    * [softmax](https://blog.csdn.net/bitcarmanlee/article/details/82320853)  
      <details>  
      <summary>笔记</summary>  
  
      ![20180902220822202](https://user-images.githubusercontent.com/7517810/107108946-0f43df00-680a-11eb-82c2-7396cfc8d2c7.jpg)
      </details>
      
      
