# FROM_ML_TO_NLP
# 基础知识
* [如何理解对数(log)](https://www.zhihu.com/question/26097157/answer/121067428)
  <details>  
  <summary>笔记</summary>  
  
  ![1c6378a1bd8dd28da10f7ae82f37cba1_r](https://user-images.githubusercontent.com/7517810/107121417-4cd35700-6860-11eb-8fb2-68b5a0f2bc51.png)  
  对数本质上是指数坐标轴的**系数**(0,1,2,3...)。对数的底数是指数坐标轴的**乘数**(2)，对数的真数是指数坐标轴的**实际数值**(1,2,4,8...)。
  </details>     
* [概率分布和概率密度](https://www.jianshu.com/p/b570b1ba92bb)
  <details>  
  <summary>笔记</summary>  
  
  **(a)概率分布函数（累积概率函数）**   
  用于描述某一段连续值的概率分布，通常用```F(x)```表示,指数值小于等于x的所有概率之和，因此必然是单调递增的（因为概率不能小于0）    
  **(b)概率(质量)函数（离散值），概率密度函数（连续值）**   
  用于描述每个数值的概率，通常用```f(x)```表示，是F(x)的导数。因此f(x)曲线下[a,b]之间的面积就是数值在[a,b]之间的概率值。  
  ![742658-6902c1e6b17050f6](https://user-images.githubusercontent.com/7517810/107124240-a04da100-6870-11eb-987b-23ff0e9167cb.jpg)  
  </details>  
* [伯努利分布/两点分布/0-1分布](https://zh.wikipedia.org/wiki/%E4%BC%AF%E5%8A%AA%E5%88%A9%E5%88%86%E5%B8%83)
  <details>  
  <summary>笔记</summary>  
  
  ![微信截图_20210208210029](https://user-images.githubusercontent.com/7517810/107305338-caf65000-6a50-11eb-9c7e-351582192b1c.png)
  进行一次伯努利实验，成功则值为1，不成功则为0.坐标轴x轴为试验次数，y轴为当试验次数为x时的成功率。  
  </details> 
* [二项分布](https://zh.wikipedia.org/wiki/%E4%BA%8C%E9%A0%85%E5%BC%8F%E5%88%86%E5%B8%83)
  <details>  
  <summary>笔记</summary>  
  
  进行n伯努利实验，如果其中k次都成功了则概率函数如下图所示，如果n=1则等价于伯努利分布  
  ![微信截图_20210206174303](https://user-images.githubusercontent.com/7517810/107131262-c8ee8e80-68a2-11eb-923b-5dad0670d022.png)  
  </details> 
* [正态分布/高斯分布](https://zh.wikipedia.org/wiki/%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83)  
  <details>  
  <summary>笔记</summary>  
    
  ![微信截图_20210206175110](https://user-images.githubusercontent.com/7517810/107131408-ea9c4580-68a3-11eb-8c1e-8622cf0e3bd4.png)
  中心极限定理：大量统计独立的随机变量的平均值的分布趋近于正态分布。
  当二项分布中n无穷大，p=0.5时，可以近似于正态分布。  
  </details> 
* [经验分布](https://zh.wikipedia.org/wiki/%E7%BB%8F%E9%AA%8C%E5%88%86%E5%B8%83%E5%87%BD%E6%95%B0)  
  <details>  
  <summary>笔记</summary>  
    
  就是经验推断整体的一个方式。直接用数据里的样本来构建一个概率分布函数，首先假设每个样本的概率都是相同的，然后将所有样本按照大小顺序叠加起来就成了经验分布的概率分布函数了。 
  </details> 
* [自信息，熵，以及最大似然](https://blog.csdn.net/yujianmin1990/article/details/71213601)
  <details>  
  <summary>笔记</summary>  
  
  **自信息，I(x)**  
  表示不确定性的程度。一个事件确定发生，是没有信息量的；而一个事件发生的概率越小，则其信息量越大。未知所带来的不确定性，就是自信息要描述的目标。  
  ![微信截图_20210206151753](https://user-images.githubusercontent.com/7517810/107128779-84f18e80-688e-11eb-8222-8c1b6b2fbc9a.png)  
  **信息熵，H(x)**  
  自信息的期望值。均匀分布时熵最大。  
  ![微信截图_20210206151842](https://user-images.githubusercontent.com/7517810/107128797-9f2b6c80-688e-11eb-9fb5-b6077f519f09.png)  
  联合熵，H(X,Y)：联合分布的混乱程度。　  
  互信息，I(X,Y)：两个变量相互间相互依赖程度。  
  [条件熵](https://zhuanlan.zhihu.com/p/26551798)，H(X|Y)：联合分布基于某变量的条件下的熵  
  交叉熵，CE(X,Y)：两个分布的不相似程度的描述，越相似，交叉熵越低，越不相似，交叉熵越高。  
  相对熵，DKL(X,Y)：两个分布的不对称程度的距离，也叫KL divergence。  
  关系：交叉熵=信息熵+相对熵，CE(X,Y)=H(X)+DKL(X,Y)  
  </details>  
* 指数分布族(The Exponential Family)和广义线性模型(Generalized Linear Model)[[概述]](https://zhuanlan.zhihu.com/p/22876460)[[证明]](https://xg1990.com/blog/archives/304)
  <details>  
  <summary>笔记</summary>  
  
  **指数分布族**  
  ![20190214085454173](https://user-images.githubusercontent.com/7517810/107122927-c0796200-6868-11eb-983a-cf482ddf2903.png)  
  指数分布族本质上是一些常用概率分布(高斯分布，伯努利分布...)的统一表达形式，用处是找出这些概率分布的共性。  
  T(y)是充分统计量，可以理解成数据无限时的y，也即目标值。  
  η是自然参数，可以理解为各个概率分布的参数的统一表达形式。  
  a(η)是对数配分函数，是归一化因子的对数形式，本质上是用于将概率归一化的。    
  也即T,a,b确定了一种分布，η是该分布的参数。  
  **广义线性模型**  
  使用广义线性模型建模时需要进行3个假设：  
  1. P(y|x;θ),即y的条件概率分布属于指数分布族  
  2. y的估计值就是P(y|x;θ)的期望值  
  3. 自然参数η和x是线性关系  
  
  广义线性模型+以上假设+伯努利分布=逻辑回归  
  广义线性模型+以上假设+高斯分布=线性回归  
  广义线性模型+以上假设+多项式分布=softmax  
  </details> 
* [拉格朗日乘子(数)法](https://www.zhihu.com/search?type=content&q=%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E4%B9%98%E5%AD%90%E6%B3%95)
  <details>  
  <summary>笔记</summary>  
  
  拉格朗日乘子法是一种寻找多元函数在其变量受到一个或多个条件的约束时的极值的方法。
  简单来说就是把条件函数乘以一个新变量lambda，然后代入原函数里后再求导找极值就完事了。
  ![微信截图_20210209200535](https://user-images.githubusercontent.com/7517810/107449786-3522f980-6b12-11eb-8614-647a1ababf93.png)
  </details> 
# 机器学习  
## 监督学习  
* [最大熵模型](https://zhuanlan.zhihu.com/p/78504132)
  <details>  
  <summary>笔记</summary>  
  
  最大熵模型认为，在所有可能的概率模型中，熵最大的模型是最好的模型。  
  本质上就是通过样本的经验分布以及最大熵的假设来寻找符合要求的分布。  
  其中事先假设样本经验分布的期望值与实际分布的期望值相同，然后使用拉格朗日乘子法求出在概率值和为1以及样本无偏假设的情况下的熵的最大值。
  </details>
* [LDA模型](https://zhuanlan.zhihu.com/p/31470216)（没搞懂，不过好像暂时没用）
  <details>  
  <summary>笔记</summary>  
  
  它可以将文档集中每篇文档的主题以概率分布的形式给出，从而通过分析一些文档抽取出它们的主题分布后，便可以根据主题分布进行主题聚类或文本分类。属于词袋模型。
  </details>

* [逻辑回归-(Logistic Regression)](https://charlesliuyx.github.io/2017/09/04/LogisticRegression%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/)  
  [为什么逻辑回归要用sigmoid?](https://www.zhihu.com/question/35322351/answer/67193153)  
  <details>  
  <summary>笔记</summary>  
  
  本质上来说，逻辑回归是假设f(y|x)符合指数分布族的分布规律，从而用广义线性模型推导出来的。
  也即，逻辑回归假设每个f(y|x)都是一个伯努利分布，而且x和每个对应的伯努利分布的自然参数呈线性关系。
  指数分布族中的自然参数是y前的系数。
  ![微信截图_20210208210008](https://user-images.githubusercontent.com/7517810/107305309-bd40ca80-6a50-11eb-80ce-e3aa53a0555b.png)
  </details>
* 支持向量机
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
      **交叉熵**：在给定的真实分布的情况下，使用现有存储策略所需要消耗的空间大小。或者理解成：把真实值代入当前预测分布后不确定部分的大小。在训练模型的过程中，我们总希望信息熵越来越小，也即通过调整参数来最优化我们的存储策略。也可以理解成不断的拟合真实的信息量分布。    
      **相对熵**：用于衡量信息熵和交叉熵之间的差异。训练过程中总希望相对熵趋近0。  
      </details>    
      [为什么在分类任务中使用交叉熵？](https://zhuanlan.zhihu.com/p/74075915)
      <details>  
      <summary>笔记</summary>  
  
      首先给出公式：KL散度(相对熵)=信息熵-交叉熵。KL散度描述的是两个分布之间的差异，而由于对于同一数据，信息熵固定，因此使用交叉熵一样可以计算两个分布间的差值。
      </details>  
  * 激活函数      
    * [softmax](https://blog.csdn.net/bitcarmanlee/article/details/82320853)  
      <details>  
      <summary>笔记</summary>  
  
      ![20180902220822202](https://user-images.githubusercontent.com/7517810/107108946-0f43df00-680a-11eb-82c2-7396cfc8d2c7.jpg)
      </details>
      
      
