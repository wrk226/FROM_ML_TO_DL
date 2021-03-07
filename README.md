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
  ![微信截图_20210211224424](https://user-images.githubusercontent.com/7517810/107728025-e1074900-6cba-11eb-8899-07e1781c52b7.png)  
  互信息，I(X,Y)：两个变量相互间相互依赖程度。  
  ![微信截图_20210211224433](https://user-images.githubusercontent.com/7517810/107728026-e1074900-6cba-11eb-8a8f-bb11947894e3.png)  
  [条件熵](https://zhuanlan.zhihu.com/p/26551798)，H(X|Y)：联合分布基于某变量的条件下的熵  
  ![微信截图_20210211224439](https://user-images.githubusercontent.com/7517810/107728027-e1074900-6cba-11eb-8e11-1674e88032f9.png)  
  交叉熵，CE(X,Y)：两个分布的不相似程度的描述，越相似，交叉熵越低，越不相似，交叉熵越高。 
  ![微信截图_20210211224142](https://user-images.githubusercontent.com/7517810/107727844-532b5e00-6cba-11eb-8b28-6da838495781.png)  
  相对熵，DKL(X,Y)：两个分布的不对称程度的距离，也叫KL divergence。  
  ![微信截图_20210211224449](https://user-images.githubusercontent.com/7517810/107728028-e19fdf80-6cba-11eb-97c6-ba8d25b65545.png)    
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
* [Cosine similarity](https://zhuanlan.zhihu.com/p/78504132)
  <details>  
  <summary>笔记</summary>   
  
  可以用于测量两个词嵌入的相似度。向量相乘除以两个向量长度之积，也就是说两个向量完全一致的时候值为1，不一致的时候小于1
  ![微信截图_20210215160220](https://user-images.githubusercontent.com/7517810/107992687-36877280-6fa7-11eb-8e58-eafc88d08fc2.png)  
  </details>
* [l1 l2 regularization]()
  <details>  
  <summary>笔记</summary>   
  
  </details>
# 机器学习  
## 监督学习  
* [EM算法/期望极大算法(Expectation Maximization algorithm)](https://zhuanlan.zhihu.com/p/78311644) 
  <details>
  <summary>什么是似然函数？（Likelihood function）</summary>
     假设我们现在有一个硬币，随机投掷一次硬币出现正面的概率为p。

     现在我们连续投掷了两次硬币，结果硬币都是正面。       
     似然函数就是：p=0.1, 0.2, 0.3...的概率。         
     也即L=p^2(p代表正面朝上，p^2就是两次都是正面朝上)         
     简而言之，似然性，是用于在已知某些观测所得到的结果时，对有关事物之性质的参数进行估值。
  </details>
  <details>
  <summary>什么是MLE/最大似然估计？(maximum likelihood estimation)</summary>
     还是上面的例子，随机投掷一次硬币出现正面的概率为p，现在连续抛两次硬币都是正面，那么当p取什么值的时候似然性最大呢？

     显而易见，p=1的概率最大，也即当p=1时似然值最大。         
     而这个p=1就是我们的最大似然估计。         
     一般来说计算MLE的时候是先估计变量的分布（伯努利分布，指数分布，高斯分布...）每个分布里都会有自带的系数。         
     比如投硬币就符合伯努利分布，里面的系数就是之前提到的p。         
     有了变量分布公式后可以由此建立最大似然函数。然后找似然函数的最大值就完事了，一般可能涉及到求导，取log值之类的数学操作。
  </details>
        </details>
  <details>
  <summary>什么是EM算法？</summary>
     EM算法实质上就是当似然函数难以找出最大值的情况下采取的迭代计算方式。

     一般来说似然函数难以求导的原因是因为里面包含隐藏变量。         
     举个例子，投掷硬币，现在有硬币A,B,C,每次投掷的时候我都会先抛一次A（A的结果不作记录），如果A是正面的话就用B投掷，如果A是反面的话就用C投掷。用B或C的结果作为这一次抛掷的结果。这时候A的正反面概率就是隐藏变量，因为我们无法直接观测到A是正面还是反面。

     EM算法步骤如下：

     1. 给要求的参数基于一个随机的初始估计值
     2. 找到另一个能使似然函数变大的参数
     3. 不断迭代直到收敛

     显而易见，这里最重要的就是第二步，如何找到一个新的更好的参数。一般方式就是直接将初始值或者上一次迭代的值代入概率分布，然后计算出期望函数，最后求出期望函数的极大值和对应的新的参数。
  </details>

* [HMM/隐马尔可夫模型(Hidden Markov model)](https://zh.wikipedia.org/wiki/%E9%9A%90%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E6%A8%A1%E5%9E%8B) 
  <details>
  <summary>什么是HMM？</summary>
  HMM模型是用于描述一个随机序列的模型。  

  这个随机序列中每一时刻（天）都有一个状态/隐藏变量/hidden variable（心情）和一个观测值（我的行为）。  
  HMM假设：1. 观测值（我的行为）仅仅取决于当前时刻的状态（今天心情）。2. 当前时刻的状态（今天心情）仅仅取决于前一时刻的状态（昨天心情）。  
  而HMM能解决的问题一般都是当一个随机过程中的某些值缺失时用于求解缺失值的方法。        
  求解HMM过程时我们会使用到的条件一般是：初始概率分布（第一天各种心情的概率），状态转移概率分布（前一天的心情对第二天心情的影响），观测概率分布（特定心情下我会做各种事的概率）  
  hmm经典三大问题：  
  1. 已知我这个月每天的行为，求解我下个月第一天会做什么(一般用前向算法，也即一天天往后推，一直推到下个月一号)
  2. 已知我这个月每天的行为，求解我这个月每天为各种心情的概率（前向后向算法都可以）
  3. 已知我这个月每天的行为，求解我这个月最可能的心路历程（一般用[维特比(viterbi algorithm)算法](https://www.zhihu.com/question/20136144/answer/763021768)，即不断删掉次优路径，一个动态规划算法）

  求解过程基本就是简单的概率运算。  
  </details>

* [CRF/条件随机场(Conditional Random Field)](https://www.cnblogs.com/kerwins-AC/p/9584862.html) 

  [CRF的直观理解](https://www.zhihu.com/question/35866596/answer/412520896) 

  <details>
  <summary>什么是CRF？</summary>
  一般在NLP中聊到的都是线性链条件随机场（linear chain CRF）

  CRF和HMM非常类似，只不过HMM的概率模型是有方向的，而CRF的概率模型是无方向的。     
  HMM中t时刻的状态仅仅与t-1时刻的状态有关，而CRF中t时刻的状态与t-1和t+1都有关（因为无方向嘛）。  
  HMM中想要求得t时刻的状态需要用t-1时刻的状态乘以状态转移矩阵，得到每个状态的概率值。然后再通过观测/发射(emission)概率矩阵来得到每个观测值的可能性。  
  而CRF中是直接使用特征函数进行打分，符合一个特征就+1分，不符合就为0。  
  这里的特征有两类，一类是t时刻与t-1,t+1时刻之间的关系特征。例1:如果昨天我心情不好，今天心情一定不会很好。例2:如果我明天开心，那么今天心情一定不会很差。  
  另一类是t时刻自己的特征。例:如果今天我心情不好就肯定不会出门购物  
  这里的特征都是非黑即白的，而且特征数量是不固定的。像HMM中，每个行动与心情都有一个对应的状态转移概率，但是CRF中就不是这样，可以一对多也可以多对一。  
  ![1475786-20180904172015500-1950356185](https://user-images.githubusercontent.com/7517810/108415979-26b49c00-71fc-11eb-94d9-3919452cbde2.png)
  狮子中，t为转移特征，s为状态特征，前面的是他们的系数。

  </details>

* [结构化感知机与普通感知机的区别](https://www.zhihu.com/question/51872633) 

* 朴素贝叶斯(naive bayes)
  <details>
  <summary>什么是朴素贝叶斯？</summary>

  贝叶斯基本公式就是P(Y|X)=P(X|Y)\*P(Y)/P(X)  
  朴素贝叶斯假设了所有特征，也即X是互相独立的。  
  Y代表类别，X代表一系列特征。  
  举个例子，抽卡：Y代表出货或者不出货，X里包含了一系列特征，即X=(x1,x2)，其中x1代表你是不是肝帝，x2代表你是不是土豪。  
  假设某游戏肝帝占比5%=P(x1)，土豪占比2%=P(x2)，抽卡出率10%=P(Y)，出货的人中土豪肝帝占比50%=P(X|Y)，现在我们想知道土豪肝帝的出率是多少，也就是求P(Y|X)=P(出货|土豪，肝帝)  
  因为x1和x2互相独立，则P(X)=(1-P(x1))\*P(x2)+P(x1)=0.95\*0.02+0.05=0.019+0.05=0.069=6.9%  
  所以代入公式,P(Y|X)=P(出货|土豪，肝帝)=50%\*10%/6.9%=0.05/0.069=72.46%>50%,因此朴素贝叶斯判定土豪肝帝大概率会出货。  

  </details>

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
      公式为```sum([p[i] * f[q[i]] for i in range(n)])```。    
      其中```p[i]```为第i件事发生的概率，```q[i]```为预测的概率(或理解成储存因子)，```f[q[i]]```为```q[i]```中的信息不确定性，一般有```f[q[i]]=log(1/q[i])```，即概率越大，不确定性越低。  
      备注：这里```f[q[i]]```也可以理解为第i件事所需要的存储空间，则```f```可以理解为一整个存储策略。  
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
      
      softmax本质上也就是个激活函数，所谓激活函数也就是为了给网络引入非线性变化而已。  
      而softmax的作用有两个，一个是将分数转变为概率，另一个是让分数大的概率不要太大，分数小的概率不要太小。  
      ![20180902220822202](https://user-images.githubusercontent.com/7517810/107108946-0f43df00-680a-11eb-82c2-7396cfc8d2c7.jpg)
      </details>
  * 网络模块      
    * [RNN](https://www.bilibili.com/video/BV1gb411j7Bs?p=146)  
      <details>  
      <summary>笔记</summary>  
      
      简而言之就是将序列分成一个个时间点，然后按照时间顺序挨个进入网络。  
      每一个时间点网络接受的输入=前一个时间点的隐藏层输出+这个一时间点的数据输入，加上一个线性函数，然后经过一个激活函数(tanh或relu)。  
      每一个时间点的输出=这一个时间点的隐藏层输入，加上一个线型函数，然后经过最后的激活函数(softmax或sigmoid)。  
      需要注意的是，所有时间点是共用同一组隐层权重和输出权重的，不同时间点也就只是在这组权重上不断迭代而已。  
      正向传播  
      ![微信截图_20210211222558](https://user-images.githubusercontent.com/7517810/107726901-2413ed00-6cb8-11eb-9b91-b1488085f666.png)  
      反向传播  
      得到每个时间点输出的loss function，然后求和，将之一步步倒推回去。        
      </details>
    * [Bidirectional RNN](https://www.bilibili.com/video/BV1gb411j7Bs?p=154)
      <details>  
      <summary>笔记</summary>  
      
      单向RNN只能获得时间点之前的信息，因此有了双向RNN模型，为了就是能获取双向的信息。  
      分为两部分，正向的和普通RNN一样，从0到t。  
      反向的就是反过来，从t到0。然后在输出的时候结合正向和反向的两个输出一起进行预测。  
      </details>
    * [GRU-gate recurrent unit](https://www.bilibili.com/video/BV1gb411j7Bs?p=152&spm_id_from=pageDriver)  
      <details>  
      <summary>笔记</summary>  
  
      GRU实质上是为了处理梯度消失(gradient vanishing)的问题的，GRU中有两个门:更新门(update gate)和重置门(reset gat)。    
      什么是门呢？可以想象门就是一个二极管，电流足够他就亮，电流不够他就黑着。    
      对应到网络中，电流就是 前一时间节点的隐层状态 和 这一时间节点的输入。然后在外面的激活函数sigmoid就是开关，用来将输入二值化(要么是1，要么是0)。    
      因此两个门就是两个不同的线性方程+sigmoid激活函数的输出，结果就是1或者0。而且所有的门都是这种形式，没有本质的区别。（就像厕所门，卧室门，没有本质的区别，只是一个指示符而已）    
      重置门就是用于决定前一时间节点的隐层状态要不要保留的一个系数。  
      剩下的和RNN就很相似了，新的隐藏状态就等于经过重置门的前一时间节点的隐藏状态+这一时间节点的输入，然后外面套一个tanh。  
      最后再用更新门决定到底是要用用新的隐藏状态作为结果还是用前一时间节点的隐藏状态作为输出。  
      ![微信截图_20210214210126](https://user-images.githubusercontent.com/7517810/107897758-f8874180-6f07-11eb-901c-3462b7247805.png)
      </details>
    * [LSTM unit-long short term memory unit](https://www.bilibili.com/video/BV1gb411j7Bs?p=153)  
      <details>  
      <summary>笔记</summary>  
      
      lstm和gru表面上的区别是多了一个输出，gru是只有隐藏状态(hidden state)，而lstm多了个单元状态(cell state)，单元状态只是用于记忆信息的，不参与输出。    
      lstm和GRU的区别在于lstm取消了GRU中的重置门，然后增加了一个遗忘门和输出门。    
      lstm流程就是迁移时间点的隐藏状态+这一时间点的输入后加sigmoid变成各种门，或者加tanh变成暂时的单元状态。    
      前一时刻的单元状态*遗忘门+暂时的单元状态*更新门=新的单元状态  
      隐藏状态/输出状态=新的单元状态*输出门  
      ![微信截图_20210214210227](https://user-images.githubusercontent.com/7517810/107897780-0341d680-6f08-11eb-85a7-3732731d1de8.png)
      </details>    
    * [注意力机制-Attention mechanism](https://www.zhihu.com/question/68482809/answer/264632289)  
      <details>  
      <summary>笔记</summary>  
  
      假设我们的任务是翻译句子，从中文翻译成英文。则中文句子中每个词都可以想象成一个<key,value>的pair。key代表词语的地址，value代表词语的向量表示。英文中与中文词语对应的词便称之为query。  
      整个过程便是：  
      1. 比较key和query的相似度，得到s  
      2. 使用softmax归一化，将s转化为权重a(其实就是概率)，这里可以保证所有s之和为1(因为softmax输出的是概率嘛。。。)  
      3. 使用权重乘以value得到最终预测时的输入值  
      值得一提的是，有时候key和value可以使用同一个值，也就是都用rnn生成的output。  
      另外，步骤1中计算相似度可以用的方法有很多，比如：点乘，cosine相似度，mlp网络(一层全连接网络)  
      ![v2-07c4c02a9bdecb23d9664992f142eaa5_1440w](https://user-images.githubusercontent.com/7517810/108157513-ec8ab380-70b0-11eb-8420-5818c52dd884.png)    
      **k,q,v是什么**  
      q，k，v分别是query，key，value。  
      对于encoder self-attention，第一次计算的初始值是每个字的embedding，q用来和k做点乘计算相似度，这些相似度经过softmax变成权重，然后权重和v相乘，其实就是v的一个加权平均。  
      如果是encoder-decoder attention，q是decoder的hidden state，k和v是encoder各个位置的hidden state。
      </details>
    * [自注意力机制self attention](https://www.zhihu.com/question/68482809/answer/264632289)  
      <details>  
      <summary>笔记</summary>  
      
      顾名思义，self attention就是自己对自己的注意力机制。也即source和target都是同一组词，其余部分和注意力机制一样。  
      自注意力的用处主要在于寻找句子内单词的联系，可以找到句法特征或者语义特征。
      </details>
    * [Transformer](https://github.com/NLP-LOVE/ML-NLP/tree/master/NLP/16.7%20Transformer)  
      <details>  
      <summary>笔记</summary>  
      
      假设这里我们的任务还是从中文翻译成英文。  
      positional encoder：在普通的seq2seq模型中我们会使用embedding来给每个输入字符在embedding space中找定位置，从而让相似的词拥有相似的向量。但是同一个词在句子的不同位置也往往有不同的意思，因此这里就引入了postitional coding，基本就是通过公式计算出来字符在sentence中位置的表达向量。然后将positional encoding和embedding相加来构成一个新的包含位置信息的embedding。  
      然后就是self attention，这个是为了获得当前字符于其他字符的关联性。也即每个中文字符与其他中文字符的关联性。这里会出现的问题是每个word可能和自己的相关性太高，导致attention weight价值低，所以这里就引入了multihead attention，意思是同时对word做多个attention,然后用这些attention做加权平均，来得到最终的attention weight.
      feed forwared:这个就是个线性模型，目的就是把attention的输出调整成适合下一层的输入。  
      decoder：  
      decoder中我们会先输入英语的embedding，然后也加上position information。  
      然后和encoder中一样，使用self attention，然后将这里的输出和encoder里的输出一起输入下一层。不过这里的self attention是加了mask的。有两种mask,一种是Padding mask，是将pad的字符的权重都设为0。还有一种是sequence mask，是将所有还未出现的word的embedding都调成0，这样的原因是在生成翻译结果的时候模型是看不到当前时间点以后的信息的，因此在训练的时候也需要把以后的信息给mask掉。  
      encoder-decoder-attention：这里对decoder输入的内容和encoder输出的内容一起做attention，可以得到每个中文字符对应每个英文字符的attention weight。这里就是可以得到中文和英文一一对应的重要性  
      最后得到的输出经过几个线性层就可以得到输出了，我们得到的输出是对于下一个词的预测，是基于softmax的，就是从多个候选中选出最合适的那个词。不断将预测出的值输入decoder知道生成了最后一个word。  
      另外，值得一提的是，在每个attention层和feedforward层后都会接一个add&norm层，这层的含义就是将输入和输出值相加（就是残差模块，和resnet里的效果一样，主要就是为了防止梯度爆炸或消失。），然后做一个norm，一般norm是layer norm，当然，我们也可以做batch norm。  
      layer norm：对于同一个数据做norm保证mean=0, std=1  
      batch norm: 对于一个batch里各个数据向量值做norm，保证同一个维度mean=0,std=1    
      </details>
         
# NLP
## 通用方法
* [language model-语言模型](https://zhuanlan.zhihu.com/p/28080127)
  <details>  
  <summary>笔记</summary>  
  
  简单地说，语言模型就是用来计算一个句子的概率的模型，也就是判断一句话是否是人话的概率.
  </details>
      
## 词嵌入(word embedding)
* [八种常用的embedding方式](https://easyai.tech/blog/nlp-%E9%A2%86%E5%9F%9F%E9%87%8C%E7%9A%848-%E7%A7%8D%E6%96%87%E6%9C%AC%E8%A1%A8%E7%A4%BA%E6%96%B9%E5%BC%8F%E5%8F%8A%E4%BC%98%E7%BC%BA%E7%82%B9/)
 
* [fastText](https://www.jiqizhixin.com/articles/2018-06-05-3) 
    <details>
    <summary>什么是fastText？</summary>
       fastText 就是使用了subword n-gram思想，将同一个词语分解成等宽的几个substring，比如apple=[app,ppl,ple]。然后对每个substring分别计算embedding(类似word2vec)，最终词语的embedding是所有substring embedding vector之和。

    一个优势是同一个substring可能出现在不同word中（词根词缀），从而可以找出词与词之间的联系，而且有助于低频词甚至是未出现词语的表达。

    fastText另一个优势是使用了多层softmax用来加速。其实就是把本来的1对N的softmax变成了1对2对4对8...的二叉树形式，每个node就相当于一次逻辑回归，也即sigmoid。
    </details>
* [NNLM-neural network language model]神经网络语言模型(https://zhuanlan.zhihu.com/p/80093548)  
  <details>  
  <summary>笔记</summary>  
  
  NNLM是非常早期的一个模型，这个模型是试图用前t-1个单词来预测第t个单词。  
  一开始是输入每个单词的one hot vector，然后乘以一个矩阵(实质上就是embedding matrix)得到这个词的向量表示。  
  然后输入一个全连接层加tanh激活函数，最后再输入对应字典大小的全连接层加softmax预测结果。  
  ![v2-8fb5317fd5f6be810c9bbdf56f2d33a1_1440w](https://user-images.githubusercontent.com/7517810/108261153-32875c00-7131-11eb-9f60-37db04af73d1.jpg)  
  </details>
* word2vec
  * [skip gram](https://www.bilibili.com/video/BV1gb411j7Bs?p=161)
    <details>  
    <summary>笔记</summary>  
    
    选定中间词（context），预测周围词(target)。并借此优化中间词的embedding。之所以叫skip gram是因为每次都是随机选择预测中词的左边第几个，或者右边第几个，中间可以间隔几个词。  
    基本流程就是：先找到中间词的embedding，然后经过一个线性操作，放入softmax，并用交叉熵作为loss function，最终通过梯度下降来优化embedding。  
    **优化：**  
    使用softmax有一个问题就是我们softmax公式里的除数是词库里所有embedding的和，算这个和很费时间。因此有了hierachical softmax，通过类似二分搜索的形式来确定embedding的所在位置。  
    **随机选择：**  
    也就是说怎么选择预测哪个周围词。最简单的方式就是限定个范围，比如前后十个词以内，然后随机挑。但这样就有个问题，有一些连接词比如of, the, and就会出现的非常多，但这些词通常对中间词的理解没有什么太大的帮助。
    **负采样：**
    先选定一对context和target，并且标记为1，然后随机从词库里挑k个词作为negative target，也即标记为0。  
    如果数据库较小k一般选择5-20，如果较大就选择2-5。一般选negative target的的时候是基于词频的3/4次幂的占比来计算取词概率的，这样就不会出现太多连接词，并且也尽量的平均分布。
    然后用一个logistic regression来分类就行了。
    放在神经网络中也就是：先找到中间词的embedding,然后经过一个线性操作，之后放入sigmoid分别对这k+1个pair进行二分类预测。从而避免了softmax中对所有词的求和。
    </details>
  * [CBOW-continues bag of words](https://easyai.tech/blog/nlp-%E9%A2%86%E5%9F%9F%E9%87%8C%E7%9A%848-%E7%A7%8D%E6%96%87%E6%9C%AC%E8%A1%A8%E7%A4%BA%E6%96%B9%E5%BC%8F%E5%8F%8A%E4%BC%98%E7%BC%BA%E7%82%B9/)
    <details>  
    <summary>笔记</summary> 
  
    使用上下文的词预测中间词
    </details>
* [GloVe-global vectors for word representation](https://www.fanyeong.com/2018/02/19/glove-in-detail/)
    <details>  
    <summary>笔记</summary> 
    
    对于每一对pair:context word和target word，基于固定窗口统计context word和target word的共现次数，记做X(会随着距离衰减)。theta作为target的embedding，e作为context的embedding，通过gradient descent最小化f(x)sum((theta\*e+bi-bj'-log(x))^2 for 所有pair。) ，这里f(x)是一个权重，一个是为了防止x=0时log(x)变成无穷大(f(0)=0)，还有一个就是给词频太大或词频太小的词一个合理的weight.还有一个有意思的部分是，theta和e是对称的，因此最后的embedding通常是取两者的平均（得到的embedding代表了两个词之间的关系，可以通过求和得到某个特定词的vector表示）  
    另外，embedding事实上是比较难以解释的，因为每一个系数都可能是多个不同属性的线性叠加，比如0.2\*性别+0.8\*食物
    ![微信截图_20210215200334](https://user-images.githubusercontent.com/7517810/108006234-e9b49380-6fc8-11eb-8c7d-14d2d1e9c56e.png)
    </details>
* [ELMO]  
  <details>  
  <summary>笔记</summary> 

  使用bi-lstm抽取文本特征
  </details>
* [GPT]  
  <details>  
  <summary>笔记</summary> 

  使用单向transformer抽取文本特征
  </details>
* [Bert-Bidirectional Encoder Representation from Transformers](https://zhuanlan.zhihu.com/p/46652512)  
  <details>  
  <summary>笔记</summary>  

  本质上就是把transformer堆叠在一起罢了。然后使用两大任务进行预训练，从而得到更精准的language model。之所以说bert是双向transformer是因为他的random mask可以同时学习到上下文的知识。
  值得一提的是，bert输入层的embedding和transformer有所不同，是token embedding, segment embedding, position embedding之和。其中segment embedding是用于句子的区分，position embedding不是用三角函数，而是学习出来的。  
  bert使用两大任务进行预训练：  
  1.masked lm，就是随机mask15%的词语，然后用上下文预测这个词。mask方法是10%不替换，10%随机替换成别的词，80%替换成占位符。  
  2.next sentence prediction，就是预测B是不是A的下一个句子。由此获得段落之间的知识。
  </details>
## 机器翻译
  * beam search  
    <details>  
    <summary>笔记</summary>  
    
    在机器翻译过程中greedy search的方式得到的答案并不理想，因为往往局部最优并不代表全局最优，而又不可能所有组合都试一遍，因此就有了beam search.  
    beam search就是同时保持k个局部最优解，从而使得答案更为理想一些。  
    **优化**      
    **length normalization**  
    就是将每一步的概率值取log。因为beam search在取局部最优时比较的是到目前为止的概率之积，由于概率都是小于1的，这就会导致越长的sentence被取到的概率越小。通过取log，概率相乘就变成了Log相加，从而避免了这个问题。
    </details>
  * [Bleu score](https://www.bilibili.com/video/BV1gb411j7Bs?p=171&spm_id_from=pageDriver)
    <details>  
    <summary>笔记</summary>  
  
    用于给翻译出的句子打分。基本思路就是从人类翻译的句子中找相同的词，然后算count之比。 
    机器翻译的词语一个就是一个count，人类翻译句子里每个词语的max count=机器翻译里的max count。  
    例如，人类翻译:what the fuck。机器翻译:what the what hell。    
    则机器翻译的count=2+1+1， 人类翻译的count=1+1+0，p=2/4=1/2。    
    上面的例子是针对gram=1来算的，在实际中还可以用gram=2,3,4...也就是词组的出现次数。      
    ![微信截图_20210216222243](https://user-images.githubusercontent.com/7517810/108151783-8187af80-70a5-11eb-9eee-72f3039c1eda.png)  
    在最后的公式里上面的比例是在e的power上的，然后外面还会加一个惩罚项，是用来防止短句子得分过高  
    ![微信截图_20210216222220](https://user-images.githubusercontent.com/7517810/108151787-82b8dc80-70a5-11eb-9238-5b3b271accca.png)  
    </details>


