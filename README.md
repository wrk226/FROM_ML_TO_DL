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
    
    * [Transformer]()  
      <details>  
      <summary>笔记</summary>  
  
      </details>
    * [Bert]()  
      <details>  
      <summary>笔记</summary>  
  
      </details>
      
# NLP
## 词嵌入(word embedding)
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
  


