<!-- 包含公式，对Github上查看不友好 -->

# 参考文档及代码

[scikit-learn:Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html)

[scikit-learn:naive_bayes.py](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/naive_bayes.py)

### 概述

朴素贝叶斯是**基于贝叶斯定理**和**特征条件独立假设**的分类方法。对于给定的训练数据集，首先基于特征条件独立假设学习输入/输出的**联合概率分布**；然后基于此模型，对于给定的输入$X$，利用贝叶斯定理求出**后验概率最大**的输出$y$。

朴素贝叶斯是生成模型。

### Naive Bayes

贝叶斯分类方法是一系列有监督学习方法，其应用贝叶斯理论，并使用“各特征对之间相互独立”的假设(Naive assumption)。

给定类别$y$和一个相互依赖的特征向量$(x_1 \cdots x_n)$ , 贝叶斯理论描述了如下理论：

$$
p(y|x_1 , \cdots , x_n) = \frac{p(x_1 \cdots x_n | y) p(y)}{ p(x_1 ,\cdots , x_n) }
$$


使用朴素的独立性假设，有 $ p(x\_i | y , x\_1 , \cdots , x\_{i-1}) = p(x\_i | y)  $

由此得到


$$
p(y|x_1 , \cdots , x_n) = \frac{ \Pi_{i=1}^n{p(x_i | y) }  p(y) }{ p(x_1 ,\cdots , x_n) }
$$


对于分类而言，$p(x\_1 , \cdots , x\_n)$是定值，因而有


$$
p(y|x_1 , \cdots , x_n) \propto  \Pi_{i=1}^n{p(x_i | y) }  p(y) 
$$


有


$$
\hat y = \arg \max\_y { Pi\_{i=1}^n{p(x\_i | y) }  p(y)  }
$$


最后，不同的Naive Bayes  ， 其对$p(y\|x)$的计算不同。
