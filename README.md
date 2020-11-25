# 交互式主题建模可视化分析系统
## 介绍
我毕业设计的主要内容是在国内外交互式主题建模的研究基础上，设计一个面向文本分析的交互式主题建模可视化分析系统。系统使用非负矩阵分解(NMF)来作为主题建模的主要算法，并参考了Jaegul Choo等人提出的UTOPIAN主题建模可视化分析系统，对其中的半监督非负矩阵分解(SS-NMF)算法和多种用户交互方式进行了实现及改进。
本系统采用Web技术，后端算法和接口使用Python语言和FastAPI Web框架编写，前端界面使用React&Ant Design框架编写。
![image -12-](https://i.loli.net/2020/11/25/O3GHAa6UVoBFdC2.png)
![image -13-](https://i.loli.net/2020/11/25/PhVHecS4jDaOiCt.png)
上图是以Bilibili网站的生活>美食分区下的[中国最好吃的菜市场？30多个小吃摊，可惜就要关了](https://www.bilibili.com/video/BV1664y1u7gw)的视频评论作为语料库，进行主题建模后的结果的简单展示。
## 系统运行演示
### Bilibili语料库爬虫界面
此界面主要用于抓取Bilibili视频评论数据，用户可以在右侧的Bilibili内嵌网页中复制自己感兴趣的视频的链接，然后选择想要爬取的评论页数，接着爬虫工具将自动爬取评论并写入数据库，最终该视频下的所有评论就可以被当作一个语料库来分析了。通过这种方式，使得用于文本分析的语料库数据更加丰富且灵活了，用户随时可以抓取并分析自己感兴趣的语料数据，而不是从固定的数据中来选择。
![image](https://i.imgur.com/vUcB4uK.png)
![image -1-](https://i.imgur.com/sxM46BS.png)
### 语料库查看/选择界面
在此界面中，用户可以详细地查看语料库中的文档数据，并勾选自己感兴趣的语料库来作为分析对象。以Bilibili视频评论语料库为例，操作界面如下若干图所示：
![image -2-](https://i.imgur.com/RRhaWbf.png)
![image -3-](https://i.imgur.com/dEEvwBC.png)
### 3D主题聚类可视化
![image -8-](https://i.imgur.com/I5eHEfY.png)
![image -14-](https://i.loli.net/2020/11/25/STYZzaoL62FmpOX.png)

### 各种用户交互来优化主题模型
本系统中提供了5种不同的用户交互类型：主题关键词优化、主题拆分、主题合并、关键词诱导的主题创建、文档诱导的主题创建。
● 以主题关键词优化为例：用户可以在主题详细信息抽屉中点击[对此主题进行关键词优化]按钮，界面就会自动跳转到用户交互Tab，在此界面中，用户可以详细调整每个关键词的权重及监督强度，然后就可以通过以上交互信息来进行SS-NMF、优化该主题了，如下：
![image -9-](https://i.loli.net/2020/11/25/9laObPBrh1XtkMR.png)
![image -10-](https://i.loli.net/2020/11/25/buKwI9OPRScZ3Np.png)
### 新文档主题分布预测
用户可以通过当前已训练的主题模型来对新文档进行拟合，从而计算出新文档的主题分布，如下图所示：
![image -11-](https://i.loli.net/2020/11/25/q5brufnDF96SKhE.png)
## 用到的框架、库

### 后端
Python
Streamlit
FastAPI 

### 数据库
SQLite
SQLAlchemy ORM
Navicat Premium

### 前端
JavaScript框架：React
路由：React Router
UI组件库：Ant Design of React

### 数据可视化库
Plotly.js
## 论文
[面向文本分析的交互式主题建模 - 浙江工业大学软件工程1605朱思威](https://docs.qq.com/doc/DV0JsSHJkWFlmQVFz) 












