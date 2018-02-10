# 知乎看山杯机器学习挑战赛
Koala队(2/964)&nbsp;&nbsp;|&nbsp;&nbsp;[赛题链接](https://biendata.com/competition/zhihu/)

#### 任务：
> 根据知乎给出的问题及话题标签的绑定关系的训练数据，训练出对未标注数据自动标注的模型。
> 标注数据中包含 300 万个问题，每个问题有 1 个或多个标签，共计1999 个标签。每个标签对应知乎上的一个「话题」，话题之间存在父子关系，并通过父子关系组织成一张有向无环图（DAG）。

#### code
##### 目录说明
* data 原始数据目录
* cache 缓存文件路径
* models 模型代码
* train 模型训练脚本
* ensemable 集成模型训练及融合脚本
* utils 数据处理及其他脚本

##### 运行说明
```bash
## 构建线下训练验证集，生成序列文件
python3 ./utils/data_preprocess.py
## 后续即可运行train目录下脚本训练模型
```
