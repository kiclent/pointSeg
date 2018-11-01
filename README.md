# pointSeg
DataFountain 自动驾驶点云分割赛开源
环境 Python3.6 & Tensouflow 1.7.1 
复赛成绩：0.85+（结果有点惨）
最近有点忙，这个脑洞就不跟大家解释那么快了，后面再整个文档给大家。
# 1、数据及任务描述。
官方描述：https://www.datafountain.cn/competitions/314/details/data-evaluation

# 2、数据处理。
由于大量的小文件读写，严重影响模型加载数据的速率，成为影响模型训练时间的一大因素，因此，需要将文件合并成一个稍大一点的文件（合并成一个文件不现实）， 以减少文件的IO次数。

文件合并程序 ：data_merge.py
本程序合并策略是将每100帧的pts, intensity, category合并在一个(n行， 6列)矩阵上，并保存为numpy的数值（.npy），数值精度使用的是np.float32（读取的时候这个一定要对应，不然数值会不一样）

flame_index pts intensity category
1 1,1,1 0.5 0
2 1,1,1 0.5 0
……   

# 3、使用
设置好相关的路径，运行 data_merge.py 将大量csv文件合并成一个大的文件。
设置好npy路径，运行 pointSeg.py （参数is_training = True 用于训练， is_trainig = False 用于预测）


