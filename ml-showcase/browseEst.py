import pandas as pd
import matplotlib.pyplot as plt # Matplotlib – Python画图工具库
import seaborn as sns # Seaborn – 统计学数据可视化工具库



# plt.plot(df_ads['点赞数'],df_ads['浏览量'],'r.', label='Training data') # 用matplotlib.pyplot的plot方法显示散点图plt.xlabel('点赞数') # x轴Labelplt.ylabel('浏览量') # y轴Labelplt.legend() # 显示图例plt.show() # 显示绘图结果！



data = pd.read_csv('易速鲜花微信软文.csv')
print(data.head())
plt.plot(data['点赞数'], data['浏览量'],'r.', label='Training data')
plt.xlabel('Likes')
plt.ylabel('Views')
plt.legend()
# plt.show()

data2 = pd.concat([data['浏览量'],data['热度指数']], axis=1)
fig = sns.boxplot(x='热度指数', y = '浏览量', data = data2)
fig.axis(ymin=0,ymax=800000)
print(fig)
plt.show()

# data = pd.concat([df_ads['浏览量'], df_ads['热度指数']], axis=1) # 浏览量和热度指数fig = sns.boxplot(x='热度指数', y="浏览量", data=data) # 用seaborn的箱线图画图fig.axis(ymin=0, ymax=800000); #设定y轴坐标

#print(head)
