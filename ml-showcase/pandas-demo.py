import pandas as pd
import plotly.express as px

stages = ['访问数','下载数','注册数','搜索数','付款数']

df_male = pd.DataFrame(dict(number = [30,15,10,6,1], stage = stages))
df_male['Sex'] = 'Male'

df_female = pd.DataFrame(dict(number = [29,17,8,3,1], stage = stages))
df_female['Sex'] = 'Female'

print(df_male)
print(df_female)

df = pd.concat([df_male, df_female], axis = 0)
print(df)

fig = px.funnel(df, x = 'number' ,y = 'stage', color = 'Sex')
fig.show()
