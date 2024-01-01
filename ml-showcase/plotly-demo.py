import plotly.express as px

# a = dict(str1=value1, str2=value2, str3=value3)
# str 表示字符串类型的键，value 表示键对应的值。使用此方式创建字典时，字符串不能带引号。
# data = dict(number = [59, 32, 18, 9, 2],stage = ['访问数','下载数','注册数','搜索数','付款数'])

data = {}
data['number'] = [59, 32, 18, 9, 2]
data['stage'] = ['访问数','下载数','注册数','搜索数','付款数']

fig = px.funnel(data,x='number',y='stage')
fig.show()
