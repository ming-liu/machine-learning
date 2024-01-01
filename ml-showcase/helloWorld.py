''' hello world '''

print('----------Hello World!----------------')
print('Hello world!')
print("Hello world!")
print('''Hello world!''')
print("""Hello world!""")


print('----------x=123,y=abc!----------------')
x = 123
y = 'abc'

print(x)
print(y)

print(x)
print(y)

print('----------Dict----------------')
dc = {}
dc['key1'] = 123
dc['key2'] = 'abc'

# 这个name到底是啥??? 怎么会变成字符串的?
data = dict(name = 'lily',age = 18)
### dict(number = [59, 32, 18, 9, 2],stage = ['访问数','下载数','注册数','搜索数','付款数'])

print(dc)
print(data)
print(data['name'])
