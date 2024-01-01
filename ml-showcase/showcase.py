import pandas as pd


fileName = '../coursera/ex1-octave/ex1data1.txt';
names = ['x','y'];
data = pd.read_csv(fileName, header=None, names = names);


print('type(data) = ' , type(data))
print('data[0,0] = ' , data.iloc[0,0])
print('data[0,1] = ' ,data.iloc[0,1])
print('*********************************')
print('*********************************')
print('top 5 of data : ')
print(data.head())


print('*********************************')
print('*********************************')
x = data['x']
y = data['y']
print('type(x) = ', type(x));
print('type(y) = ', type(y));

print('top 5 of x : ')
print(x.head())
