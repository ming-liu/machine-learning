X1 = [3,5];
X2 = [4,8];
X3 = X2 - X1;

X = [X1;X2;X3];
y = X(:,2);
x = X(:,1);




plot(x, y, 'rx', 'MarkerSize', 10);
axis([0, 10, 0, 10]);
