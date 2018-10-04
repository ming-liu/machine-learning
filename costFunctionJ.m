function J = costFunctionJ(X,y,theta)

 pridiction = X * theta;
 sqrErrors = (pridiction - y) .^ 2;

 m = size(X,1);
 J = 1/(2*m) * sum(sqrErrors);


