function [error_test] = ...
    calTest(X, y,X_test,ytest,lambda)

    theta = trainLinearReg(X,y,lambda);
    error_test = linearRegCostFunction(X_test,ytest,theta,0);
end
