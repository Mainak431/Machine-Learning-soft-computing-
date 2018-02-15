%load the data which has pixel value for river and not a river
function losgistic_river
    river_data = load('my_datar4.txt');
    not_river_data = load('my_datanr4.txt');

    X = [river_data(:,1); not_river_data(:,1)];
    Y = [river_data(:,2); not_river_data(:,2)];
    disp(Y)
    %normalize the data
    x1_mean = mean(X);
    x1_std = std(X);
    X1 = (X - x1_mean)./ x1_std;

    X2 = X.^2;
    x2_mean = mean(X2);
    x2_std = std(X2);
    X2 = (X2 - x2_mean) ./ x2_std;


    m = length (X);
    %select the hypothesis function h = g(w0 + w1 * x + w2 * x^2 )

    X = [ones(m,1) X1 X2];
    W = zeros(3,1);
    disp(length(X))
    %update the W value
    iteration = 500;
    alpha = 0.1;
    
    j_val = [];
    for i = 1:iteration
        H = sigmoid(X*W);
        J = sum( - ((Y .* log(H)) +( (1 - Y) .* log(1 - H))));
        grad = X' * (Y - H);

        W = (W )+ ((alpha/m) .* grad);

    end

    test = imread('4.gif');
    shape_test = size(test);
    rows = shape_test(1);
    columns = shape_test(2);
    predict = zeros(rows,columns);
    for i=1:rows
         for j=1:columns
             predict(i,j) = [1 ,((double(test(i,j)) - x1_mean)./ x1_std ),((double(test(i,j))^2 - x2_mean)./ x2_std )] * W;
         end
    end
    predict = sigmoid(predict);
    

    predict(predict >= 0.5) = 1;
    predict(predict < 0.5) = 0;
    figure();
    imshow(predict);
    %define the sigmoid function
    function out= sigmoid(Z)
        out = 1 ./ (1 + exp(-Z));
    end
end