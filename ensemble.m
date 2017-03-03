%Import X_train, y_train
Xtrain = importdata('X_train.mat');
ytrain = importdata('y_train.mat');

%Import X_test, y_test
Xtest = importdata('X_test.mat');
ytest = importdata('y_test.mat');

ytrain = ytrain';
Y_train_trans = full(ind2vec(ytrain(:,:),10));
Mdl = cell(10,1);
for i = 1:10
    Mdl{i} = fitcsvm(Xtrain, Y_train_trans(i,:), 'KernelFunction','polynomial','PolynomialOrder',2);
end

lab_arr = zeros(3251);
for i = 1:3251
    maxscore = -9999;
    lab = 0;
    for j = 1:10
        [label,score] = predict(Mdl{j},Xtest(i,:));
        if label == 1 & (abs(score(:,1)) > maxscore)
            maxscore = score;
            lab = j;
        end
    end
    lab_arr(i) = lab;
end

Md1 = fitcknn(Xtrain,ytrain,'NumNeighbors',7);


Xtest = Xtest';
Xtrain = Xtrain';
ytrain = full(ind2vec(ytrain));
net = feedforwardnet(25);
net = train(net,Xtrain,ytrain);
yPred = net(Xtest);
yPred = vec2ind(yPred);
yPred = yPred';
Xtrain = Xtrain';
Xtest = Xtest';

cnt = 0;
for i = 1:3251
    svm = lab_arr(i);
    knn = predict(Md1, Xtest(i,:));
    nn = yPred(i);
    act = ytest(i,:);
    
    if svm == knn || svm == nn
        pred = svm;
    elseif knn == svm || knn == nn
        pred = knn;
    elseif nn == svm || nn == knn
        pred = nn;
    else
        pred = knn;
    end
    if pred == act
        cnt = cnt+1;
    end
end

disp('Accuracy of Ensemble');
accuracyens = cnt/32.51
%accuracyens = 95.7552
