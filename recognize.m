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

cnt = 0;
for i = 1:3251
    if ytest(i,:) == lab_arr(i)
        cnt = cnt+1;
    end
end
disp('Accuracy of SVM');
accuracysvm = cnt/32.51
%accuracysvm = 90.4030%


t = 0;
f = 0;
Md1 = fitcknn(Xtrain,ytrain,'NumNeighbors',7);

for i = 1:3251
	ypred = predict(Md1, Xtest(i,:));
	yact = ytest(i,:);
	if ypred == yact
		t = t+1;
	else
		f = f+1;
	end
end

disp('Accuracy of K Nearest Neighbors');
accuracyknn = t*100.0/(t+f)
%accuracyknn = 95.2015%


count_match = 0;
Xtest = Xtest';
Xtrain = Xtrain';
ytrain = full(ind2vec(ytrain));
net = feedforwardnet(25);
net = train(net,Xtrain,ytrain);
yPred = net(Xtest);
yPred = vec2ind(yPred);
yPred = yPred';
Xtrain = Xtrain';
for i = 1:size(yPred,1)
    if yPred(i) == ytest(i)
        count_match = count_match + 1;
    end
end
disp('Accuracy of Neural Network');
accuracyann = count_match/32.51
%accuracyann = 89.6032%