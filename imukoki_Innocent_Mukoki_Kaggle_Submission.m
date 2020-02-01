%% Kaggle
train = readtable('train.csv');
test = readtable('test.csv');
% Fitting a logistic regression model
pClass = train.Pclass;

for index = 1:891
    if(strcmp(train.Sex(index), 'male') == 1)
       sex1(index) = 0;
    else
        sex1(index) = 1;
    end
end
sex = sex1';

%Replacing training data set with mean
% Replacing the Nans in age with the mean of the ages
for indTr = 1:891
    if(isnan(train.Age(indTr)) == 1)
       age(indTr) = nanmean(train.Age);
    else
        age(indTr) = train.Age(indTr);
    end
end
ageTr = age';

Fare = train.Fare;

for indG  = 1:891
    if(ageTr(indG) <= 18)
        ageTr(indG) = 1;
    elseif(ageTr(indG) > 18 & ageTr(indG) <= 35)
        ageTr(indG) = 2;
    elseif(ageTr(indG) > 35)
        ageTr(indG) = 3;
    end
end
ageTrain = ageTr;

Survived = train.Survived;
trainLog = table(pClass, sex, ageTrain);
mdl = fitglm(table2array(trainLog),Survived,'Distribution','binomial')

%Test data set
pClassT = test.Pclass;
ageT = test.Age;
for ind = 1:418
    if(strcmp(test.Sex(ind), 'male') == 1)
       sex2(ind) = 0;
    else
        sex2(ind) = 1;
    end
end
sexT = sex2';

% Replacing the Nans in age with the mean of the ages
for indTe = 1:418
    if(isnan(test.Age(indTe)) == 1)
       age2(indTe) = nanmean(test.Age);
    else
        age2(indTe) = test.Age(indTe);
    end
end
ageTe = age2';

FareT = test.Fare;
for indTest  = 1:418
    if(ageTe(indTest) <= 18)
        ageTe(indTest) = 1;
    elseif(ageTe(indTest) > 18 & ageTe(indTest) <= 35)
        ageTe(indTest) = 2;
    elseif(ageTe(indTest) > 35)
        ageTe(indTest) = 3;
    end
end
ageTest = ageTe
testLog = table(pClassT,sexT,ageTest);
% Performance of the model
mdlPerf = round(predict(mdl, table2array(testLog)))

Kaggle = table(test.PassengerId,mdlPerf);
Kaggle.Properties.VariableNames = {'PassengerId','Survived'};
writetable(Kaggle,'Kaggle_Submission.csv')