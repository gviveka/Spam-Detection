# install libraries
install.packages("pls")
install.packages("boot")
install.packages("ipred")
install.packages("RWeka")
install.packages("e1071")

# Load the dataset
spambase <- read_csv("C:/course work/613/project/spambase.csv")
View(spambase)

#PCA: creating principal components
pca1=prcomp(na.omit(spambase[-58]),center = T,scale. = T)
print(pca1)
plot(pca1,type='l')
axes=predict(pca1,newdata=spambase)
data=cbind(spambase,axes)

sample = sample.split(data, SplitRatio = 0.7)
train = subset(data, sample == TRUE)
test = subset(data, sample == FALSE)
testing_y=test$Response

#GLM:

logistic_model = glm(Response~PC1+PC2+PC3+PC4+PC5+PC6 , data = data, family = "binomial") 
logistic_probs = predict(logistic_model,test, type ="response")
logistic_pred_y = rep("0",length(test$Response))
logistic_pred_y[logistic_probs > 0.5] = "1"
table(logistic_pred_y, test$Response)
mean(logistic_pred_y != test$Response)
cv.error=cv.glm(data,logistic_model,K=10)$delta[1]
cv.error

#LDA:

lda.fit=lda(Response~PC1+PC2+PC3+PC4+PC5,data=train)
lda.fit
lda.pred=predict(lda.fit ,test,type="Response")
names(lda.pred)
lda.class=lda.pred$class
table(lda.class ,test$Response)
mean(lda.class!=test$Response)

#QDA:

qda.fit=qda(Response~PC1+PC2+PC3+PC4+PC5,data=train,cv=TRUE)
qda.pred=predict(qda.fit ,test,type="Response")
names(qda.pred)
qda.class=qda.pred$class
table(test$Response,qda.class)
mean(qda.class!=test$Response)
X=cbind(train[,c(59,60,61,62,63)])
Y=cbind(train[,58])
lda_wrapper=function(object,newdata){predict(object,newdata)$class}
errorest_cv(x=X,y=Y,train = qda,classify = lda_wrapper,num_folds = 10)

#KNN:

library(class)
pc1<-train$PC1
pc2<-train$PC2
pc3<-train$PC3
pc4=train$PC4
pc5=train$PC5
tpc1<-test$PC1
tpc2<-test$PC2
tpc3<-test$PC3
tpc4=test$PC4
tpc5=test$PC5
train.X=cbind(pc1,pc2,pc3,pc4,pc5)
test.X=cbind(tpc1,tpc2,tpc3,tpc4,tpc5)
train.Y=train$Response
test.Y=test$Response
knn.pred=knn(train.X,test.X,train.Y,k=2)
table(knn.pred,test.Y)
mean(knn.pred!=test.Y)
library(chemometrics)
X=data.frame(data)
grp=as.factor(data$Response)
tr=sample(dim(data)[1],dim(data)[1]*0.7)
knnEval(X,grp,tr,kfold=10,knnvec = seq(1,10,by=5),plotit = T)

#J-48
library(RWeka)
data$Response = factor(data$Response)
j48model = J48(Response ~., data = train, control = Weka_control(R= TRUE))
j48model
train$Response = factor (train$Response)
test$Response=factor(test$Response)
j48model = J48(Response ~., data = train, control = Weka_control(R= TRUE))
j48.pred = predict(j48model, newdata= test)
table(j48.pred,test$Response)
mean(j48.pred != test$Response)
roc(j48.pred,test$Response,plot=T)


#SVM
library(e1071)
svmmodel=svm(Response~.-Response,kernel="linear",data=train,cost=0.1)
svm.probs=predict(svmmodel,newdata=test)
table(svm.probs,test$Response)
mean(svm.probs!=test$Response)
