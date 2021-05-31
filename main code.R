#In this paper, we utilize a large set of firm characteristics as our predictors of stock returns
#and involve the forecasting approaches including OLS, Lasso, principal components regression, 
#neural network and forecasting combination with different weight schemes to do out-of-sample 
#forecasting through rolling window method. 

rm(list=ls())
set.seed(123)
library(glmnet)
library(neuralnet)
library(pls)
DataSet=read.csv("/Users/Desktop/procdata.csv")[,-1]
attach(DataSet)

# Drop variables with over 20% missingness
num.na=rep(NA,ncol(DataSet))
for (i in 1:ncol(DataSet)) {
  num.na[i]=sum(is.na(DataSet[,i]))
}
DropVar=which(num.na>0.2*nrow(DataSet))
colnames(DataSet)[DropVar]
DataSet=DataSet[,-DropVar]
# Drop variables with constant value
Var_var=rep(NA,ncol(DataSet))
for (i in 1:ncol(DataSet)) {
  Var_var[i]=var(DataSet[,i],na.rm = TRUE)
}
DropVar=which(Var_var==0)
colnames(DataSet)[DropVar]
DataSet=DataSet[,-DropVar]

# Standardization by month
for (Y in min(Year):max(Year)) {
  for (M in 1:12){
    ScaleIndex=which(Year==Y & Month==M)
    DataSet[ScaleIndex,-c(1,2,3,4)]=scale(DataSet[ScaleIndex,-c(1,2,3,4)])
  }
}
DataSet[is.na(DataSet)]=0
detach(DataSet)
attach(DataSet)
DataX=as.matrix(DataSet[,-c(1,2,3,4)])

DJI=c(59176,22592,14593,19561,18542,14541,76076,11308,16851,11850,19502,86868,12490,59328,22111,
      47896,43449,22752,10107,57665,21936,18163,66181,59459,92655,17830,65875,26403,92611,55976)#permno: codename of each company

# Creat arrays to store the predictions
WL=3 #number of rolling window methods
K=7

PredSet=array(dim = c(WL,K,nrow(DataSet)))
colnames(PredSet)=c("Benchmark","OLS", "Lasso","PCR","NN1","NN2","NN3")

UniForecast=array(dim = c(WL,ncol(DataX),nrow(DataX)))
colnames(UniForecast)=colnames(DataX)

ComForecast=array(dim = c(WL,10,nrow(DataSet)))
colnames(ComForecast)=c("uniMean","uniMedian","uniTrimMean","uniMSE_W","uniCov_W",
                        "methodsMean","methodsMedian","methodsTrimMean","methodsMSE_W","methodsCov_W")

lasso.var=matrix(0,nrow = ncol(DataX),ncol = WL+1)
rownames(lasso.var)=colnames(DataX)
lasso.num=matrix(nrow = 120,ncol = WL)
pc.num=matrix(nrow = 120,ncol = WL)

MSFE=array(dim = c(WL,120,K))
dimnames(MSFE)[[3]]=colnames(PredSet)
  
ComMSFE=array(dim = c(WL,120,10))
dimnames(ComMSFE)[[3]]=colnames(ComForecast)

gamma=1/6
lb1=ceiling(ncol(DataX)*gamma)+1
ub1=ceiling(ncol(DataX)*(1-gamma))
lb2=ceiling((K-1)*gamma)+1
ub2=ceiling((K-1)*(1-gamma))

Error=0
# Rolling Window
for (W in 1:WL) {
  CountIndex=1
  for (Y in min(Year):max(Year)) {
    for (M in 1:12) {
      TestIndex=which(Year==Y+W & Month==M)
      if(length(TestIndex)==0)
        break
      TrainIndex=which((Year==Y & Month>=M) | (Year==Y+W & Month<M) | (Year>Y & Year<Y+W))

      TrainSet=DataSet[TrainIndex,-c(1,2,3)]
      X_Train=as.matrix(TrainSet[,-1])
      TestSet=DataSet[TestIndex,-c(1,2,3)]
      X_Test=as.matrix(TestSet[,-1])
      
      TrainError=matrix(nrow = length(TrainIndex),ncol = K-1)
      colnames(TrainError)=colnames(PredSet)[-1]
      UniTrianError=matrix(nrow = length(TrainIndex),ncol = ncol(DataX))
      colnames(UniTrianError)=colnames(UniForecast)

      TrainMSE=rep(0,K-1)
      names(TrainMSE)=colnames(PredSet)[-1]
      UniTrianMSE=rep(0,ncol(DataX))
      names(UniTrianMSE)=colnames(UniForecast)
            
      # History Average (Benchmark)----------------------------------------
      for (ip in DJI) {
        PredSet[W,"Benchmark",which(permno==ip & (rownames(DataSet) %in% TestIndex))]=
          mean(Ret[which(permno==ip & (rownames(DataSet) %in% TrainIndex))])
      }
      MSFE[W,CountIndex+12*W,"Benchmark"]=mean((PredSet[W,"Benchmark",TestIndex]-Ret[TestIndex])^2)


      # OLS (Kitchen Sink)-------------------------------------------------
      ols.fit=lm(Ret~., data = TrainSet)
      PredSet[W,"OLS",TestIndex]=predict(ols.fit,TestSet)
      MSFE[W,CountIndex+12*W,"OLS"]=mean((Ret[TestIndex]-PredSet[W,"OLS",TestIndex])^2)
      TrainError[,"OLS"]=Ret[TrainIndex]-predict(ols.fit,TrainSet)
      TrainMSE["OLS"]=mean(TrainError[,"OLS"]^2)

      # Lasso--------------------------------------------------------------
      optlambda=cv.glmnet(X_Train,TrainSet$Ret,alpha = 1)$lambda.min
      lasso.fit=glmnet(X_Train,TrainSet$Ret,alpha = 1,lambda = optlambda)
      PredSet[W,"Lasso",TestIndex]=predict(lasso.fit,newx=X_Test)
      MSFE[W,CountIndex+12*W,"Lasso"]=mean((Ret[TestIndex]-PredSet[W,"Lasso",TestIndex])^2)
      TrainError[,"Lasso"]=Ret[TrainIndex]-predict(lasso.fit,newx=X_Train)
      TrainMSE["Lasso"]=mean(TrainError[,"Lasso"]^2)
      
      lasso.num[CountIndex,W]=lasso.fit$df
      lasso.var[which(lasso.fit$beta!=0),W]=lasso.var[which(lasso.fit$beta!=0),W]+1

      # PCR----------------------------------------------------------------
      pcr.fit=pcr(Ret~.,data=TrainSet, center=FALSE, scale=FALSE)
      pvar=cumsum(pcr.fit$Xvar)/pcr.fit$Xtotvar
      ncomp=min(which(pvar>=0.8))
      PredSet[W,"PCR",TestIndex]=predict(pcr.fit,TestSet,ncomp=ncomp)
      MSFE[W,CountIndex+12*W,"PCR"]=mean((Ret[TestIndex]-PredSet[W,"PCR",TestIndex])^2)
      TrainError[,"PCR"]=Ret[TrainIndex]-predict(pcr.fit,TrainSet,ncomp=ncomp)
      TrainMSE["PCR"]=mean(TrainError[,"PCR"]^2)
      
      pc.num[CountIndex,W]=ncomp

      # Neural Network-----------------------------------------------------
      
      nn1.fit=neuralnet(Ret~.,TrainSet,hidden = 1)
      PredSet[W,"NN1",TestIndex]=compute(nn1.fit,TestSet)$net.result
      MSFE[W,CountIndex+12*W,"NN1"]=mean((Ret[TestIndex]-PredSet[W,"NN1",TestIndex])^2)
      TrainError[,"NN1"]=Ret[TrainIndex]-compute(nn1.fit,TrainSet)$net.result
      TrainMSE["NN1"]=mean(TrainError[,"NN1"]^2)
      
      nn2.fit=neuralnet(Ret~.,TrainSet,hidden = c(1,1))
      PredSet[W,"NN2",TestIndex]=compute(nn2.fit,TestSet)$net.result
      MSFE[W,CountIndex+12*W,"NN2"]=mean((Ret[TestIndex]-PredSet[W,"NN2",TestIndex])^2)
      TrainError[,"NN2"]=Ret[TrainIndex]-compute(nn2.fit,TrainSet)$net.result
      TrainMSE["NN2"]=mean(TrainError[,"NN2"]^2)
      
      nn3.fit=neuralnet(Ret~.,TrainSet,hidden = c(1,1,1))
      PredSet[W,"NN3",TestIndex]=compute(nn3.fit,TestSet)$net.result
      MSFE[W,CountIndex+12*W,"NN3"]=mean((Ret[TestIndex]-PredSet[W,"NN3",TestIndex])^2)
      TrainError[,"NN3"]=Ret[TrainIndex]-compute(nn3.fit,TrainSet)$net.result
      TrainMSE["NN3"]=mean(TrainError[,"NN3"]^2)
      
      
      # =========================Combination Forecasting=========================
      
      # UniVariate Forecasting
      for (i in colnames(DataX)) {
        uni.fit=lm(Ret~.,data = DataSet[TrainIndex,c("Ret",i)])
        UniForecast[W,i,TestIndex]=predict(uni.fit,TestSet)
        UniTrianError[,i]=Ret[TrainIndex]-predict(uni.fit,TrainSet)
        UniTrianMSE[i]=mean(UniTrianError[,i]^2)
      }
      
      # Equal Weights------------------------------------------------------
      ComForecast[W,"uniMean",TestIndex]=apply(UniForecast[W,,TestIndex], 2, mean)
      ComMSFE[W,CountIndex+12*W,"uniMean"]=mean((Ret[TestIndex]-ComForecast[W,"uniMean",TestIndex])^2)
      
      ComForecast[W,"methodsMean",TestIndex]=apply(PredSet[W,-1,TestIndex], 2, mean)
      ComMSFE[W,CountIndex+12*W,"methodsMean"]=mean((Ret[TestIndex]-ComForecast[W,"methodsMean",TestIndex])^2)
      
      # Median-------------------------------------------------------------
      ComForecast[W,"uniMedian",TestIndex]=apply(UniForecast[W,,TestIndex], 2, median)
      ComMSFE[W,CountIndex+12*W,"uniMedian"]=mean((Ret[TestIndex]-ComForecast[W,"uniMedian",TestIndex])^2)
      
      ComForecast[W,"methodsMedian",TestIndex]=apply(PredSet[W,-1,TestIndex], 2, median)
      ComMSFE[W,CountIndex+12*W,"methodsMedian"]=mean((Ret[TestIndex]-ComForecast[W,"methodsMedian",TestIndex])^2)
      
      # Trimmed Mean-------------------------------------------------------
      for (j in TestIndex) {
        ComForecast[W,"uniTrimMean",j]=mean(sort(UniForecast[W,,j])[lb1:ub1])
        ComForecast[W,"methodsTrimMean",j]=mean(sort(PredSet[W,-1,j])[lb2:ub2])
      }
      ComMSFE[W,CountIndex+12*W,"uniTrimMean"]=mean((Ret[TestIndex]-ComForecast[W,"uniTrimMean",TestIndex])^2)
      ComMSFE[W,CountIndex+12*W,"methodsTrimMean"]=mean((Ret[TestIndex]-ComForecast[W,"methodsTrimMean",TestIndex])^2)

      # Time-varying MSE based weights-------------------------------------
      Weight.uni.MSE=(1/UniTrianMSE)/sum(1/UniTrianMSE)
      ComForecast[W,"uniMSE_W",TestIndex]=apply(Weight.uni.MSE*UniForecast[W,,TestIndex], 2, sum)
      ComMSFE[W,CountIndex+12*W,"uniMSE_W"]=mean((Ret[TestIndex]-ComForecast[W,"uniMSE_W",TestIndex])^2)
      
      Weight.methods.MSE=(1/TrainMSE)/sum(1/TrainMSE)
      ComForecast[W,"methodsMSE_W",TestIndex]=apply(Weight.methods.MSE*PredSet[W,-1,TestIndex], 2, sum)
      ComMSFE[W,CountIndex+12*W,"methodsMSE_W"]=mean((Ret[TestIndex]-ComForecast[W,"methodsMSE_W",TestIndex])^2)
      
      # Time-varying Covariance based weights------------------------------
      l=rep(1,ncol(DataX))
      CatchError=try(solve(cov(UniTrianError)),silent = TRUE)
      if('try-error' %in% class(CatchError)){
        Weight.uni.Cov=rep(1/ncol(DataX),ncol(DataX))
        Error=Error+1
      }else
        Weight.uni.Cov=as.vector((solve(cov(UniTrianError))%*%l) / as.numeric(t(l)%*%solve(cov(UniTrianError))%*%l))
      
      ComForecast[W,"uniCov_W",TestIndex]=apply(Weight.uni.Cov*UniForecast[W,,TestIndex], 2, sum)
      ComMSFE[W,CountIndex+12*W,"uniCov_W"]=mean((Ret[TestIndex]-ComForecast[W,"uniCov_W",TestIndex])^2)
      
      l=rep(1,K-1)
      Weight.methods.Cov=as.vector((solve(cov(TrainError))%*%l) / as.numeric(t(l)%*%solve(cov(TrainError))%*%l))
      ComForecast[W,"methodsCov_W",TestIndex]=apply(Weight.methods.Cov*PredSet[W,-1,TestIndex], 2, sum)
      ComMSFE[W,CountIndex+12*W,"methodsCov_W"]=mean((Ret[TestIndex]-ComForecast[W,"methodsCov_W",TestIndex])^2)
      
      
      
      CountIndex=CountIndex+1
    }
    
  }
  
}


summary(pc.num)
summary(lasso.num)

sort(lasso.var[,1],decreasing = TRUE)[1:10]
sort(lasso.var[,2],decreasing = TRUE)[1:10]
sort(lasso.var[,3],decreasing = TRUE)[1:10]
lasso.var[,WL+1]=rowSums(lasso.var)
sort(lasso.var[,WL+1],decreasing = TRUE)[1:10]



MSE=matrix(nrow = ncol(PredSet)+ncol(ComForecast), ncol = WL)
rownames(MSE)=c(colnames(PredSet),colnames(ComForecast))

R2_OS=matrix(nrow = ncol(PredSet)+ncol(ComForecast), ncol = WL)
rownames(R2_OS)=c(colnames(PredSet),colnames(ComForecast))

CSFE=array(dim = c(WL,120,ncol(PredSet)+ncol(ComForecast)))
dimnames(CSFE)[[3]]=c(colnames(PredSet),colnames(ComForecast)) 

for (W in 1:WL) {
  for (i in rownames(MSE)) {
    if(i %in% colnames(PredSet)){
      MSE[i,W]=mean(MSFE[W,,i], na.rm=TRUE)
      CSFE[W,(12*W+1):120,i]=cumsum(MSFE[W,(12*W+1):120,i]-MSFE[W,(12*W+1):120,"Benchmark"])
    }else{
      MSE[i,W]=mean(ComMSFE[W,,i], na.rm=TRUE)
      CSFE[W,(12*W+1):120,i]=cumsum(ComMSFE[W,(12*W+1):120,i]-MSFE[W,(12*W+1):120,"Benchmark"])
    }
    R2_OS[i,W]=(1-(MSE[i,W]/MSE["Benchmark",W]))*100
  }
}
R2_OS[which(R2_OS[,1]>1),1] 
R2_OS[which(R2_OS[,2]>1),2] 
R2_OS[which(R2_OS[,3]>1),3] 


MAE=matrix(nrow = ncol(PredSet)+ncol(ComForecast), ncol = WL)
rownames(MAE)=c(colnames(PredSet),colnames(ComForecast))
for (W in 1:WL) {
  for (i in rownames(MAE)) {
    if(i %in% colnames(PredSet)){
      MAE[i,W]=mean(abs(Ret[Year>=(min(Year)+W)]-PredSet[W,i,Year>=(min(Year)+W)]))
    }else
      MAE[i,W]=mean(abs(Ret[Year>=(min(Year)+W)]-ComForecast[W,i,Year>=(min(Year)+W)]))
  }
}


# Clark-West test, comparing forecasts from nested models=========================

CW=array(dim = c(WL,length(Ret),ncol(PredSet)+ncol(ComForecast)))
dimnames(CW)[[3]]=c(colnames(PredSet),colnames(ComForecast))
for (W in 1:WL) {
  for (i in dimnames(CW)[[3]]) {
    if(i %in% colnames(PredSet)){
      CW[W,,i]=(Ret-PredSet[W,"Benchmark",])^2-(Ret-PredSet[W,i,])^2+
        (PredSet[W,"Benchmark",]-PredSet[W,i,])^2
    }else{
      CW[W,,i]=(Ret-PredSet[W,"Benchmark",])^2-(Ret-ComForecast[W,i,])^2+
        (PredSet[W,"Benchmark",]-ComForecast[W,i,])^2
    }
  }
}

CW.pvalue=matrix(nrow = ncol(PredSet)+ncol(ComForecast), ncol = WL)
rownames(CW.pvalue)=c(colnames(PredSet),colnames(ComForecast))
for (W in 1:WL){
  for (i in rownames(CW.pvalue)) {
    TempSet=data.frame(CW=CW[W,Year>=(min(Year)+W),i])
    cw.fit=lm(CW~.,TempSet)
    cw.t=summary(cw.fit)$coef[3]
    CW.pvalue[i,W]=1-pnorm(cw.t)
  }
}

CW.pvalue[which(CW.pvalue[,1]<0.05),1]
CW.pvalue[which(CW.pvalue[,2]<0.05),2]
CW.pvalue[which(CW.pvalue[,3]<0.05),3]

CW.pvalue[which(CW.pvalue[,1]<0.01),1]
CW.pvalue[which(CW.pvalue[,2]<0.01),2]
CW.pvalue[which(CW.pvalue[,3]<0.01),3]



# Model confidence set=========================================================
alpha=0.01
T_alpha=qnorm(1-alpha/2)
Loss=array(dim = c(WL,120,ncol(PredSet)+ncol(ComForecast)))
dimnames(Loss)[[3]]=c(colnames(PredSet),colnames(ComForecast))
Loss[,,colnames(PredSet)]=MSFE[,,colnames(PredSet)]
Loss[,,colnames(ComForecast)]=ComMSFE[,,colnames(ComForecast)]

include=list(MCS1=dimnames(Loss)[[3]],MCS2=dimnames(Loss)[[3]],MCS3=dimnames(Loss)[[3]])

for (W in 1:WL) {
  L_temp=Loss[W,,]

  repeat{
    dij=matrix(nrow = ncol(L_temp),ncol = ncol(L_temp))
    rownames(dij)=colnames(L_temp)
    colnames(dij)=colnames(L_temp)
    tij=matrix(nrow = ncol(L_temp),ncol = ncol(L_temp))
    rownames(tij)=colnames(L_temp)
    colnames(tij)=colnames(L_temp)
    for (i in colnames(L_temp)) {
      for(j in colnames(L_temp)) {
        dij[i,j]=mean(L_temp[,i]-L_temp[,j],na.rm=TRUE)
        tij[i,j]=dij[i,j]/var(L_temp[,i]-L_temp[,j],na.rm=TRUE)
      }
    }
    Tmax=max(abs(tij),na.rm = TRUE)
    if(Tmax<T_alpha)
      break
    exclude=which.max(rowSums(dij))
    include[[W]]=include[[W]][-exclude]
    L_temp=L_temp[,-exclude]
    if(length(ncol(L_temp))==0)
      break
  }
}
include


iYear=c(min(Year+1):max(Year))
iWindow=12*(iYear-min(Year))+1
par(mfrow=c(2,3),lwd=4,cex.main=4,font.axis=2,cex.axis=2)
for (i in dimnames(CSFE)[[3]][2:7]) {
  plot(CSFE[1,,i],type = "l",col=2,xaxt="n",main = i,xlab = '',ylab = '',
       xlim = c(17,115),ylim = c(min(CSFE[,,i],na.rm = TRUE),max(CSFE[,,i],na.rm = TRUE)))
  lines(CSFE[2,,i],col=3)
  lines(CSFE[3,,i],col=4)
  abline(h=0,lty=2)
  axis(side = 1,at = iWindow,labels = iYear)
}

par(mfrow=c(2,3),lwd=4,cex.main=4,font.axis=2,cex.axis=2)
for (i in dimnames(CSFE)[[3]][c(8:10,13:15)]) {
  plot(CSFE[1,,i],type = "l",col=2,xaxt="n",main = i,xlab = '',ylab = '',
       xlim = c(17,115),ylim = c(min(CSFE[,,i],na.rm = TRUE),max(CSFE[,,i],na.rm = TRUE)))
  lines(CSFE[2,,i],col=3)
  lines(CSFE[3,,i],col=4)
  abline(h=0,lty=2)
  axis(side = 1,at = iWindow,labels = iYear)
}

par(mfrow=c(2,2),lwd=4.8,cex.main=4.8,font.axis=2,cex.axis=2.4)
for (i in dimnames(CSFE)[[3]][c(11,12,16,17)]) {
  plot(CSFE[1,,i],type = "l",col=2,xaxt="n",main = i,xlab = '',ylab = '',
       xlim = c(17,115),ylim = c(min(CSFE[,,i],na.rm = TRUE),max(CSFE[,,i],na.rm = TRUE)))
  lines(CSFE[2,,i],col=3)
  lines(CSFE[3,,i],col=4)
  abline(h=0,lty=2)
  axis(side = 1,at = iWindow,labels = iYear)
}



