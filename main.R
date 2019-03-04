#Script reproduces results for universal algorithms for quantile regression

#load data
df = read.csv("data.csv", header = T)
#pick the target
file_name = "wind"  #(file_name = "solar")

if (file_name == "wind") {
  target_name = c("wind_onshore_generation_actual")
  features = c("windspeed_10m", "temperature")
  
} else {
  target_name = c("solar_generation_actual")
  features = c("radiation_direct_horizontal", 
               "radiation_diffuse_horizontal")
}

#pick training and test
train = subset(df, df$year == 2015)
test = subset(df, df$year == 2016)

#scaling features to [0, 1]
features_min = apply(train[, features], 2, min)
features_max = apply(train[, features], 2, max)
features_min_train = matrix(rep(features_min, dim(train)[1]), ncol = length(features), byrow = T)
features_max_train = matrix(rep(features_max, dim(train)[1]), ncol = length(features), byrow = T)
train_scaled = (train[, features] - features_min_train) / (features_max_train - features_min_train)
features_min_test = matrix(rep(features_min, dim(test)[1]), ncol = length(features), byrow = T)
features_max_test = matrix(rep(features_max, dim(test)[1]), ncol = length(features), byrow = T)
test_scaled = (test[, features] - features_min_test) / (features_max_test - features_min_test)

train_target_scaled = cbind(train[, target_name], train_scaled)
names(train_target_scaled)[1] = target_name

#create experts
quantiles_seq = seq(0.05, 0.95, 0.05)
N = length(quantiles_seq)
#Quantile Random Forests (QRF)
library(quantregForest)
qrf <- quantregForest(x=train_scaled, y=(train[[target_name]]))
qrf_quantiles  <- predict(qrf, test_scaled, what=quantiles_seq)

#Gradient Boosting Decision Trees (GBDT)
formula = as.formula(paste(target_name, paste(features, collapse=" + "), sep=" ~ "))
library(gbm)
gbm_quantiles = matrix(0, nrow = dim(test_scaled)[1], ncol = N)
for (i in 1:N) {
  gbm_model = gbm(formula, distribution = list(name = "quantile", alpha = quantiles_seq[i]),
      data = train_target_scaled)
  gbm_quantiles[,i]  <- predict(gbm_model,  test_scaled, n.trees = 100)
}

#Quantile Regression (QR)
library(quantreg)
rq <- rq(formula, data=train_target_scaled, tau = quantiles_seq)
rq_quantiles = predict.rq(rq, test_scaled)

calc_loss = function(outcomes, predictions, q) {
  #function calculates pinball loss for vectors outcomes
  #and predictions for quantile q
  losses = outcomes - predictions
  for (j in 1:length(losses)) {
    losses[j] = ifelse(losses[j] > 0, q*losses[j], (q-1)*losses[j])
  }
  return(losses)
}

#calculate losses of WAA, Average and experts
outcomes = test[[target_name]]
T = length(outcomes)
N_exp = 3   #number of experts
WAA_quantiles = matrix(0, nrow = T, ncol = N)
avg_quantiles = matrix(0, nrow = T, ncol = N)
losses_WAA = matrix(0, nrow = T, ncol = N)
Losses_WAA = matrix(0, nrow = T, ncol = N)
losses_avg = matrix(0, nrow = T, ncol = N)
Losses_avg = matrix(0, nrow = T, ncol = N)
losses_experts = array(0, dim = c(N_exp, T, N))
Losses_experts = array(0, dim = c(N_exp, T, N))
weights_norm = array(0, c(N, T, N_exp))
for (i in 1:N) {
  X = as.matrix(cbind(qrf_quantiles[,i], gbm_quantiles[,i], rq_quantiles[,i]))
  WAA_quantiles[,i] = WAA(outcomes, X, q = quantiles_seq[i], C = 0.01)$gamma
  weights_norm[i, ,] = WAA(outcomes, X, q = quantiles_seq[i], C = 0.01)$weights_norm
  avg_quantiles[, i] = apply(X, 1, mean)
  losses_WAA[,i] = calc_loss(outcomes, WAA_quantiles[,i], quantiles_seq[i])
  Losses_WAA[,i] = cumsum(losses_WAA[,i])
  losses_avg[,i] = calc_loss(outcomes, avg_quantiles[,i], quantiles_seq[i])
  Losses_avg[,i] = cumsum(losses_avg[,i])
  for (j in 1:N_exp)  {
    losses_experts[j,,i] = calc_loss(outcomes, X[,j ], quantiles_seq[i])
    Losses_experts[j,,i] = cumsum(losses_experts[j,,i])
  }
 }

q = 0.25   #pick quantile to produce graphs
ind = which(quantiles_seq == q)
losses_aggr = data.frame(cbind(losses_experts[1,,ind], losses_experts[2,,ind], losses_experts[2,,ind],
                               losses_avg[,ind], losses_WAA[,ind]))
names(losses_aggr) = c("QRF", "GBDT", "QR", "Average", "WAA")
losses_aggr$month = test$month
monthly_losses = aggregate(. ~ month, losses_aggr, sum)
library(xtable)
xtable(monthly_losses/1000, digits=c(0, 1, 1, 1, 1,1, 1))
round(apply(monthly_losses, 2, sum)/1000, 1)

total_losses = cbind(Losses_experts[1, T, ], Losses_experts[2, T, ], Losses_experts[3, T, ],
                     Losses_avg[T, ], Losses_WAA[T, ])

q_sel = c(5,10,15,19)
total_losses1 = data.frame(total_losses[q_sel,])
xtable(total_losses1/1000, digits= c(1,1,1,1,1,1))

counts <- matrix(0, nrow = length(q_sel)*5, ncol = N_exp)
counts = data.frame(counts)
names(counts) = c("loss", "method", "quantile")
counts$loss = as.vector(as.matrix(total_losses1))
counts$method = c(rep("QRF", length(q_sel)), rep("GBDT", length(q_sel)), rep("QR", length(q_sel)), rep("Average", length(q_sel)), rep("WAA", length(q_sel)))
counts$quantile = as.factor(c(rep(c(0.25, 0.5, 0.75, 0.95), 5)))

library(ggplot2)
pdf(paste("total_losses_", file_name, ".pdf", sep=""), height = 8.5, width = 8.5, paper = "special")
ggplot(counts, aes(fill=method, y=loss, x=quantile)) + 
  geom_bar(stat="identity", position=position_dodge())+
  scale_fill_brewer(palette="Paired")+
  theme_minimal()+
  theme(text = element_text(size=35), plot.title = element_text(hjust = 0.5))
dev.off()

library(reshape2)
melted_cormat <- melt(weights_norm[ind,,])
head(melted_cormat)
name_experts = c("QRF", "GBDT", "QR")
melted_cormat$experts = name_experts[melted_cormat$Var2]
melted_cormat$Time = melted_cormat$Var1
melted_cormat$weights = melted_cormat$value
pdf(paste("weights_", file_name, 100*q, ".pdf", sep=""), height = 8.5, width = 8.5, paper = "special")
ggplot(data = melted_cormat, aes(x=Time, y=experts, fill=weights)) +   geom_tile() +
  scale_fill_distiller(direction = 1, limit = c(0, max(weights_norm[ind,,])))+
  theme(text = element_text(size=30), plot.title = element_text(hjust = 0.5))
dev.off()

#calculate predictions for WAAQR
selected_quantiles = c(0.25, 0.5, 0.75)
X_test = as.matrix(test_scaled)
gamma = matrix(0, nrow = T, ncol = length(selected_quantiles))
loss_WAAQR = matrix(0, nrow = T, ncol = length(selected_quantiles))
Loss_WAAQR = matrix(0, nrow = T, ncol = length(selected_quantiles))
for (s in 1:length(selected_quantiles)) {
  gamma[, s] = WAAQR(outcomes, X_test, q = selected_quantiles[s], M = 1500, M0 = 300, a = 0.1, sigma = 3)$gamma
  loss_WAAQR[, s] = calc_loss(outcomes, gamma[, s], selected_quantiles[s])
  Loss_WAAQR[, s] = cumsum(loss_WAAQR[, s])
}

#predictions of QR trained on training data set
library(quantreg)
qr1 <- rq(formula, data=train_target_scaled, tau = selected_quantiles)
gamma_rq = predict.rq(qr1, test_scaled)
loss_rq = matrix(0, nrow = T, ncol = length(selected_quantiles))
Loss_rq = matrix(0, nrow = T, ncol = length(selected_quantiles))
for (s in 1:length(selected_quantiles)) {
  loss_rq[, s] = calc_loss(outcomes, gamma_rq[, s], selected_quantiles[s])
  Loss_rq[, s] = cumsum(loss_rq[, s])
}

for (s in 1:length(selected_quantiles)) {
  pdf(paste(file_name, 25*s, "_Loss_diff.pdf", sep=""), height = 8.5, width = 8.5, paper = "special")
  plot(Loss_rq[,s]-Loss_WAAQR[,s], lwd=3,cex=2,cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2, type = "l", xlab = "Time", ylab = "", main= "")
  lines(rep(0, T), lwd=3,cex=2,cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)
  dev.off()
}