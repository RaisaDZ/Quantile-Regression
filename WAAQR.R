WAAQR = function(outcomes, X, q, M, M0, a, sigma)  {
  #Inputs: outcomes - target vector,
  #        X - matrix of features where rows are observations and columns are features
  #        q - quantile
  #        M - maximum number of iteration
  #        M0 - the length of "burn-in" period
  #        a - regualrization parameter
  #        sigma - standard deviation
  
  #Outputs: gamma - calculated predictions for WAAQR,
  #         theta - sampling parameters at each iteration,
  #         lik - log-likelihood of parameters at each iteration
  
  X <- cbind(rep(1, dim(X)[1]), X)  #add bias
  T <- dim(X)[1]   #time
  n <- dim(X)[2]   #dimension
  
  omega <- function(a, q, theta, ksi, outcomes) {
    #likelihood of parameters theta
    T = length(outcomes)
    losses = outcomes - ksi
    for (j in 1:T) {
      losses[j] = ifelse(losses[j] > 0, q*losses[j], (q-1)*losses[j])
    }
    
    w <- - a * sum(abs(theta)) - 1/sqrt(T) * sum(losses)
    return(w)
  }
  
  omega0 <- function(a, theta) {
    #likelihood of parameters theta at time t = 0
    w <- - a *  sum(abs(theta))
    return(w)
  }
  
  gamma <- matrix(0, T, ncol = 1)
  theta <- array(0, dim = c(T, M, n))
  lik = matrix(0, nrow = T, ncol = M)
  for (t in 1:T) {
    #initial estimates of theta
    if (t > 1) {
      theta[t, 1, ] <- theta[t-1, M,  ] 
    }
    ksi_tot = 0
    for (m in 2:M)  {
      theta_old <- matrix(theta[t, m-1, ], ncol = n)  #theta from previous step m-1
      theta_new <- theta_old + matrix(rnorm(n, 0, sigma^2), nrow = 1, ncol = n) #sample new params
      if (t > 1) {
        ksi_old <- X[1:(t-1), ] %*% t(theta_old)  #old probs
        ksi_new <- X[1:(t-1), ] %*% t(theta_new)  #new probs
        omega_old = omega(a, q, theta_old, ksi_old, outcomes[1:(t-1)])
        omega_new = omega(a, q, theta_new, ksi_new, outcomes[1:(t-1)])
      } else {
        omega_old = omega0(a, theta_old)
        omega_new = omega0(a, theta_new)
      }
      alpha0 <- exp(omega_new - omega_old)
      alpha <- min(1, alpha0)
      rand <- runif(1, 0, 1)  #flip a coin
      if (alpha >= rand)  {  #accept new params
        theta[t, m,  ] <- theta_new
        lik[t, m] = omega_new 
      } else {              #keep old params
        theta[t, m,  ] <- theta[t, m-1,  ]
        lik[t, m] = omega_old
      }
      ksi <- X[t, ] %*% theta[t, m,  ]
      #burn-in
      if (m > M0) {
        ksi_tot <- ksi_tot + ksi
      }
    }
    gamma[t, ] <- ksi_tot / (M-M0)
  }
  return(list(gamma = gamma, theta = theta, lik = lik[, 2:M]))
}
