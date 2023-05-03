# Implementation of Non-centered pCN within Gibbs Sampler
# Algorithm 4 in Chen, Dunlop, Papaspilopoulos, Stuart arXiv:1803.03344
# for an alpha-regular tau-scaled Laplace prior in the white noise model
# with fixed alpha and tau chosen via hierarchical Bayes,
# This code reproduces simulations presented in Section 4 of
# Agapiou and Savva 2022 arXiv:2209.06045
#---------------------------------------------------

rm(list=ls())
graphics.off()

# Define the truth on spectral and spatial domain
# and the observed data X
#------------------------------------------------------
set.seed(1)

n=200 # noise precision
L=200 # truncation point of the expansion

l_vec=1:L #coefficient index
j_vec=l_vec #spatial domain
truth_coefs=sin(10*l_vec)*(l_vec^(-2.25)) 
basis_mat=outer(j_vec,l_vec,function(j_vec,l_vec) sqrt(2)*sin(l_vec*pi*j_vec/L))
truth=basis_mat%*%truth_coefs #Sobolev regularity beta=1.75
# Truth taken from Section 3 of Szabo, van der Vaart and van Zanten 2015

X_coefs=truth_coefs+rnorm(L)/sqrt(n) # generate X coefficients
X=basis_mat%*%X_coefs # observation in spatial domain


# Define the prior and the orthogonal transformation required in pCN
#---------------------------------------------------------------
alpha=1.75 # prior regularity
scale_vec=l_vec^(-1/2-alpha) #prior scaling gamma_ell

a0=1 # Invere-Gamma hyper-prior shape
b0=1 # Inverse-Gamma hyper-prior scale
lt=n^(-1/(3+2*alpha)) #left truncation of Inverse-Gamma hyper prior see (2.3) in arXiv:2209.06045

# White noise representation for Laplace prior (p=1)
T=function(xi,tau){
  return(tau*scale_vec*sign(xi)*(-log(2-2*pnorm(abs(xi))))) 
} 
# write theta drawn from Laplace with scaling tau 
# as T(xi,tau) where xi is white Gaussian

# Metropolis within Gibbs sampler initialization and parameters
#--------------------------------------------------------------
I=40000 #number of posterior samples
post_draws_tau=rep(NA,I) #empty vector to store the tau chain
post_draws_laplace=matrix(NA,nrow=I,ncol=L) #empty matrix to store the theta chain

tau_old=1 # initial value of parameter tau
old=rnorm(L) # initial value of xi white noise
post_draws_tau[1]=tau_old
post_draws_laplace[1,]=T(old,tau_old)

step_size=0.2 # whitened pCN stepsize for xi (should be tuned to give acc prob~0.3)
step_compl=sqrt(1-step_size^2)

acc=0 #acceptance probability counter for v-chain
tau_acc=0 # acceptance probability counter for tau-chain


#Metropolis within Gibbs sampler for sampling Pi(theta, tau|X)
#see Algorithm 4 in "Dimension-robust MCMC in Bayesian inverse problems" by Chen et al.
#--------------------------------------------------------------------------------------
for(i in 2:I){
  
  #---Update xi using pCN---#
  # 
  eta=rnorm(L)
  proposal=step_compl*old+step_size*eta 
  
  laccept_prob=n*(X_coefs%*%(T(proposal,tau_old)-T(old,tau_old)))+(n/2)*(sum(T(old,tau_old)^2)-sum(T(proposal,tau_old)^2))
  
  #Accept/reject xi
  if(log(runif(1))<=laccept_prob){
    old=proposal
    acc=acc+1 
  }


  #---Update tau (using independence sampler)---#
  #---Use Gaussian likelihood of tau as proposal, obtained by completing the square-#
  v=T(old,1) # theta=v*tau
  tau_var=1/(n*sum(v^2)) #variance of the proposal of tau
  tau_mean=n*tau_var*(X_coefs%*%v) #mean of the proposal of tau
  tau_proposal=rnorm(1,mean=tau_mean,sd=sqrt(tau_var))
  
    if(tau_proposal>lt){
      laccept_prob=(a0+1)*(log(tau_old)-log(tau_proposal))+b0*((1/tau_old)-(1/tau_proposal)) #log of acceptance probability for tau
    
      #Accept/reject
      if(log(runif(1))<=laccept_prob){ 
        tau_old=tau_proposal
        tau_acc=tau_acc+1 
      }
    }
post_draws_laplace[i,]=T(old,tau_old) # transform to get posterior theta-draws
post_draws_tau[i+1]=tau_old 
}

# Acceptance probabilities
print(acc/I) #for theta
print(tau_acc/I) #for tau

#Plots and results
#---------------------------------------
#tau
#-----
ts.plot(post_draws_tau[(I/2):I],ylab='tau') #time series plot of posterior draws of tau

#theta
#-----
#Posterior mean
postm_coefs=apply(post_draws_laplace[(I/2+1):I,],2,mean) #use I/2 iterations as warm-up
postm_laplace=basis_mat%*%postm_coefs

#Plot: (1)posterior draws (black line), (2)posterior mean (red line)
plot(j_vec/L,truth,type='l',xlab='',ylab=expression(paste(theta, "(t)")), ylim=c(-.975,0), lwd=2.5) #true curve
lines(j_vec/L,postm_laplace, col='red', lwd=2.5) #hierarchical Bayes Laplace posterior mean


#Calculate the mean squared error of the truth
sq_error=mean((truth-postm_laplace)^2)

