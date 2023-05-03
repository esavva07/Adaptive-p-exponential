# Implementation of Non-centered pCN within Gibbs Sampler
# Algorithm 4 in Chen, Dunlop, Papaspilopoulos, Stuart arXiv:1803.03344
# for an alpha-regular tau-scaled Laplace prior in the white noise model
# with fixed tau and alpha chosen via hierarchical Bayes,
# This code reproduces simulations presented in Section 4 of
# Agapiou and Savva 2022 arXiv:2209.06045
#---------------------------------------------------

rm(list=ls())
graphics.off()

# Define the truth on spectral and spatial domain
# and the observed data 
#------------------------------------------------------
set.seed(1)

n=10^3 #noise precision
L=2000 #truncation point of the expansion

l_vec=1:L #coefficient index
j_vec=l_vec #spatial domain
truth_coefs=sin(l_vec)*(l_vec^(-3/2)) 
basis_mat=outer(j_vec,l_vec,function(j_vec,l_vec) sqrt(2)*cos((l_vec-1/2)*pi*j_vec/L)) #orthonormal basis matrix from "Bayes procedures for adaptive inference in inverse problems for the white noise model" by Knapik et al., see section 3 
truth=basis_mat%*%truth_coefs #Sobolev regularity beta=1
#Truth taken from Section 3 of Knapik, Szabo, van der Vaart and van Zanten 2016

X_coefs=truth_coefs+rnorm(L)/sqrt(n) # generate X coefficients
X=basis_mat%*%X_coefs # observation in spatial domain


# Define the prior and the orthogonal transformation required in pCN
#---------------------------------------------------------------
ubar=0.5 #see Assumption (2.4) in arXiv:2209.06045
obar=100

# White noise representation for Laplace prior (p=1) 
T=function(xi,alpha){ 
  return(l_vec^(-1/2-alpha)*sign(xi)*(-log(2-2*pnorm(abs(xi))))) 
}
# write theta drawn from Laplace with regularity alpha 
# as T(xi,alpha) where xi is white Gaussian

# Metropolis within Gibbs sampler initialization and parameters
#--------------------------------------------------------------
I=40000 #number of posterior samples
post_draws_alpha=rep(NA,I) #empty vector to store the alpha-chain
post_draws_laplace=matrix(NA,nrow=I,ncol=L) #empty matrix to store the theta-chain

alpha_old=rexp(1) #hyper-prior on alpha is exponential, truncated to have support in [ubar,obar]
old=rnorm(L) # initial value of xi white noiseZ
post_draws_alpha[1]=alpha_old 
post_draws_laplace[1,]=T(old,alpha_old)

step_size=0.06 # whitened pCN stepsize for xi (should be tuned to give acc prob~0.3)
step_compl=sqrt(1-step_size^2) 
a_step_size=0.4 #step size for alpha proposal (should be tuned to give acc prob~0.3)

acc=0 #acceptance probability counter for v-chain
alpha_acc=0 #acceptance probability counter for alpha-chain

#Metropolis within Gibbs sampler for sampling Pi(theta, alpha|X)
#see Algorithm 4 in arXiv:1803.03344
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
for(i in 2:I){
  
  #---Update xi using pCN---#
  # 
  eta=rnorm(L)
  proposal=step_compl*old+step_size*eta 
  
  laccept_prob=n*(X_coefs%*%(T(proposal,alpha_old)-T(old,alpha_old)))+(n/2)*(sum(T(old,alpha_old)^2)-sum(T(proposal,alpha_old)^2))
  
  #Accept/reject xi
  if(log(runif(1))<=laccept_prob){ #U~uniform(0,1), if logU<=laccept_prob then old=proposal otherwise old=old
    old=proposal
    acc=acc+1 #count how many proposals of theta are accepted
  }
  
  #---Update alpha (using RWM method)---#
  alpha_proposal=alpha_old+a_step_size*rnorm(1)
    
  if(alpha_proposal>=ubar & alpha_proposal<=obar){
    aux_1=X_coefs%*%(T(old,alpha_proposal)-T(old,alpha_old))
    aux_2=sum(T(old,alpha_old)^2)-sum(T(old,alpha_proposal)^2)
    laccept_prob_a=(alpha_old-alpha_proposal)+(n/2)*(aux_2)+(n*aux_1) 
    
    #Accept/reject alpha
    if(log(runif(1))<=laccept_prob_a){ 
      alpha_old=alpha_proposal
      alpha_acc=alpha_acc+1 
    }
  }
  
post_draws_laplace[i,]=T(old,alpha_old) # transform to get posterior theta-draws 
post_draws_alpha[i+1]=alpha_old 
}


# Acceptance probabilities
print(acc/I) #for theta 
print(alpha_acc/I) #for alpha

#Plots and results
#---------------------------------------
#alpha
#-----
ts.plot(post_draws_alpha[2:I],ylab='alpha') #time series plot of posterior draws of alpha

#theta
#-----
#Posterior mean
postm_coefs=apply(post_draws_laplace[(I/2+1):I,],2,mean) #use I/2 iterations as warm-up
postm_laplace=basis_mat%*%postm_coefs

#Plot: (1)posterior draws (black line), (2)posterior mean (red line)
plot(j_vec/L,truth,type='l',xlab='t',ylab=expression(paste(theta, "(t)")), ylim=c(0,1.6), lwd=2.5) #true curve
lines(j_vec/L,postm_laplace,col='red',lwd=2.5) #hierarchical Bayes Laplace posterior mean


#Calculate the mean squared error of the truth
sq_error=mean((truth-postm_laplace)^2)

