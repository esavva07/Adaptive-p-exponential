# Implentation of non-centered Gibbs Sampler
# for an alpha-regular Gaussian prior (no scaling)
# with hyper-prior on alpha.
# This code reproduces simulations presented in Section 4 of
# Agapiou and Savva 2022 arXiv:2209.06045
#----------------------------

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

# Gibbs sampler initialization and parameters
# Write theta=T(z,alpha) where z is white Gaussian
#--------------------------------------------------------------
I=40000 #number of posterior samples
post_draws_gauss=matrix(NA,nrow=I,ncol=L) #empty matrix to store the theta-chain
post_draws_alpha=rep(NA,I) #empty vector to store the alpha-chain

alpha_old=1
post_draws_alpha[1]=alpha_old

step_size=0.3 #step size for alpha-chain (should be tuned to give acc prob ~0.3)
acc=0 #acceptance probability counter for alpha-chain


# Gibbs sampler for sampling Pi(theta, alpha|X)
for(i in 1:I){
  #---Update v using expressions for Gaussian conditional posterior---#
  postv_gauss_coefs=l_vec^(1+2*alpha_old)/(n+(l_vec^(1+2*alpha_old))) #coefs of posterior variance for z
  postm_gauss_coefs=n*(l_vec^(1/2+alpha_old)/(n+(l_vec^(1+2*alpha_old))))*X_coefs #coefs of posterior mean for z
  poststd_gauss_coefs=sqrt(postv_gauss_coefs) #coefs of posterior sd for z
  
  z_coefs=postm_gauss_coefs+poststd_gauss_coefs*rnorm(L)
  
  #alpha update(RWM)
  alpha_proposal=alpha_old+step_size*rnorm(1)
  if (alpha_proposal>=ubar & alpha_proposal<=obar) {
    scale_vec=l_vec^(-1/2-alpha_old) #prior scaling gamma_ell
    sq_scale_vec=l_vec^(-1-2*alpha_old)
    scale_vec_prop=l_vec^(-1/2-alpha_proposal) #prior scaling gamma_ell
    sq_scale_vec_prop=l_vec^(-1-2*alpha_proposal)
    aux_1_old=sum(sq_scale_vec*(z_coefs^2))
    aux_1_proposal= sum(sq_scale_vec_prop*(z_coefs^2))
    aux_2_old=sum(X_coefs*scale_vec*z_coefs)
    aux_2_proposal=sum(X_coefs*scale_vec_prop*z_coefs)
    laccept_prob=(alpha_old-alpha_proposal)-(n/2)*(aux_1_proposal-aux_1_old)+n*(aux_2_proposal-aux_2_old) #log of acceptance probability
    if(log(runif(1)) <= laccept_prob){
      alpha_old=alpha_proposal
      acc=acc+1
    }
    post_draws_alpha[i+1]=alpha_old
  }
  theta_coefs=l_vec^(-1/2-alpha_old)*z_coefs
  post_draws_gauss[i,]=theta_coefs
}  

# Acceptance probability for alpha chain
print(acc/I) 

#Plots and results
#---------------------------------------
#alpha
#-----
ts.plot(post_draws_alpha[2:I],ylab='alpha') #time series plot of posterior draws of alpha

#theta
#-----
#Posterior mean
postm_coefs=apply(post_draws_gauss[(I/2+1):I,],2,mean) #use I/2 iterations as warm-up
postm_gauss=basis_mat%*%postm_coefs

#Plot: (1)posterior draws (black line), (2)posterior mean (red line)
plot(j_vec/L,truth,type='l',xlab='t',ylab=expression(paste(theta, "(t)")), ylim=c(0,1.6), lwd=2.5) #true curve
lines(j_vec/L,postm_gauss,col='red',lwd=2.5) #hierarchical Bayes gauss posterior mean


#Calculate the mean squared error of the truth
sq_error=mean((truth-postm_gauss)^2)

