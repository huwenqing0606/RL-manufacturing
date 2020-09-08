import numpy as np
import tensorflow as tf
#import tensorflow_probability as tfp
import math

"""
implements gaussian policy gradient 
imcluding the update, action, etc
"""
class Gaussian_Policy_Gradient:
    def __init__(self,seed, action_dim):
        self.seed = seed
        self.theta_mu = None
        self.theta_sigma = None
        self.action_dim = action_dim
        self.action = tf.Variable([0.]*action_dim, name = "action")
        self._mu = tf.Variable([0.]*action_dim, name="mu")
        self._sigma = tf.Variable([1.]*action_dim, name="sigma")
       # self.dist = tfp.distributions.MultivariateNormalDiag(loc = self._mu, scale_diag = self._sigma)
        

    """
    computes mu as a linear function of the state
    input: state tensor of shape (state_dim,1)
    """
    def mu(self, state):
        #initialize weights if not yet done
        if (not tf.is_tensor(self.theta_mu)) or (not tf.is_tensor(self.theta_sigma)):
            self.init = tf.random_normal_initializer(seed=self.seed)
            state_dim = np.shape(state)[0]
            self.state_dim = state_dim
            self.currentState = tf.Variable([[0.]]*state_dim, name = "state")
            self.theta_mu = tf.Variable(self.init(shape= (self.action_dim,state_dim)),trainable = True, name = "theta_mu")
            self.theta_sigma = tf.Variable(self.init(shape= (self.action_dim,state_dim)),trainable = True, name = "theta_sigma")
        #print(tf.tensordot(self.theta_mu,state,axes=1))
        mu = tf.reshape(tf.tensordot(self.theta_mu,state,axes=1),(self.action_dim,))
        if not tf.is_tensor(self._mu):
            self._mu= tf.Variable(mu,name = "mu")
        else: self._mu.assign(mu)
        return(mu)
    """
    computes sigma as a matrix_exp('linear function') of the state
    input: state tensor of shape (state_dim,1)
    output: vector containing 
    """
    def sigma(self, state):
        #initialize weights if not yet done
        if (not tf.is_tensor(self.theta_mu)) or (not tf.is_tensor(self.theta_sigma)):
            self.init = tf.random_normal_initializer(seed=self.seed)
            state_dim = np.shape(state)[0]
            self.state_dim = state_dim
            self.currentState = tf.Variable([[0.]]*state_dim, name = "state")
            self.theta_mu = tf.Variable(self.init(shape= (self.action_dim,state_dim)),trainable = True, name = "theta_mu")
            self.theta_sigma = tf.Variable(self.init(shape= (self.action_dim,state_dim)),trainable = True, name = "theta_sigma")
        sigma_ln = tf.reshape(tf.tensordot(self.theta_sigma,state,axes=1),(self.action_dim,))
        sigma = tf.exp(sigma_ln)#tf.linalg.expm(sigma_ln)
        if not tf.is_tensor(self._mu):
            self._sigma = tf.Variable(sigma, name = "sigma")
        else: self._sigma.assign(sigma)
        return(sigma)
    
    """
    determines an action to be taken
    """
    def takeAction(self, state):
        #compute parameters
        mu = self.mu(state)
        sigma = self.sigma(state)
        #sample the action
        #action = self.dist.sample()
        self.action.assign(tf.random.normal((self.action_dim,),mean = mu, stddev=sigma))
        #save state
        self.currentState.assign(state)
        return(self.action)

    """
    Computes the probability density for a given action
    and returns the gradients wrt [theta_mu, theta_sigma]
    state is set through latest takeAction call
    """        
    def policyAndGradients(self, action):
        # gradient does not compute yet... assign destroys the tape
        # gradient wrt mu and sigma -> chain rule
        # computation not figured out yet
        with tf.GradientTape(persistent= True) as grad:
            grad.watch([self.theta_mu, self.theta_sigma])
            out = self.density(action, self.currentState)
        dA_dTheta = grad.gradient(out,[self.theta_mu, self.theta_sigma])
        return out, dA_dTheta
    
    def density(self, action, state):
        mu = self.mu(state)
        sigma = self.sigma(state)
        density = 1/(tf.sqrt((2*np.pi)**self.action_dim)*tf.reduce_prod(sigma))*tf.exp(-0.5*tf.reduce_sum((action-mu)**2/(sigma**2)))
        return(density)

    """
    wrapper to obtain the derivatives of policy at the action given the state and parameters wrt the parameters
    d pi(action|state,theta)/d theta
    from policyAndGradients
    """
    def gaussAction_gradient_weights(self, action):
        _, grad = self.policyAndGradients(action)
        return grad

    def gaussian_policygradient(self, A_c_grad_weights, Q_grad_A_c):
        #output the deterministic policy gradient of the cost with respect to the theta#
        print("Policy gradient; Q_grad_A_c:",Q_grad_A_c)
        policygradient = [tf.linalg.tensordot(Q_grad_A_c,grad,axes=1) for grad in A_c_grad_weights]
        return policygradient

    def update_weights(self, action,A_c_grad_gaussAction, Q_grad_A_c,lr_theta):
        gaussAction_gradient_weights = self.gaussAction_gradient_weights(action)
        A_c_grad_weights = [A_c_grad_gaussAction @ grad for grad in gaussAction_gradient_weights]
        policygradient = self.gaussian_policygradient(A_c_grad_weights,Q_grad_A_c)

        self.theta_mu.assign(self.theta_mu-lr_theta*policygradient[0])
        self.theta_sigma.assign(self.theta_sigma-lr_theta*policygradient[1])                          
        return [self.theta_mu, self.theta_sigma]
