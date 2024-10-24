""" CS5340 Lab 3: Hidden Markov Models
See accompanying PDF for instructions.

Name: Liu Yichao
Email: yichao.liu@u.nus.edu
Student ID: A0304386A
"""

import numpy as np
import scipy.stats
from scipy.special import softmax
from sklearn.cluster import KMeans


def initialize(n_states, x):
    """Initializes starting value for initial state distribution pi
    and state transition matrix A.

    A and pi are initialized with random starting values which satisfies the
    summation and non-negativity constraints.
    """
    seed = 5340
    np.random.seed(seed)

    pi = np.random.random(n_states)
    A = np.random.random([n_states, n_states])

    # We use softmax to satisify the summation constraints. Since the random
    # values are small and similar in magnitude, the resulting values are close
    # to a uniform distribution with small jitter
    pi = softmax(pi)
    A = softmax(A, axis=-1)

    # Gaussian Observation model parameters
    # We use k-means clustering to initalize the parameters.
    x_cat = np.concatenate(x, axis=0)
    kmeans = KMeans(n_clusters=int(n_states), random_state=seed).fit(x_cat[:, None])
    mu = kmeans.cluster_centers_[:, 0]
    std = np.array([np.std(x_cat[kmeans.labels_ == l]) for l in range(n_states)])
    phi = {'mu': mu, 'sigma': std}

    return pi, A, phi


"""E-step"""
def e_step(x_list, pi, A, phi):
    """ E-step: Compute posterior distribution of the latent variables,
    p(Z|X, theta_old). Specifically, we compute
      1) gamma(z_n): Marginal posterior distribution, and
      2) xi(z_n-1, z_n): Joint posterior distribution of two successive
         latent states

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        pi (np.ndarray): Current estimated Initial state distribution (K,)
        A (np.ndarray): Current estimated Transition matrix (K, K)
        phi (Dict[np.ndarray]): Current estimated gaussian parameters

    Returns:
        gamma_list (List[np.ndarray]), xi_list (List[np.ndarray])
    """
    n_states = pi.shape[0]
    gamma_list = [np.zeros([len(x), n_states]) for x in x_list]
    xi_list = [np.zeros([len(x)-1, n_states, n_states]) for x in x_list]

    """ YOUR CODE HERE
    Use the forward-backward procedure on each input sequence to populate 
    "gamma_list" and "xi_list" with the correct values.
    Be sure to use the scaling factor for numerical stability.
    """
    N = len(x_list[0]) # seq length
    obs_times = len(x_list) # number of sequences
    # ########################## old parameters ###################################
    # p(x_n|z_nk = 1) ~ N(mu[k], sigma[k])
    mu, sigma = phi['mu'], phi['sigma'] 

    # #################### normalized alpha and beta ##############################
    # for a single sequence, we have:
    # alpha(n) = p(x_1, ..., x_n, z_n) \in R^{n_states}
    # beta(n) = p(x_{n+1}, ..., x_N | z_n) \in R^{n_states}
    # alpha_hat(n) = alpha(n) / p(x_1, ..., x_n) = p(z_n | x_1, ..., x_n) 
    # beta_hat(n) = beta(n) / p(x_{n+1}, ..., x_N | x_1, ..., x_n)
    alpha_hat = np.zeros((obs_times, N, n_states))
    beta_hat = np.zeros((obs_times, N, n_states))

    # ######################## scaling factors c ##################################
    # c_n = p(x_n | x_1, ..., x_{n-1}) = (p(x_1, ..., x_n) / (p(x_1, ..., x_{n-1}))
    # c_n \in R, is a scalar
    c = np.zeros((obs_times, N))

    # ##################### calculate emission probability ########################
    # emission[i, n, k] = p(x_n^(i) | z_nk = 1) = N(x_n^(i) | mu[k], sigma[k])
    # precompute emission probability for each sequence to save computation
    emission = np.zeros((obs_times, N, n_states))
    for i in range(obs_times):
        for n in range(N):
            for k in range(n_states):
                emission[i, n, k] = scipy.stats.norm.pdf(x_list[i][n], mu[k], sigma[k])

    # ################ initialize alpha_hat(1), c(1) ##############################
    for i in range(obs_times):
        # alpha(1) = p(x_1, z_1) = p(x_1 | z_1) p(z_1)
        # c(1) = \sum_{z_1} alpha(1)
        # alpha_hat(1) = alpha(1) / c(1)
        for k in range(n_states):
            alpha_hat[i, 0, k] = pi[k] * emission[i, 0, k]
        c[i, 0] = np.sum(alpha_hat[i, 0])
        alpha_hat[i, 0] /= c[i, 0]

    # #################### initialize beta_hat(N) #################################
    for i in range(obs_times):
        # beta(N) = beta_hat(N) = \mathbf{1}
        for k in range(n_states):
            beta_hat[i, N-1, k] = 1

    # ################### forward pass to compute alpha_hat, c ####################
    for i in range(obs_times):
        # rewrite the recursion: 
        #   c_n * alpha_hat(n) 
        # = p(x_n | z_n) \sum_{z_{n-1}} alpha_hat(n-1) p(z_n | z_{n-1})
        # := alpha_tilde(n)
        # we use: c_n = \sum_{z_n} alpha_tilde(n) 
        for n in range(1, N):
            # for memory efficiency we use alpha_tilde_n to store the intermediate result
            alpha_tilde_n = np.zeros((n_states)) 
            for k in range(n_states): # z_{nk}^(i) = 1
                for j in range(n_states): # z_{(n-1)j}^(i) = 1
                    alpha_tilde_n[k] += alpha_hat[i, n-1, j] * A[j, k]
            for k in range(n_states): # z_nk^(i) = 1
                # multiply by p(x_n^(i) | z_nk^(i) = 1)
                alpha_tilde_n[k] *= emission[i, n, k]
            c[i, n] = np.sum(alpha_tilde_n)
            alpha_hat[i, n] = alpha_tilde_n / c[i, n]
    
    # ###################### backward pass to compute beta_hat ####################
    for i in range(obs_times):
        # rewrite the recursion:
        #   c_{n+1} beta_hat(n) 
        # = \sum_{z_{n+1}} beta_hat(n+1) p(x_{n+1} | z_{n+1}) p(z_{n+1} | z_n)
        # := beta_tilde(n)
        for n in range(N-2, -1, -1):
            beta_tilda_n = np.zeros(n_states) # similar to alpha_tilde_n
            for j in range(n_states): # z_nj = 1
                for k in range(n_states): # z_(n+1)k = 1
                    # emission = p(x_{n+1}^(i) | z_{n+1}k^(i) = 1)
                    beta_tilda_n[j] += beta_hat[i, n+1, k] * emission[i, n+1, k] * A[j, k]
            beta_hat[i, n] = beta_tilda_n / c[i, n+1]

    # ###################### compute gamma and xi ###############################
    for i in range(obs_times):
        # gamma(n) = alpha_hat(n) * beta_hat(n)
        for n in range(N):
            gamma_list[i][n] = alpha_hat[i, n] * beta_hat[i, n]
        # xi(z_{n-1}, z_n) = 
        #   1/c_n * alpha_hat(n-1) * p(x_n | z_n) * p(z_n | z_{n-1}) * beta_hat(n)
        for n in range(1, N):
            for j in range(n_states): # z_{n-1}j^(i) = 1
                for k in range(n_states): # z_nk^(i) = 1
                    # emission = p(x_n^(i) | z_nk^(i) = 1)
                    xi_list[i][n-1, j, k] = \
                        alpha_hat[i, n-1, j] * beta_hat[i, n, k] * emission[i, n, k] * A[j, k] / c[i, n]

    return gamma_list, xi_list


"""M-step"""
def m_step(x_list, gamma_list, xi_list):
    """M-step of Baum-Welch: Maximises the log complete-data likelihood for
    Gaussian HMMs.
    
    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        gamma_list (List[np.ndarray]): Marginal posterior distribution
        xi_list (List[np.ndarray]): Joint posterior distribution of two
          successive latent states

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.
    """

    n_states = gamma_list[0].shape[1]
    pi = np.zeros([n_states])
    A = np.zeros([n_states, n_states])
    phi = {'mu': np.zeros(n_states),
           'sigma': np.zeros(n_states)}

    """ YOUR CODE HERE
    Compute the complete-data maximum likelihood estimates for pi, A, phi.
    """

    return pi, A, phi


"""Putting them together"""
def fit_hmm(x_list, n_states):
    """Fit HMM parameters to observed data using Baum-Welch algorithm

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        n_states (int): Number of latent states to use for the estimation.

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Time-independent stochastic transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.

    """

    # We randomly initialize pi and A, and use k-means to initialize phi
    # Please do NOT change the initialization function since that will affect
    # grading
    pi, A, phi = initialize(n_states, x_list)

    """ YOUR CODE HERE
     Populate the values of pi, A, phi with the correct values. 
    """

    return pi, A, phi
