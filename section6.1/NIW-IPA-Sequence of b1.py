import numpy as np
from scipy.linalg import det, inv
from scipy.special import multigammaln
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import time
import pyomo.environ as pyo
import copy
import csv

def generate_bivariate_instance(N,P,μ_data_generating,Σ_data_generating,b1_rho,b2max,b3max,precision):
    np.random.seed(N+P) # fix the pseudorandom seed, if desired
    Xp=np.random.multivariate_normal(mean=μ_data_generating,cov=Σ_data_generating,size=N)
    Xbar_data=np.mean(Xp,axis=0)    
    b1=np.minimum(N*P,round(b1_rho*N*P,0))
    b2=np.zeros(N,dtype=int)
    while np.sum(b2)==0:  # Ensure sum(b2) > 0
        b2 = np.minimum(P,np.random.randint(0,b2max+1,size=N))  
    b3=np.round(np.random.uniform(0,b3max,size=(N,P)),precision)
    return (Xp,Xbar_data,b1,b2,b3)

def posterior_normal_inverse_wishart(μ_0, κ_0, v_0, Λ_0, data):
    data=np.array(data,dtype=float)
    n = data.shape[0]  # vmber of observations
    x_bar = np.mean(data, axis=0) # Sample mean vector
    # Calculate posterior parameters
    κ_n = κ_0 + n
    μ_n = (κ_0 * μ_0 + n * x_bar) / κ_n
    v_n = v_0 + n
    S = np.zeros_like(Λ_0)
    for i in range(n):
      diff = (data[i,:] - x_bar)
      S = S + np.outer(diff, diff)
    Λ_n = Λ_0 + S + (κ_0 * n / κ_n) * np.outer(x_bar - μ_0, x_bar - μ_0)
    return μ_n, κ_n, v_n, Λ_n

def kl_normal_inverse_wishart(μ_p, κ_p, v_p, Λ_p, μ_q, κ_q, v_q, Λ_q):
    d = len(μ_p)

    # Mean and covariance of the normal component
    Sigma_p = Λ_p / (κ_p * (v_p - d - 1))
    Sigma_q = Λ_q / (κ_q * (v_q - d - 1))
    
    # KL divergence between the normal components
    μ_diff = μ_q - μ_p
    Sigma_q_inv = inv(Sigma_q)
    KL_normal = 0.5 * (
        np.log(det(Sigma_q) / det(Sigma_p))
        - d
        + np.trace(Sigma_q_inv @ Sigma_p)
        + μ_diff.T @ Sigma_q_inv @ μ_diff
    )

    # KL divergence between inverse Wishart components
    Λ_q_inv = inv(Λ_q)
    trace_term = np.trace(Λ_q_inv @ Λ_p)
    
    KL_inv_wishart = 0.5 * (
        - v_p * np.log(det(Λ_p))
        + v_p * np.log(det(Λ_q))
        + trace_term * v_q
        - d * v_q
        + 2 * (multigammaln(v_q / 2, d) - multigammaln(v_p / 2, d))
        + d * (v_p - v_q)
    )
    δI = KL_normal + KL_inv_wishart   
    return δI

def kl_normal_inverse_wishart_given_X(μ_prior,κ_prior,v_prior,Λ_prior,X,μ_tgt,κ_tgt,v_tgt,Λ_tgt):
    (μ_pert,κ_pert,v_pert,Λ_pert)=posterior_normal_inverse_wishart(μ_prior,κ_prior,v_prior,Λ_prior,X)
    δI=kl_normal_inverse_wishart(μ_tgt,κ_tgt,v_tgt,Λ_tgt,μ_pert,κ_pert,v_pert,Λ_pert)
    return δI

def create_list_from_integer(k):
    return list(range(1,k+1))

def list_to_dict(lst):
    return {i+1: item for i,item in enumerate(lst)}

def matrix_to_dict(matrix):
    N = len(matrix)
    P = len(matrix[0])
    result_dict = {}
    for n in range(N):
        for p in range(P):
            result_dict[(n+1,p+1)] = matrix[n][p]
    return result_dict

def matrix_of_matrices_to_dict(matrix,N,P):
    # Initialize an empty dictionary
    result_dict = {}
    # Iterate over the matrix dimensions to construct keys (n,p,k,l)
    for n in range(N):
        for p in range(P):
            for k in range(N):
                for l in range(P):
                    result_dict[(n+1,p+1,k+1,l+1)] = matrix[n][p][k][l]
    return result_dict

def plot_bivariate_normals(μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,N):
    # Create a grid of points for the plot
    x = np.linspace(-6,5,500)
    y = np.linspace(-3,7,500)
    X,Y = np.meshgrid(x,y)
    pos = np.dstack((X,Y))
    # Create the normal distributions with different means and covariance matrices
    rv1 = multivariate_normal(mean=μ_prior,cov=Σ_prior)
    rv2 = multivariate_normal(mean=μ_post,cov=Σ_post)
    rv3 = multivariate_normal(mean=μ_tgt,cov=Σ_tgt)
    # Plot contours for each distribution
    plt.figure(figsize=(8,6))
    plt.contour(X,Y,rv1.pdf(pos),levels=5,cmap='Blues')
    plt.contour(X,Y,rv2.pdf(pos),levels=5,cmap='Greens')
    plt.contour(X,Y,rv3.pdf(pos),levels=5,cmap='Reds')
    # Mark the mean points for each distribution
    plt.scatter(μ_prior[0],μ_prior[1],color='blue',zorder=5)
    plt.text(μ_prior[0] + 0.2,μ_prior[1],'μ_prior',color='blue')
    plt.scatter(μ_post[0],μ_post[1],color='green',zorder=5)
    plt.text(μ_post[0] + 0.2,μ_post[1],'μ_post',color='green')
    plt.scatter(μ_tgt[0],μ_tgt[1],color='red',zorder=5)
    plt.text(μ_tgt[0] + 0.2,μ_tgt[1],'μ_tgt',color='red')
    # Add labels and title
    plt.title('Prior, Posterior, and Target Distributions')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.grid(True)
    plt.savefig(f"C:/Users/brian/Documents/PA/Plots/NIW_IPA_N{N}_Unpoisoned.png", dpi=300, bbox_inches='tight')     
    plt.show()
    return

def plot_bivariate_normals_with_poisoning(μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,μ_poisoned,Σ_poisoned,title,N,b1):
    # Create a grid of points for the plot
    x = np.linspace(-6,5,500)
    y = np.linspace(-3,7,500)
    X,Y = np.meshgrid(x,y)
    pos = np.dstack((X,Y))
    # Create the normal distributions with different means and covariance matrices
    rv1 = multivariate_normal(mean=μ_prior,cov=Σ_prior)
    rv2 = multivariate_normal(mean=μ_post,cov=Σ_post)
    rv3 = multivariate_normal(mean=μ_tgt,cov=Σ_tgt)
    rv4 = multivariate_normal(mean=μ_poisoned,cov=Σ_poisoned)
    # Plot contours for each distribution
    plt.figure(figsize=(8,6))
    plt.contour(X,Y,rv1.pdf(pos),levels=5,cmap='Blues')
    plt.contour(X,Y,rv2.pdf(pos),levels=5,cmap='Greens')
    plt.contour(X,Y,rv3.pdf(pos),levels=5,cmap='Reds')
    plt.contour(X,Y,rv4.pdf(pos),levels=5,cmap='Purples')
    # Mark the mean points for each distribution
    plt.scatter(μ_prior[0],μ_prior[1],color='blue',zorder=5)
    plt.text(μ_prior[0] + 0.2,μ_prior[1],'μ_prior',color='blue')
    plt.scatter(μ_post[0],μ_post[1],color='green',zorder=5)
    plt.text(μ_post[0] + 0.2,μ_post[1],'μ_post',color='green')
    plt.scatter(μ_tgt[0],μ_tgt[1],color='red',zorder=5)
    plt.text(μ_tgt[0] + 0.2,μ_tgt[1],'μ_tgt',color='red')
    plt.scatter(μ_poisoned[0],μ_poisoned[1],color='purple',zorder=5)
    plt.text(μ_poisoned[0] + 0.2,μ_poisoned[1]-0.2,'μ_poisoned',color='purple')
    # Add labels and title
    plt.title(f"Prior, Posterior, Target, and {title} Poisoned Distributions")
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.grid(True)
    plt.savefig(f"C:/Users/brian/Documents/PA/Plots/NIW_EPA_N{N}_Poisoned_b1_{b1}.png", dpi=300, bbox_inches='tight')     
    plt.show()
    return

def plot_bivariate_normals_with_poisoningXpX(μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,μ_poisoned,Σ_poisoned,title,Xp,X,N,b1):
    # Ensure Xp and X are numpy arrays
    Xp = np.array(Xp)
    X = np.array(X)
    
    # Find the extents of the data and means
    all_points = np.vstack([Xp,X,μ_prior,μ_post,μ_tgt,μ_poisoned])
    x_min,y_min = np.min(all_points,axis=0) - 1  # Add some margin
    x_max,y_max = np.max(all_points,axis=0) + 1
    
    # Create a grid of points for the plot based on the data extents
    x = np.linspace(-6,5,500)
    y = np.linspace(-3,7,500)
    X_grid,Y_grid = np.meshgrid(x,y)
    pos = np.dstack((X_grid,Y_grid))
    
    # Create the normal distributions with different means and covariance matrices
    rv1 = multivariate_normal(mean=μ_prior,cov=Σ_prior)
    rv2 = multivariate_normal(mean=μ_post,cov=Σ_post)
    rv3 = multivariate_normal(mean=μ_tgt,cov=Σ_tgt)
    rv4 = multivariate_normal(mean=μ_poisoned,cov=Σ_poisoned)
    
    # Plot contours for each distribution
    plt.figure(figsize=(10,8))
    plt.contour(X_grid,Y_grid,rv1.pdf(pos),levels=5,cmap='Blues')
    plt.contour(X_grid,Y_grid,rv2.pdf(pos),levels=5,cmap='Greens')
    plt.contour(X_grid,Y_grid,rv3.pdf(pos),levels=5,cmap='Reds')
    plt.contour(X_grid,Y_grid,rv4.pdf(pos),levels=5,cmap='Purples')
    
    # Mark the mean points for each distribution
    plt.scatter(μ_prior[0],μ_prior[1],color='blue',zorder=5)
    plt.text(μ_prior[0] + 0.2,μ_prior[1],'μ_prior',color='blue')
    plt.scatter(μ_post[0],μ_post[1],color='green',zorder=5)
    plt.text(μ_post[0] + 0.2,μ_post[1],'μ_post',color='green')
    plt.scatter(μ_tgt[0],μ_tgt[1],color='red',zorder=5)
    plt.text(μ_tgt[0] + 0.2,μ_tgt[1],'μ_tgt',color='red')
    plt.scatter(μ_poisoned[0],μ_poisoned[1],color='purple',zorder=5)
    plt.text(μ_poisoned[0] + 0.2,μ_poisoned[1],'μ_poisoned',color='purple')
    
    # Plot the points in Xp and X without comparison
    for i in range(len(Xp)):
        # Plot the "Clean Data" (Xp) and "Attacked Data" (X) points
        plt.scatter(Xp[i,0],Xp[i,1],color='lightblue',edgecolor='black',label='Clean Data' if i == 0 else "",zorder=6)
        plt.scatter(X[i,0],X[i,1],color='red',edgecolor='black',label='Attacked Data' if i == 0 else "",zorder=7)
        
        # Check if the points are significantly different (based on the condition)
        if np.any(np.abs(Xp[i] - X[i]) > 0.01):  
            # Plot a gray arrow between the points if they are different
            plt.arrow(
                Xp[i,0],Xp[i,1],
                X[i,0] - Xp[i,0],X[i,1] - Xp[i,1],
                color='grey',head_width=0.1,length_includes_head=True,zorder=5
            )
    
    # Add labels, legend, and title
    plt.title(f"Prior, Posterior, Target, and {title} Poisoned Distributions")
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"C:/Users/brian/Documents/PA/Plots/NIW_IPA_N{N}_Poisoned_b1_{b1}.png", dpi=300, bbox_inches='tight')     
    plt.show()

def approximate_gradient(μ_prior,κ_prior,v_prior,Λ_prior,Xp,μ_tgt,κ_tgt,v_tgt,Λ_tgt,precision):
    X=np.array(Xp,dtype=float)
    N,P = X.shape
    gradient = np.zeros_like(X)
    
    (μ_post,κ_post,v_post,Λ_post)=posterior_normal_inverse_wishart(μ_prior,κ_prior,v_prior,Λ_prior,Xp)
    kl_post=kl_normal_inverse_wishart(μ_post,κ_post,v_post,Λ_post,μ_tgt,κ_tgt,v_tgt,Λ_tgt)
    epsilon = 1e-5 
    
    for n in range(N):
        for p in range(P):
            X_perturbed = X.copy()
            X_perturbed[n,p] += epsilon
            (μ_pert,κ_pert,v_pert,Λ_pert)=posterior_normal_inverse_wishart(μ_prior,κ_prior,v_prior,Λ_prior,X_perturbed)
            
            gradient[n,p] = (kl_normal_inverse_wishart(μ_pert,κ_pert,v_pert,Λ_pert,μ_tgt,κ_tgt,v_tgt,Λ_tgt)-kl_post)/epsilon # Finite difference approximation
    return np.round(gradient,precision)

def approximate_hessian(μ_prior, κ_prior, v_prior, Λ_prior, Xp, μ_tgt, κ_tgt, v_tgt, Λ_tgt, precision):
    X = np.array(Xp, dtype=float)
    N, P = X.shape
    H = np.zeros((N, P, N, P))  # Hessian matrix
    epsilon = 1e-5  # Small step size for vmerical stability

    # Compute KL divergence at the unperturbed Xp
    μ_post, κ_post, v_post, Λ_post = posterior_normal_inverse_wishart(μ_prior, κ_prior, v_prior, Λ_prior, X)
    kl_base = kl_normal_inverse_wishart(μ_post, κ_post, v_post, Λ_post, μ_tgt, κ_tgt, v_tgt, Λ_tgt)

    for n in range(N):
        for p in range(P):
            for k in range(N):
                for l in range(P):
                    # Perturb X in both directions
                    X_np_kl_plus = X.copy()
                    X_np_kl_mivs = X.copy()
                    X_np_kl_plus[n, p] += epsilon
                    X_np_kl_plus[k, l] += epsilon
                    X_np_kl_mivs[n, p] -= epsilon
                    X_np_kl_mivs[k, l] -= epsilon

                    X_np_plus = X.copy()
                    X_np_mivs = X.copy()
                    X_kl_plus = X.copy()
                    X_kl_mivs = X.copy()

                    X_np_plus[n, p] += epsilon
                    X_np_mivs[n, p] -= epsilon
                    X_kl_plus[k, l] += epsilon
                    X_kl_mivs[k, l] -= epsilon

                    # Compute perturbed KL divergences
                    μ_np_plus, κ_np_plus, v_np_plus, Λ_np_plus = posterior_normal_inverse_wishart(μ_prior, κ_prior, v_prior, Λ_prior, X_np_plus)
                    kl_np_plus = kl_normal_inverse_wishart(μ_np_plus, κ_np_plus, v_np_plus, Λ_np_plus, μ_tgt, κ_tgt, v_tgt, Λ_tgt)

                    μ_np_mivs, κ_np_mivs, v_np_mivs, Λ_np_mivs = posterior_normal_inverse_wishart(μ_prior, κ_prior, v_prior, Λ_prior, X_np_mivs)
                    kl_np_mivs = kl_normal_inverse_wishart(μ_np_mivs, κ_np_mivs, v_np_mivs, Λ_np_mivs, μ_tgt, κ_tgt, v_tgt, Λ_tgt)

                    μ_kl_plus, κ_kl_plus, v_kl_plus, Λ_kl_plus = posterior_normal_inverse_wishart(μ_prior, κ_prior, v_prior, Λ_prior, X_kl_plus)
                    kl_kl_plus = kl_normal_inverse_wishart(μ_kl_plus, κ_kl_plus, v_kl_plus, Λ_kl_plus, μ_tgt, κ_tgt, v_tgt, Λ_tgt)

                    μ_kl_mivs, κ_kl_mivs, v_kl_mivs, Λ_kl_mivs = posterior_normal_inverse_wishart(μ_prior, κ_prior, v_prior, Λ_prior, X_kl_mivs)
                    kl_kl_mivs = kl_normal_inverse_wishart(μ_kl_mivs, κ_kl_mivs, v_kl_mivs, Λ_kl_mivs, μ_tgt, κ_tgt, v_tgt, Λ_tgt)

                    μ_np_kl_plus, κ_np_kl_plus, v_np_kl_plus, Λ_np_kl_plus = posterior_normal_inverse_wishart(μ_prior, κ_prior, v_prior, Λ_prior, X_np_kl_plus)
                    kl_np_kl_plus = kl_normal_inverse_wishart(μ_np_kl_plus, κ_np_kl_plus, v_np_kl_plus, Λ_np_kl_plus, μ_tgt, κ_tgt, v_tgt, Λ_tgt)

                    μ_np_kl_mivs, κ_np_kl_mivs, v_np_kl_mivs, Λ_np_kl_mivs = posterior_normal_inverse_wishart(μ_prior, κ_prior, v_prior, Λ_prior, X_np_kl_mivs)
                    kl_np_kl_mivs = kl_normal_inverse_wishart(μ_np_kl_mivs, κ_np_kl_mivs, v_np_kl_mivs, Λ_np_kl_mivs, μ_tgt, κ_tgt, v_tgt, Λ_tgt)

                    # Compute second derivative using central finite differences
                    H[n, p, k, l] = (kl_np_kl_plus - kl_np_plus - kl_kl_plus + 2 * kl_base - kl_np_mivs - kl_kl_mivs + kl_np_kl_mivs) / (epsilon ** 2)

    return np.round(H, precision)

def solve_Q2L1_for_Q2L(zN,P,μ_prior,κ_prior,v_prior,Λ_prior,μ_data_generating,Σ_data_generating,Λ_data_generating,Xprime,Xest,Xbar_data,b1,b2,b3,μ_tgt,κ_tgt,v_tgt,Λ_tgt,MP_time_limit,solver_name):
    start_time = time.time() # Start the clock
    ### The next two lines construct a quadratic approximation based on the point Xest rather than Xp ###
    grad_list = approximate_gradient(μ_prior,κ_prior,v_prior,Λ_prior,Xest,μ_tgt,κ_tgt,v_tgt,Λ_tgt,precision)
    H_list = approximate_hessian(μ_prior,κ_prior,v_prior,Λ_prior,Xest,μ_tgt,κ_tgt,v_tgt,Λ_tgt,precision)
    # Create a Pyomo model
    model = pyo.ConcreteModel()
    # Sets
    model.N = pyo.Set(initialize=create_list_from_integer(N))
    model.P = pyo.Set(initialize=create_list_from_integer(P))
    # Parameters
    model.Xp=pyo.Param(model.N,model.P,initialize=matrix_to_dict(Xprime))
    model.Xest=pyo.Param(model.N,model.P,initialize=matrix_to_dict(Xest))
    model.grad=pyo.Param(model.N,model.P,initialize=matrix_to_dict(grad_list))
    model.H=pyo.Param(model.N,model.P,model.N,model.P,initialize=matrix_of_matrices_to_dict(H_list,N,P))
    model.b1=pyo.Param(initialize=b1)
    model.b2=pyo.Param(model.N,initialize=list_to_dict(b2))
    model.b3=pyo.Param(model.N,model.P,initialize=matrix_to_dict(b3))
    # Define BigM as a scalar parameter that is the largest of (1) possible magnitudes of X_np and (2) possible deviations from X'_np
    def compute_big_m(model):
        # Extract all elements from Xp and b3
        magnitudes = []
        for n in model.N:
            for p in model.P:
                magnitudes.append(abs(model.Xp[n,p]+model.b3[n,p]))
                magnitudes.append(abs(model.Xp[n,p]-model.b3[n,p]))
                magnitudes.append(abs(model.b3[n,p]))
        # Find the maxiμm magnitude
        return max(magnitudes)
    model.BigM = pyo.Param(initialize=compute_big_m)
    model.f=pyo.Param(initialize=kl_normal_inverse_wishart_given_X(μ_prior,κ_prior,v_prior,Λ_prior,Xest,μ_tgt,κ_tgt,v_tgt,Λ_tgt))
    # Decision Variables
    def initialize_X(model,n,p):
        return model.Xest[n,p]
    model.X=pyo.Var(model.N,model.P,domain=pyo.Reals,initialize=initialize_X)
    model.Ω=pyo.Var(model.N,model.P,domain=pyo.Binary,initialize=0) # 1 of |X(n,p)-X'(n,p)|>0 and 0 otherwise
    # Forμlate model
    # Objective function is a 2nd order approximation around Xest rather than X'
    model.objfnvalue = pyo.Objective(expr = model.f+sum(sum(model.grad[n,p]*(model.X[n,p]-model.Xest[n,p]) for n in model.N) for p in model.P)+(1/2)*sum(sum(sum(sum(model.H[n,p,k,l]*(model.X[n,p]-model.Xest[n,p])*(model.X[k,l]-model.Xest[k,l]) for n in model.N) for p in model.P) for k in model.N) for l in model.P),sense = pyo.minimize)
    model.LimitTotalChanges=pyo.Constraint(expr=sum(sum(model.Ω[n,p] for n in model.N) for p in model.P)<=model.b1)
    model.LimitChangeInEntryI=pyo.ConstraintList()
    for n in model.N:
        model.LimitChangeInEntryI.add(sum(model.Ω[n,p] for p in model.P)<=model.b2[n])
    model.LimitEachChange=pyo.ConstraintList()
    for n in model.N:
        for p in model.P:
            model.LimitEachChange.add(model.X[n,p]-model.Xp[n,p]<=model.b3[n,p])
            model.LimitEachChange.add(model.Xp[n,p]-model.X[n,p]<=model.b3[n,p])
    model.SigmaCalcs=pyo.ConstraintList()
    for n in model.N:
        for p in model.P:
            model.SigmaCalcs.add(-model.BigM*model.Ω[n,p]<=model.X[n,p]-model.Xp[n,p])
            model.SigmaCalcs.add(model.X[n,p]-model.Xp[n,p]<=model.BigM*model.Ω[n,p])

    # Impose upper and lower bounds on DVs and objective function to speed convergence by solver
    model.DVbounds=pyo.ConstraintList()
    for n in model.N:
        for p in model.P:
            model.DVbounds.add(model.X[n,p]<=model.Xp[n,p]+model.b3[n,p])
            model.DVbounds.add(model.X[n,p]>=model.Xp[n,p]-model.b3[n,p])
            model.DVbounds.add(model.Ω[n,p]<=1)
            model.DVbounds.add(model.Ω[n,p]>=0)

    try:
        if (solver_name=='CPLEX'):
            solver = pyo.SolverFactory('cplex')
            solver.options['optimalitytarget'] = 3  # Solves a non-convex Q2L1
            solver.options['timelimit'] = MP_time_limit
            solver.solve(model,tee=False)
        elif (solver_name=="SCIP"):
            solver=pyo.SolverFactory('scip', solver_io='nl')
            solver.options['limits/time']=MP_time_limit  # Given SCIP the maxiμm time allowable
            solver.solve(model,tee=False)
        else:
            pyo.SolverFactory('mindtpy').solve(model,time_limit=MP_time_limit,tee=False)
        attempt='success'
    except:
        attempt='fail'
    end_time = time.time() # Stop the clock
    elapsed_time = end_time - start_time # Calculate the elapsed time

    if (attempt=='success'):
        #print('success')
        solution={
            'X':[[float(pyo.value(model.X[n,p])) for p in model.P] for n in model.N],
            'Ω':[[int(pyo.value(model.Ω[n,p])) for p in model.P] for n in model.N],
            'q(X)':round(pyo.value(model.objfnvalue),precision),
            'time':round(elapsed_time,precision)
            }
    else:
        print('fail')
        solution={
            'X':[[float(pyo.value(model.Xest[n,p])) for p in model.P] for n in model.N],
            'Ω':[[int(0) for p in model.P] for n in model.N],
            'q(X)':999999,
            'time':round(elapsed_time,precision)
            }
    return solution

def solve_Q2L(N,P,μ_prior,κ_prior,v_prior,Λ_prior,μ_data_generating,Σ_data_generating,Λ_data_generating,Xp,Xbar_data,b1,b2,b3,μ_tgt,κ_tgt,v_tgt,Λ_tgt,MP_time_limit,solver_name):
    stored_solutions={}
    k_iter=1
    termination_check=SA_termination_tolerance+1
    Xest=Xp #construct quadratic approximation about the point Xest
    f_k=1 #temporary initialization to define f_k
    start_time = time.time() # Start the clock
    elapsed_time=0

    while (k_iter<=SA_max_iterations and elapsed_time<SA_max_time and termination_check>SA_termination_tolerance):
        solution_Q2L1=solve_Q2L1_for_Q2L(N,P,μ_prior,κ_prior,v_prior,Λ_prior,μ_data_generating,Σ_data_generating,Λ_data_generating,Xp,Xest,Xbar_data,b1,b2,b3,μ_tgt,κ_tgt,v_tgt,Λ_tgt,MP_time_limit,solver_name)
        stored_solutions[k_iter]=solution_Q2L1
        if k_iter==1:
            f_kmivs1=kl_normal_inverse_wishart_given_X(μ_prior,κ_prior,v_prior,Λ_prior,np.array(solution_Q2L1['X']),μ_tgt,κ_tgt,v_tgt,Λ_tgt)+SA_termination_tolerance+1 
        else:
            f_kmivs1=f_k
        f_k=kl_normal_inverse_wishart_given_X(μ_prior,κ_prior,v_prior,Λ_prior,np.array(solution_Q2L1['X']),μ_tgt,κ_tgt,v_tgt,Λ_tgt)
        termination_check=f_kmivs1-f_k
        if f_k<f_kmivs1: # only accept a better solution
            print(f"-Accepting a better solution on iteration {k_iter}")
            Xest=solution_Q2L1['X']
            solution=solution_Q2L1
        else:
            print(f"-Rejecting a worse solution on iteration {k_iter}")
        k_iter=k_iter+1
        end_time = time.time() # Stop the clock
        elapsed_time = end_time - start_time # Calculate the elapsed time
    # make sure the reported time is the cuμlative time,not the time of the last Q2L1 solution
    solution['time']=round(elapsed_time,precision)
    solution['iterations']=k_iter-1
    
    return (solution,stored_solutions)

def generate_target_distribution(μ_prior,κ_prior,v_prior,Λ_prior,N,P,μ_data_generating,Σ_data_generating,Λ_data_generating,Xp,Xbar_data,b1,b2,b3,μ_tgt,κ_tgt,v_tgt,Λ_tgt,MP_time_limit,SA_max_time,SA_max_iterations,SA_termination_tolerance):
   solvers={'CPLEX'}#,'SCIP','MindtPy'}
   ### Compute the Unpoisoned Bayesian Posterior Distribution & KL Divergences,and Create Plot ###
   ### Compute the Bayesian Posterior Distribution ###
   (μ_post,κ_post,v_post,Λ_post)=posterior_normal_inverse_wishart(μ_prior,κ_prior,v_prior,Λ_prior,Xp)

   δ_kl_ex=round(kl_normal_inverse_wishart(μ_post,κ_post,v_post,Λ_post,μ_tgt,κ_tgt,v_tgt,Λ_tgt),precision)
   δ_kl_in=round(kl_normal_inverse_wishart(μ_tgt,κ_tgt,v_tgt,Λ_tgt,μ_post,κ_post,v_post,Λ_post),precision)
   plot_bivariate_normals(μ_prior,Λ_prior/(κ_prior*(v_prior-3)),μ_post,Λ_post/(κ_post*(v_post-3)),μ_tgt,Λ_tgt/(κ_tgt*(v_tgt-3)),N)
       
   for solver_name in solvers:
       print('******************************')
       print(f"Q2L Results with {solver_name} to generate target distribution")
       (solution_Q2L,Q2L_trace)=solve_Q2L(N,P,μ_prior,κ_prior,v_prior,Λ_prior,μ_data_generating,Σ_data_generating,Λ_data_generating,Xp,Xbar_data,b1,b2,b3,μ_tgt,κ_tgt,v_tgt,Λ_tgt,MP_time_limit,solver_name)
       X=solution_Q2L['X']
       (μ_poisoned,κ_poisoned,v_poisoned,Λ_poisoned)=posterior_normal_inverse_wishart(μ_prior,κ_prior,v_prior,Λ_prior,X)
       δI_poisoned=round(kl_normal_inverse_wishart(μ_tgt,κ_tgt,v_tgt,Λ_tgt,μ_poisoned,κ_poisoned,v_poisoned,Λ_poisoned),precision)
       print(f"μ_poisoned={μ_poisoned}")
       print(f"δI(X)={δI_poisoned}")
       print(f"time={solution_Q2L['time']} sec")
       title='Q2L-'+solver_name[0]
       #plot_bivariate_normals_with_poisoning(μ_prior,Λ_prior/(κ_prior*(v_prior-3)),μ_post,Λ_post/(κ_post*(v_post-3)),μ_tgt,Λ_tgt/(κ_tgt*(v_tgt-3)),μ_poisoned,Λ_poisoned/(κ_poisoned*(v_poisoned-3)),title,N,b1)
       plot_bivariate_normals_with_poisoningXpX(μ_prior,Λ_prior/(κ_prior*(v_prior-3)),μ_post,Λ_post/(κ_post*(v_post-3)),μ_tgt,Λ_tgt/(κ_tgt*(v_tgt-3)),μ_poisoned,Λ_poisoned/(κ_poisoned*(v_poisoned-3)),title,Xp,X,N,b1)       
   return (μ_poisoned,κ_poisoned,v_poisoned,Λ_poisoned)

def run_methods(μ_prior,κ_prior,v_prior,Λ_prior,N,P,μ_data_generating,Σ_data_generating,Λ_data_generating,Xp,Xbar_data,b1,b2,b3,μ_tgt,κ_tgt,v_tgt,Λ_tgt,MP_time_limit,SA_max_time,SA_max_iterations,SA_termination_tolerance):
   solvers={'CPLEX'}#,'SCIP','MindtPy'}
   #print('Xp=',Xp)
   #print('******************************')
   ### Compute the Unpoisoned Bayesian Posterior Distribution & KL Divergences,and Create Plot ###
   ### Compute the Bayesian Posterior Distribution ###
   (μ_post,κ_post,v_post,Λ_post)=posterior_normal_inverse_wishart(μ_prior,κ_prior,v_prior,Λ_prior,Xp)

   δ_kl_ex=round(kl_normal_inverse_wishart(μ_post,κ_post,v_post,Λ_post,μ_tgt,κ_tgt,v_tgt,Λ_tgt),precision)
   δ_kl_in=round(kl_normal_inverse_wishart(μ_tgt,κ_tgt,v_tgt,Λ_tgt,μ_post,κ_post,v_post,Λ_post),precision)
   print('******************************')
   print(f"δE(X')={δ_kl_ex}; δI(X')={δ_kl_in}")
   print(f"b1={b1} out of {N*P} entries may be perμted")
   #print(f"b2={b2} out of {P} entries in row n are perμtable")
   #print(f"b3={b3}")
   if b1==0:
       plot_bivariate_normals(μ_prior,Λ_prior/(κ_prior*(v_prior-3)),μ_post,Λ_post/(κ_post*(v_post-3)),μ_tgt,Λ_tgt/(κ_tgt*(v_tgt-3)),N)
       
   for solver_name in solvers:
       #print('******************************')
       print(f"Q2L Results with {solver_name}")
       (solution_Q2L,Q2L_trace)=solve_Q2L(N,P,μ_prior,κ_prior,v_prior,Λ_prior,μ_data_generating,Σ_data_generating,Λ_data_generating,Xp,Xbar_data,b1,b2,b3,μ_tgt,κ_tgt,v_tgt,Λ_tgt,MP_time_limit,solver_name)
       X=solution_Q2L['X']
       (μ_poisoned,κ_poisoned,v_poisoned,Λ_poisoned)=posterior_normal_inverse_wishart(μ_prior,κ_prior,v_prior,Λ_prior,X)
       δI_poisoned=round(kl_normal_inverse_wishart(μ_tgt,κ_tgt,v_tgt,Λ_tgt,μ_poisoned,κ_poisoned,v_poisoned,Λ_poisoned),precision)
       #print(f"μ_poisoned={μ_poisoned}")
       print(f"δI(X)={δI_poisoned}")
       print(f"time={solution_Q2L['time']} sec")
       title='Q2L-'+solver_name[0]
       #plot_bivariate_normals_with_poisoning(μ_prior,Λ_prior/(κ_prior*(v_prior-3)),μ_post,Λ_post/(κ_post*(v_post-3)),μ_tgt,Λ_tgt/(κ_tgt*(v_tgt-3)),μ_poisoned,Λ_poisoned/(κ_poisoned*(v_poisoned-3)),title,N,b1)
       plot_bivariate_normals_with_poisoningXpX(μ_prior,Λ_prior/(κ_prior*(v_prior-3)),μ_post,Λ_post/(κ_post*(v_post-3)),μ_tgt,Λ_tgt/(κ_tgt*(v_tgt-3)),μ_poisoned,Λ_poisoned/(κ_poisoned*(v_poisoned-3)),title,Xp,X,N,b1)

   l2_norm = np.linalg.norm(μ_poisoned - μ_tgt)
   frobenius_norm = np.linalg.norm(Λ_post - Λ_tgt, 'fro')
   return (δI_poisoned,solution_Q2L['time'],l2_norm,frobenius_norm)

def plot_dictionary(data,y_axis_title,N,file_title):
    # Extract keys and values
    x = list(data.keys())
    y = list(data.values())
    # Create the plot
    plt.plot(x, y, marker='o')  # marker='o' adds dots at data points
    plt.xlabel(r'$\rho$')
    plt.ylabel(y_axis_title)
    plt.xlim(0, 1)  # Set x-axis range
    plt.ylim(bottom=0)    # Set y-axis lower bound
    plt.grid(True)
    plt.savefig(f"C:/Users/brian/Documents/PA/Plots/NIW_IPA_N{N}_{file_title}.png", dpi=300, bbox_inches='tight')     
    plt.show()
    return

def dump_dictionaries_to_file(dict1,dict2,dict3,dict4):
    # Combine dictionaries into a list of rows
    data = [dict1, dict2, dict3, dict4]
    # Get the headers from the keys of any dictionary (they are the same in all)
    headers = list(dict1.keys())
    
    # Write to a CSV file
    with open('/Users/brian/Documents/PA/Plots/NIW_IPA_output.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()  # Write the header row
        writer.writerows(data)  # Write the data rows
    print("CSV file 'output.csv' has been created.")
    return

# MAIN PROGRAM BEGINS HERE
# Specify user-determined parameters
precision=6 # vmber of decimal places for selected calculations

# Prior distribution parameters
μ_prior = np.array([3, 3]) #prior mean vector
κ_prior= 1 #prior precision
v_prior= 5 #prior degrees of freedom
Λ_prior= np.array([[3, -1], [-1, 2]]) #prior scale matrix

# Data generating distribution parameters
μ_data_generating=np.array([-4,1])
Σ_data_generating=np.array([[3,0],[0,3]])
Λ_data_generating=np.linalg.inv(Σ_data_generating)
N=10 # of independent observations, indexed on n
P=2 # of elements within an observation, indexed on p

# Poisoning-related parameters
μ_tgt = np.array([1,-1]) #prior mean vector
κ_tgt = 11 #prior precision
v_tgt = 15 #prior degrees of freedom
Λ_tgt = np.array([[20, 11], [11,24]]) #prior scale matrix

b1_rho=1 # approximate percentage of elements that can change
b2max=100 # when generating b2-values,imposes allows up to round(b2max*b1/K,0) or 1 change,whichever is larger
b3max=3000 # maxiμm magnitude of any change in an entry k; for now,we've assumed it's 5,for simplicity

# Solver related parameters
MP_time_limit=15 # time limit on all solvers for any math program
SA_max_time=60 # max time for Q1L process
SA_max_iterations=40
SA_termination_tolerance=0 # terminate based on lack of eps improvement in objective function value between iterations
'''
# Jury rig a target distribution
(Xp,Xbar_data,b1,b2,b3)=generate_bivariate_instance(N,P,μ_data_generating,Σ_data_generating,b1_rho,b2max,b3max,precision)
(μ_tgt,κ_tgt,v_tgt,Λ_tgt)=generate_target_distribution(μ_prior,κ_prior,v_prior,Λ_prior,N,P,μ_data_generating,Σ_data_generating,Λ_data_generating,Xp,Xbar_data,b1,b2,b3,μ_tgt,κ_tgt,v_tgt,Λ_tgt,MP_time_limit,SA_max_time,SA_max_iterations,SA_termination_tolerance)
print(f'μ_tgt={μ_tgt}')
print(f'κ_tgt={κ_tgt}')
print(f'v_tgt={v_tgt}')
print(f'Λ_tgt={Λ_tgt}')
'''
b1_rho_values=[0,0.05,0.1]#,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]

stored_b1_δI={}
stored_b1_time={}
stored_b1_L2_norm={}
stored_b1_Frobenius={}
for b1_rho_counter,b1_rho in enumerate(b1_rho_values, 1): 
    ### Generate the instance data ###
    (Xp,Xbar_data,b1,b2,b3)=generate_bivariate_instance(N,P,μ_data_generating,Σ_data_generating,b1_rho,b2max,b3max,precision)
    (b1_δI,b1_time,b1_l2_norm,b1_frobenius_norm)=run_methods(μ_prior,κ_prior,v_prior,Λ_prior,N,P,μ_data_generating,Σ_data_generating,Λ_data_generating,Xp,Xbar_data,b1,b2,b3,μ_tgt,κ_tgt,v_tgt,Λ_tgt,MP_time_limit,SA_max_time,SA_max_iterations,SA_termination_tolerance)
    stored_b1_δI[b1_rho]=b1_δI
    stored_b1_time[b1_rho]=b1_time
    stored_b1_L2_norm[b1_rho]=b1_l2_norm
    stored_b1_Frobenius[b1_rho]=b1_frobenius_norm    

#print(stored_b1_δI)
#print(stored_b1_time)
plot_dictionary(stored_b1_δI,r'$\delta_{I}$-values',N,'IPA_KL_dist')
plot_dictionary(stored_b1_time,'Required Computational Effort (sec)',N,'EPA_times')
plot_dictionary(stored_b1_L2_norm,r'L2 norm, $\mu_{pois}$ and $\mu_{tgt}$',N,'IPA_L2_norms')
plot_dictionary(stored_b1_Frobenius,r'Frobenius norm, $\Lambda_{pois}$ and $\Lambda_{tgt}$',N,'IPA_Frob_norms')
dump_dictionaries_to_file(stored_b1_δI,stored_b1_time,stored_b1_L2_norm,stored_b1_Frobenius)
print(f'μ_tgt={μ_tgt}')
print(f'κ_tgt={κ_tgt}')
print(f'v_tgt={v_tgt}')
print(f'Λ_tgt={Λ_tgt}')
#μ_tgt=[-0.34208695  0.84284747]
#κ_tgt=11
#v_tgt=15
#Λ_tgt=[[17.88870591  4.84223519]
# [ 4.84223519 17.20649539]]