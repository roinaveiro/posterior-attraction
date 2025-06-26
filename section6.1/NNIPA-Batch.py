import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pyomo.environ as pyo
import time
import copy
import csv

def generate_bivariate_instance(N,P,i,b1_rho,b2max,b3max,precision):
    b1_rho_int=int(100*b1_rho)
    np.random.seed(N+P+i+b1_rho_int+b2max+b3max) # fix the pseudorandom seed as a function of N, P, i, and the other parameters
    μ_prior=np.random.randint(-5,6,size=P)
    Σ_prior=np.random.randint(0,5,size=(P,P))
    Σ_prior=(Σ_prior+Σ_prior.T)     # Part 1 of modifying the covariance matrix to ensure it is PD
    for i in range(P):              # Part 2 of modifying the covariance matrix to ensure it is PD
        Σ_prior[i, i] = sum(abs(Σ_prior[i])) + 1  
    Σ_prior=0.5*Σ_prior
    μ_data_generating=np.random.randint(-5,6,size=P)
    Σ_data_generating=np.random.randint(0,5,size=(P,P))
    Σ_data_generating=(Σ_data_generating+Σ_data_generating.T)  # Part 1 of modifying the covariance matrix to ensure it is PD
    for i in range(P):                                         # Part 2 of modifying the covariance matrix to ensure it is PD
        Σ_data_generating[i, i] = sum(abs(Σ_data_generating[i])) + 1
    Σ_data_generating=0.5*Σ_data_generating
    μ_tgt=np.random.randint(-5,6,size=P)
    Σ_tgt=np.random.randint(0,5,size=(P,P))
    Σ_tgt=(Σ_tgt+Σ_tgt.T)  # Part 1 of modifying the covariance matrix to ensure it is PD
    for i in range(P):     # Part 2 of modifying the covariance matrix to ensure it is PD
        Σ_tgt[i, i] = sum(abs(Σ_tgt[i])) + 1  
    Σ_tgt=0.5*Σ_tgt    
    Xp=np.random.multivariate_normal(mean=μ_data_generating,cov=Σ_data_generating,size=N)
    Xbar_data=np.mean(Xp,axis=0)
    b1=np.minimum(N*P,round(b1_rho*N*P,0))
    b2=np.zeros(N,dtype=int)
    while np.sum(b2)==0:  # Ensure sum(b2) > 0
        b2 = np.minimum(P,np.random.randint(0,b2max+1,size=N)) 
    b3=np.round(np.random.uniform(0,b3max,size=(N,P)),precision)
    return (μ_prior,Σ_prior,μ_data_generating,Σ_data_generating,μ_tgt,Σ_tgt,Xp,Xbar_data,b1,b2,b3)

def kl_bivariate_normal_normal(μ1,Λ1,μ2,Λ2,P): # for exlusive KL, '1' is the posterior distribution and '2' is the target distribution
    return 0.5*((μ2-μ1).T @ Λ2 @ (μ2-μ1) + np.trace(Λ2 @ np.linalg.inv(Λ1)) - np.log(np.linalg.det(np.linalg.inv(Λ1))/np.linalg.det(np.linalg.inv(Λ2)))-P)

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
                         
def kl_normal_normal_given_X(X,μ_prior,Λ_prior,Λ_data_generating,Λ_post,μ_tgt,Λ_tgt,P):
    Xbar_data=np.mean(X,axis=0)
    μ_post=np.linalg.inv(Λ_post) @ (Λ_prior @ μ_prior+N*Λ_data_generating @ Xbar_data)
    δI=kl_bivariate_normal_normal(μ_tgt,Λ_tgt,μ_post,Λ_post,P)
    return δI

def approximate_gradient(Xp, μ_prior, Λ_prior, Λ_data_generating, Λ_post, μ_tgt, Λ_tgt, P, precision):
    X = np.array(Xp, dtype=float)
    N, P = X.shape
    gradient = np.zeros_like(X)
    epsilon = 1e-5  # Small perturbation for numerical differentiation

    # Compute KL divergence for the original X once
    kl_base = kl_normal_normal_given_X(X, μ_prior, Λ_prior, Λ_data_generating, Λ_post, μ_tgt, Λ_tgt, P)

    for n in range(N):
        for p in range(P):
            X_perturbed = X.copy()
            X_perturbed[n, p] += epsilon  # Small perturbation
            kl_perturbed = kl_normal_normal_given_X(X_perturbed, μ_prior, Λ_prior, Λ_data_generating, Λ_post, μ_tgt, Λ_tgt, P)
            gradient[n, p] = (kl_perturbed - kl_base) / epsilon  # Finite difference approximation

    return np.round(gradient, precision)

def approximate_hessian(Xp, μ_prior, Λ_prior, Λ_data_generating, Λ_post, μ_tgt, Λ_tgt, P, precision):
    X = np.array(Xp, dtype=float)
    N, P = X.shape
    H = np.zeros((N, P, N, P))  # Hessian tensor
    epsilon = 1e-5  # Small step size for numerical stability

    # Compute KL divergence at the unperturbed X
    kl_base = kl_normal_normal_given_X(X, μ_prior, Λ_prior, Λ_data_generating, Λ_post, μ_tgt, Λ_tgt, P)

    for n in range(N):
        for p in range(P):
            for k in range(N):
                for l in range(P):
                    # Perturb X in both (n,p) and (k,l) directions
                    X_np_kl_plus = X.copy()
                    X_np_kl_minus = X.copy()
                    X_np_kl_plus[n, p] += epsilon
                    X_np_kl_plus[k, l] += epsilon
                    X_np_kl_minus[n, p] -= epsilon
                    X_np_kl_minus[k, l] -= epsilon

                    X_np_plus = X.copy()
                    X_np_minus = X.copy()
                    X_kl_plus = X.copy()
                    X_kl_minus = X.copy()

                    X_np_plus[n, p] += epsilon
                    X_np_minus[n, p] -= epsilon
                    X_kl_plus[k, l] += epsilon
                    X_kl_minus[k, l] -= epsilon

                    # Compute KL divergences for perturbed matrices
                    kl_np_plus = kl_normal_normal_given_X(X_np_plus, μ_prior, Λ_prior, Λ_data_generating, Λ_post, μ_tgt, Λ_tgt, P)
                    kl_np_minus = kl_normal_normal_given_X(X_np_minus, μ_prior, Λ_prior, Λ_data_generating, Λ_post, μ_tgt, Λ_tgt, P)
                    kl_kl_plus = kl_normal_normal_given_X(X_kl_plus, μ_prior, Λ_prior, Λ_data_generating, Λ_post, μ_tgt, Λ_tgt, P)
                    kl_kl_minus = kl_normal_normal_given_X(X_kl_minus, μ_prior, Λ_prior, Λ_data_generating, Λ_post, μ_tgt, Λ_tgt, P)
                    kl_np_kl_plus = kl_normal_normal_given_X(X_np_kl_plus, μ_prior, Λ_prior, Λ_data_generating, Λ_post, μ_tgt, Λ_tgt, P)
                    kl_np_kl_minus = kl_normal_normal_given_X(X_np_kl_minus, μ_prior, Λ_prior, Λ_data_generating, Λ_post, μ_tgt, Λ_tgt, P)

                    # Compute second derivative using central finite differences
                    H[n, p, k, l] = (kl_np_kl_plus - kl_np_plus - kl_kl_plus + 2 * kl_base - kl_np_minus - kl_kl_minus + kl_np_kl_minus) / (epsilon ** 2)

    return np.round(H, precision)

def plot_bivariate_normals(μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt):
    # Create a grid of points for the plot
    x = np.linspace(-5,5,500)
    y = np.linspace(-5,5,500)
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
    plt.show()
    return

def plot_bivariate_normals_with_poisoning(μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,μ_poisoned,title):
    # Create a grid of points for the plot
    x = np.linspace(-5,5,500)
    y = np.linspace(-5,5,500)
    X,Y = np.meshgrid(x,y)
    pos = np.dstack((X,Y))
    # Create the normal distributions with different means and covariance matrices
    rv1 = multivariate_normal(mean=μ_prior,cov=Σ_prior)
    rv2 = multivariate_normal(mean=μ_post,cov=Σ_post)
    rv3 = multivariate_normal(mean=μ_tgt,cov=Σ_tgt)
    rv4 = multivariate_normal(mean=μ_poisoned,cov=Σ_post)
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
    plt.text(μ_poisoned[0] + 0.2,μ_poisoned[1],'μ_poisoned',color='purple')
    # Add labels and title
    plt.title(f"Prior, Posterior, Target, and {title} Poisoned Distributions")
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.grid(True)
    plt.show()
    return

def plot_bivariate_normals_with_poisoningXpX(i_count,b1_name,b2max,b3max,μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,μ_poisoned,title,Xp,X):
    # Ensure Xp and X are NumPy arrays
    Xp = np.array(Xp)
    X = np.array(X)
    if Xp.shape[0]>5 :  # Don't graph dense plots    
        return

    # Find the extents of the data and means
    all_points = np.vstack([Xp,X,μ_prior,μ_post,μ_tgt,μ_poisoned])
    x_min,y_min = np.min(all_points,axis=0) - 1  # Add some margin
    x_max,y_max = np.max(all_points,axis=0) + 1
    
    # Create a grid of points for the plot based on the data extents
    x = np.linspace(x_min,x_max,500)
    y = np.linspace(y_min,y_max,500)
    X_grid,Y_grid = np.meshgrid(x,y)
    pos = np.dstack((X_grid,Y_grid))
    
    # Create the normal distributions with different means and covariance matrices
    rv1 = multivariate_normal(mean=μ_prior,cov=Σ_prior)
    rv2 = multivariate_normal(mean=μ_post,cov=Σ_post)
    rv3 = multivariate_normal(mean=μ_tgt,cov=Σ_tgt)
    rv4 = multivariate_normal(mean=μ_poisoned,cov=Σ_post)
    
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
        plt.scatter(Xp[i,0],Xp[i,1],color='lightblue',edgecolor='black',label='Clean Data' if i == 0 else "",zorder=7)
        plt.scatter(X[i,0],X[i,1],color='red',edgecolor='black',label='Attacked Data' if i == 0 else "",zorder=6)
        
        # Check if the points are significantly different (based on the condition)
        if np.any(np.abs(Xp[i] - X[i]) > 0.01):  
            # Plot a gray arrow between the points if they are different
            plt.arrow(
                Xp[i,0],Xp[i,1],
                X[i,0] - Xp[i,0],X[i,1] - Xp[i,1],
                color='grey',head_width=0.1,length_includes_head=True,zorder=5
            )
    
    # Add labels,legend,and title
    plt.title(f"Prior, Posterior, Target, and {title} poisoned Distributions")
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"C:/Users/brian/Documents/PA/Plots/NN_IPA_N{N}_b1_{b1_name}_b2_{b2max}_b3_{b3max}_i{i_count+1}_{title}.png", dpi=300, bbox_inches='tight')     
    plt.show()

def solve_P2L_δI(N,P,μ_prior,Σ_prior,Λ_prior,Σ_data_generating,Λ_data_generating,Xp,Σ_post,Λ_post,μ_tgt,Σ_tgt,Λ_tgt,b1,b2,b3,precision,MP_time_limit,solver_name):
    start_time = time.time() # Start the clock
    # Create a Pyomo model
    model = pyo.ConcreteModel()
    # Sets
    model.N = pyo.Set(initialize=create_list_from_integer(N))
    model.P = pyo.Set(initialize=create_list_from_integer(P))
    # Parameters
    model.μ_prior=pyo.Param(model.P,initialize=list_to_dict(μ_prior))
    model.Σ_prior=pyo.Param(model.P,model.P,initialize=matrix_to_dict(Σ_prior))
    model.Λ_prior=pyo.Param(model.P,model.P,initialize=matrix_to_dict(Λ_prior))
    model.Σ_data_generating=pyo.Param(model.P,model.P,initialize=matrix_to_dict(Σ_data_generating))
    model.Λ_data_generating=pyo.Param(model.P,model.P,initialize=matrix_to_dict(Λ_data_generating))
    model.Xp=pyo.Param(model.N,model.P,initialize=matrix_to_dict(Xp))
    model.Σ_post=pyo.Param(model.P,model.P,initialize=matrix_to_dict(Σ_post))
    model.Λ_post=pyo.Param(model.P,model.P,initialize=matrix_to_dict(Λ_post))
    model.μ_tgt=pyo.Param(model.P,initialize=list_to_dict(μ_tgt))
    model.Σ_tgt=pyo.Param(model.P,model.P,initialize=matrix_to_dict(Σ_tgt))
    model.Λ_tgt=pyo.Param(model.P,model.P,initialize=matrix_to_dict(Λ_tgt))
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
        # Find the maximum magnitude
        return max(magnitudes)
    model.BigM = pyo.Param(initialize=compute_big_m)
    # Decision Variables
    def initialize_X(model,n,p):
        return model.Xp[n,p]
    model.X=pyo.Var(model.N,model.P,domain=pyo.Reals,initialize=initialize_X) 
    model.Xbar=pyo.Var(model.P,domain=pyo.Reals)
    model.Ω=pyo.Var(model.N,model.P,domain=pyo.Binary,initialize=0) # 1 of |X(n,p)-X'(n,p)|>0 and 0 otherwise
    model.μ_post=pyo.Var(model.P,initialize=list_to_dict(μ_prior))
    # Formulate model
    # Objective function 
    model.objfnvalue = pyo.Objective(
        expr =0.5*(sum(sum((model.μ_post[i]-model.μ_tgt[i])*model.Λ_post[i,j]*(model.μ_post[j]-model.μ_tgt[j]) for i in model.P) for j in model.P)
                                     + sum(sum(model.Λ_post[i,j]*model.Σ_tgt[j,i] for i in model.P) for j in model.P)
                                     - np.log(np.linalg.det(Σ_tgt)/np.linalg.det(Σ_post))-P),sense=pyo.minimize)
    model.Calculate_Xbar=pyo.ConstraintList()
    for p in model.P:
        model.Calculate_Xbar.add(model.Xbar[p]==sum(model.X[n,p] for n in model.N)/N)
    model.Calculate_μ_post=pyo.ConstraintList() 
    for p in model.P:
        model.Calculate_μ_post.add(model.μ_post[p]==sum(model.Σ_post[p,i]*(sum(model.Λ_prior[i,j]*model.μ_prior[j] for j in model.P) 
                                                        + N*sum(model.Λ_data_generating[i,j]*model.Xbar[j] for j in model.P)) 
                                                        for i in model.P))

    model.LimitTotalChanges=pyo.Constraint(expr=sum(sum(model.Ω[n,p] for n in model.N) for p in model.P)<=model.b1)
    model.LimitChangeInEntryI=pyo.ConstraintList()
    for n in model.N:
        model.LimitChangeInEntryI.add(sum(model.Ω[n,p] for p in model.P)<=model.b2[n])
    model.LimitEachChange=pyo.ConstraintList()
    for n in model.N:
        for p in model.P:
            model.LimitEachChange.add(model.X[n,p]-model.Xp[n,p]<=model.b3[n,p])
            model.LimitEachChange.add(model.Xp[n,p]-model.X[n,p]<=model.b3[n,p])
    model.ΩCalcs=pyo.ConstraintList()
    for n in model.N:
        for p in model.P:
            model.ΩCalcs.add(-model.BigM*model.Ω[n,p]<=model.X[n,p]-model.Xp[n,p])
            model.ΩCalcs.add(model.X[n,p]-model.Xp[n,p]<=model.BigM*model.Ω[n,p])

    # Impose upper and lower bounds on DVs to impose a hyperrectangle and help the solver converge
    model.DVbounds=pyo.ConstraintList()
    for n in model.N:
        for p in model.P:
            model.DVbounds.add(model.X[n,p]<=model.Xp[n,p]+model.b3[n,p])  #bounds that will not cut off an optimal solution
            model.DVbounds.add(model.X[n,p]>=model.Xp[n,p]-model.b3[n,p])
             
    try:    
        if (solver_name=='CPLEX'):
            solver = pyo.SolverFactory('cplex')
            solver.options['optimalitytarget'] = 3  # Solves a non-convex Q2L1
            solver.options['timelimit'] = MP_time_limit 
            solver.solve(model,tee=False)
        elif (solver_name=="SCIP"):
            solver=pyo.SolverFactory('scip', solver_io='nl')
            solver.options['limits/time']=MP_time_limit  # Given SCIP the maximum time allowable
            solver.solve(model,tee=False)
        else:
            pyo.SolverFactory('mindtpy').solve(model,time_limit=MP_time_limit,tee=False)
        attempt='success'
    except:
        attempt='fail'
    
    end_time = time.time() # Stop the clock
    elapsed_time = end_time - start_time # Calculate the elapsed time   

    if (attempt=='success'):
        solution={
        'X':[[pyo.value(model.X[n,p]) for p in model.P] for n in model.N],
        'Ω':[[pyo.value(model.Ω[n,p]) for p in model.P] for n in model.N],
        'μ_poisoned':[pyo.value(model.μ_post[p]) for p in model.P],
        'δI(X)':pyo.value(model.objfnvalue),           
        'time':round(elapsed_time,precision)
        }    
    else:
        solution={
        'X':[[pyo.value(model.Xp[n,p]) for p in model.P] for n in model.N],
        'Ω':[[0 for p in model.P] for n in model.N],
        'μ_poisoned':[999999 for p in model.P],
        'δI(X)':999999,    
        'time':round(elapsed_time,precision)
        }  
    return solution

def solve_Q2L1_for_Q2L(N,P,Xest,Xprime,μ_prior,Λ_prior,Λ_data_generating,Σ_post,Λ_post,μ_tgt,Λ_tgt,b1,b2,b3,MP_time_limit,precision,solver_name):
    start_time = time.time() # Start the clock
    ### The next two lines construct a quadratic approximation based on the point Xest rather than Xp ###
    grad_list = approximate_gradient(Xest,μ_prior,Λ_prior,Λ_data_generating,Λ_post,μ_tgt,Λ_tgt,P,precision)
    H_list = approximate_hessian(Xest,μ_prior,Λ_prior,Λ_data_generating,Λ_post,μ_tgt,Λ_tgt,P,precision)
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
        # Find the maximum magnitude
        return max(magnitudes)
    model.BigM = pyo.Param(initialize=compute_big_m)
    model.f=pyo.Param(initialize=kl_normal_normal_given_X(Xest,μ_prior,Λ_prior,Λ_data_generating,Λ_post,μ_tgt,Λ_tgt,P))
    # Decision Variables
    def initialize_X(model,n,p):
        return model.Xest[n,p]
    model.X=pyo.Var(model.N,model.P,domain=pyo.Reals,initialize=initialize_X) 
    model.Ω=pyo.Var(model.N,model.P,domain=pyo.Binary,initialize=0) # 1 of |X(n,p)-X'(n,p)|>0 and 0 otherwise
    # Formulate model
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
            solver.options['limits/time']=MP_time_limit  # Given SCIP the maximum time allowable
            solver.solve(model,tee=False)
        else:
            pyo.SolverFactory('mindtpy').solve(model,time_limit=MP_time_limit,tee=False)
        attempt='success'
    except:
        attempt='fail'

    end_time = time.time() # Stop the clock
    elapsed_time = end_time - start_time # Calculate the elapsed time
    
    if (attempt=='success'):
        solution={
            'X':[[pyo.value(model.X[n,p]) for p in model.P] for n in model.N],
            'Ω':[[pyo.value(model.Ω[n,p]) for p in model.P] for n in model.N],
            'q(X)':round(pyo.value(model.objfnvalue),precision),
            'time':round(elapsed_time,precision)
            }
    else:
        solution={
            'X':[[pyo.value(model.Xest[n,p]) for p in model.P] for n in model.N],
            'Ω':[[0 for p in model.P] for n in model.N],
            'q(X)':999999,
            'time':round(elapsed_time,precision)
            }
    return solution

def solve_Q2L(N,P,Xprime,μ_prior,Λ_prior,Λ_data_generating,Σ_post,Λ_post,μ_tgt,Λ_tgt,b1,b2,b3,MP_time_limit,precision,QA_max_time,QA_max_iterations,QA_termination_tolerance,solver_name):
    stored_solutions={}
    k_iter=1
    termination_check=QA_termination_tolerance+1
    Xest=Xprime #construct quadratic approximation about the point Xest
    f_k=1 #temporary initialization to define f_k
    start_time = time.time() # Start the clock
    elapsed_time=0
    
    while (k_iter<=QA_max_iterations and elapsed_time<QA_max_time and termination_check>QA_termination_tolerance):
        solution_Q2L1=solve_Q2L1_for_Q2L(N,P,Xest,Xprime,μ_prior,Λ_prior,Λ_data_generating,Σ_post,Λ_post,μ_tgt,Λ_tgt,b1,b2,b3,MP_time_limit,precision,solver_name)
        stored_solutions[k_iter]=solution_Q2L1
        if k_iter==1: 
            f_kminus1=kl_normal_normal_given_X(solution_Q2L1['X'],μ_prior,Λ_prior,Λ_data_generating,Λ_post,μ_tgt,Λ_tgt,P)+QA_termination_tolerance+1 
        else:
            f_kminus1=f_k
        f_k=kl_normal_normal_given_X(solution_Q2L1['X'],μ_prior,Λ_prior,Λ_data_generating,Λ_post,μ_tgt,Λ_tgt,P)
        termination_check=f_kminus1-f_k
        if f_k<f_kminus1: # only accept a better solution
            #print(f"-Accepting a better solution on iteration {k_iter}")
            Xest=solution_Q2L1['X']
            solution=solution_Q2L1
        #else:
            #print(f"-Rejecting a worse solution on iteration {k_iter}")
        k_iter=k_iter+1
        end_time = time.time() # Stop the clock
        elapsed_time = end_time - start_time # Calculate the elapsed time    
    # make sure the reported time is the cumulative time, not the time of the last Q2L1 solution
    solution['time']=round(elapsed_time,precision)
    solution['iterations']=k_iter-1
    return (solution,stored_solutions)   

def solve_2M2(N,P,μ_prior,Σ_prior,Λ_prior,Σ_data_generating,Λ_data_generating,Xp,Σ_post,Λ_post,μ_tgt,Σ_tgt,Λ_tgt,b1,b2,b3,precision,MP_time_limit,solver_name):
    start_time = time.time() # Start the clock
    # Create a Pyomo model
    model = pyo.ConcreteModel()
    # Sets
    model.N = pyo.Set(initialize=create_list_from_integer(N))
    model.P = pyo.Set(initialize=create_list_from_integer(P))
    # Parameters
    model.μ_prior=pyo.Param(model.P,initialize=list_to_dict(μ_prior))
    model.Σ_prior=pyo.Param(model.P,model.P,initialize=matrix_to_dict(Σ_prior))
    model.Λ_prior=pyo.Param(model.P,model.P,initialize=matrix_to_dict(Λ_prior))
    model.Σ_data_generating=pyo.Param(model.P,model.P,initialize=matrix_to_dict(Σ_data_generating))
    model.Λ_data_generating=pyo.Param(model.P,model.P,initialize=matrix_to_dict(Λ_data_generating))
    model.Xp=pyo.Param(model.N,model.P,initialize=matrix_to_dict(Xp))
    model.Σ_post=pyo.Param(model.P,model.P,initialize=matrix_to_dict(Σ_post))
    model.Λ_post=pyo.Param(model.P,model.P,initialize=matrix_to_dict(Λ_post))
    model.μ_tgt=pyo.Param(model.P,initialize=list_to_dict(μ_tgt))
    model.Σ_tgt=pyo.Param(model.P,model.P,initialize=matrix_to_dict(Σ_tgt))
    model.Λ_tgt=pyo.Param(model.P,model.P,initialize=matrix_to_dict(Λ_tgt))
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
        # Find the maximum magnitude
        return max(magnitudes)
    model.BigM = pyo.Param(initialize=compute_big_m)
    # Decision Variables
    def initialize_X(model,n,p):
        return model.Xp[n,p]
    model.X=pyo.Var(model.N,model.P,domain=pyo.Reals,initialize=initialize_X) 
    model.Xbar=pyo.Var(model.P,domain=pyo.Reals)
    model.Ω=pyo.Var(model.N,model.P,domain=pyo.Binary,initialize=0) # 1 of |X(n,p)-X'(n,p)|>0 and 0 otherwise
    model.μ_post=pyo.Var(model.P,initialize=list_to_dict(μ_prior))
    # Formulate model
    # Objective function 
    model.objfnvalue = pyo.Objective(expr =sum((model.μ_post[p]-model.μ_tgt[p])**2 for p in model.P),sense=pyo.minimize)

    model.Calculate_Xbar=pyo.ConstraintList()
    for p in model.P:
        model.Calculate_Xbar.add(model.Xbar[p]==sum(model.X[n,p] for n in model.N)/N)
    model.Calculate_μ_post=pyo.ConstraintList() 
    for p in model.P:
        model.Calculate_μ_post.add(model.μ_post[p]==sum(model.Σ_post[p,i]*(sum(model.Λ_prior[i,j]*model.μ_prior[j] for j in model.P) 
                                                        + N*sum(model.Λ_data_generating[i,j]*model.Xbar[j] for j in model.P)) 
                                                        for i in model.P))
    
    model.LimitTotalChanges=pyo.Constraint(expr=sum(sum(model.Ω[n,p] for n in model.N) for p in model.P)<=model.b1)
    model.LimitChangeInEntryI=pyo.ConstraintList()
    for n in model.N:
        model.LimitChangeInEntryI.add(sum(model.Ω[n,p] for p in model.P)<=model.b2[n])
    model.LimitEachChange=pyo.ConstraintList()
    for n in model.N:
        for p in model.P:
            model.LimitEachChange.add(model.X[n,p]-model.Xp[n,p]<=model.b3[n,p])
            model.LimitEachChange.add(model.Xp[n,p]-model.X[n,p]<=model.b3[n,p])
    model.ΩCalcs=pyo.ConstraintList()
    for n in model.N:
        for p in model.P:
            model.ΩCalcs.add(model.Xp[n,p]-model.X[n,p]<=model.BigM*model.Ω[n,p])
            model.ΩCalcs.add(model.X[n,p]-model.Xp[n,p]<=model.BigM*model.Ω[n,p])
    
    # Impose upper and lower bounds on DVs to impose a hyperrectangle and help the solver converge
    model.DVbounds=pyo.ConstraintList()
    for n in model.N:
        for p in model.P:
            model.DVbounds.add(model.X[n,p]<=model.Xp[n,p]+model.b3[n,p])  #bounds that will not cut off an optimal solution
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
            solver.options['limits/time']=MP_time_limit  # Given SCIP the maximum time allowable
            solver.solve(model,tee=False)
        else:
            pyo.SolverFactory('mindtpy').solve(model,time_limit=MP_time_limit,tee=False)
        attempt='success'
    except:
        attempt='fail'
    
    end_time = time.time() # Stop the clock
    elapsed_time = end_time - start_time # Calculate the elapsed time   
    
    if (attempt=='success'):
        solution={
            'X':[[pyo.value(model.X[n,p]) for p in model.P] for n in model.N],
            'Ω':[[pyo.value(model.Ω[n,p]) for p in model.P] for n in model.N],
            'μ_poisoned':[pyo.value(model.μ_post[p]) for p in model.P],
            '2M(X)':round(pyo.value(model.objfnvalue),precision),           
            'time':round(elapsed_time,precision)
            }    
    else:
        solution={
            'X':[[pyo.value(model.Xp[n,p]) for p in model.P] for n in model.N],
            'Ω':[[0 for p in model.P] for n in model.N],
            'μ_poisoned':999999,
            '2M(X)':999999,           
            'time':round(elapsed_time,precision)
            }    
    return solution

def FGA2(N,P,Xp,b1,b2,b3,μ_prior,Λ_prior,Λ_data_generating,Λ_post,μ_tgt,Λ_tgt,precision): 
    start_time = time.time()  # Start the clock
    Xprime = copy.deepcopy(Xp)
    X = Xprime
    N,P = len(X),len(X[0])  # Dimensions of the matrix
    modified_entries = set()  # To track which entries have been modified
    row_changes = [0 for _ in range(N)]  # Track row-specific changes
    # Compute the gradient
    grad = approximate_gradient(X,μ_prior,Λ_prior,Λ_data_generating,Λ_post,μ_tgt,Λ_tgt,P,precision)
    # Flatten and sort entries by gradient magnitude
    grad_abs = [(i,j,abs(grad[i][j])) for i in range(N) for j in range(P)]
    grad_abs.sort(key=lambda x: x[2],reverse=True)  # Sort by absolute gradient value, descending

    # Initialize a list to store the changes to apply
    updates = []
    # Apply updates iteratively based on gradient magnitude
    for i,j, _ in grad_abs:
        if len(updates) >= b1:  # Stop when the global budget is reached
            break
        if (i,j) in modified_entries:
            continue  # Skip this entry if it has been modified already   
        # Ensure we respect b2[i] constraints
        if row_changes[i] < b2[i]:  # Only apply to X[i][0] or X[i][1] when b2[i] allows it
            # Compute perturbation size, clipped to b3
            delta = -b3[i][j] * (grad[i][j] > 0) + b3[i][j] * (grad[i][j] < 0)  # -1 or 1 based on gradient sign
            updates.append((i,j,delta))
            row_changes[i] += 1
            modified_entries.add((i,j))  # Mark this entry as modified
    # Apply the updates to the matrix X
    for i,j,delta in updates:
        X[i][j] = X[i][j] + delta
    end_time = time.time()  # Stop the clock
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    solution = {
        'X':[[float(X[n][p]) for p in range(P)] for n in range(N)],
        'δI(X)':float(kl_normal_normal_given_X(X,μ_prior,Λ_prior,Λ_data_generating,Λ_post,μ_tgt,Λ_tgt,P)),
        'time':round(elapsed_time,precision)
    }

    return solution

###  ROI CODE AND FUNCTION DEFINITIONS START HERE ###

import torch
from methods.pgd import ProjectedGradientDescent
from methods.iscd import ISCD
from models.multivariate_normal_known_variance import MultivariateNormalKnownVarianceModel
from constraints.combined_constraint import CombinedConstraint
#from utils.plotting import visualize_posteriors_and_data_2D

def generate_config(μ_prior,Σ_prior,Σ_data_generating,μ_tgt,Σ_tgt,Xp,b1,b2,b3):

    model = MultivariateNormalKnownVarianceModel(
        prior_mean=μ_prior,
        prior_cov=Σ_prior,
        known_cov=Σ_data_generating)

    target_posterior_params = {
        'mean': μ_tgt,
        'cov': Σ_tgt
    }

    X_clean = torch.tensor(Xp, dtype=torch.float32)
    data_dict = {"X" : X_clean}
    target_posterior = model.define_adversarial_posterior(target_posterior_params, Xp.shape[0])

    clean_posterior = model.compute_posterior(data_dict)

    # For PGD
    constraintl0 = CombinedConstraint(max_row_changes=b2, 
                    max_total_changes=b1, max_element_change=b3)
    constraints_pgd = [constraintl0]

    # For ISCD
    constraints_iscd = {
        "X": {
            "max_element_change": b3,
            "max_row_changes": b2,
            "max_changes": b1
        }
    }

    return data_dict, model, clean_posterior, target_posterior, constraints_pgd, constraints_iscd

###  ROI CODE AND FUNCTION DEFINITIONS END HERE ###

def compute_change_metrics(Xp: np.ndarray, X: np.ndarray):
    Xp=np.array(Xp)
    X=np.array(X)
    # Ensure the matrices have the same shape
    assert Xp.shape == X.shape, "Matrices must have the same shape"
    
    # Compute the absolute difference
    diff = np.abs(Xp - X)
    #print(diff)    
    num_changed_elements = np.sum(diff >= 0.01)
    
    num_changed_rows = np.sum(np.any(diff >= 0.01, axis=1))  
    
    mean_Xp = np.mean(Xp, axis=0)
    mean_X = np.mean(X, axis=0)
    euclidean_distance = np.linalg.norm(mean_Xp - mean_X)
    
    return num_changed_rows, num_changed_elements, euclidean_distance

def dump_aggregate_data(stored_aggregate_results,file_path_aggregate):
    # Extract column headers from the first dictionary (assuming all dictionaries have the same structure)
    fieldnames = stored_aggregate_results[next(iter(stored_aggregate_results))].keys()
    # Write the dictionary of dictionaries to the CSV file
    with open(file_path_aggregate, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for stored_aggregate_results_id, stored_aggregate_results_data in stored_aggregate_results.items():
            writer.writerow(stored_aggregate_results_data)
    print(f"Data written to {file_path_aggregate}")
    return 1

def run_methods(i,b1_rho,b1_name,b2max,b3max,μ_prior,Σ_prior,Λ_prior,N,P,μ_data_generating,Σ_data_generating,Λ_data_generating,Xp,Xbar_data,b1,b2,b3,μ_tgt,Σ_tgt,Λ_tgt,MP_time_limit,QA_max_time,QA_max_iterations,QA_termination_tolerance,kl_function,SCD_epsilon,SCD_verbose,SCD_max_oscillations,SCD_second_order,PGD_lr,PGD_init_noise,PGD_max_iter,PGD_verbose,PGD_tolerance,PGD_no_change_steps):
    solvers={'MindtPy','SCIP','CPLEX'}    
    #print('Xp=',Xp)
    #print('******************************')
    print(f"Iteration {i+1} with b1_rho={b1_rho}, b2={b2max}, b3={b3max}")
    #print('******************************')
    ### Compute the Unpoisoned Bayesian Posterior Distribution & KL Divergences, and Create Plot ###
    ### Compute the Bayesian Posterior Distribution ###
    Λ_post=Λ_prior+N*Λ_data_generating
    Σ_post=np.linalg.inv(Λ_post)
    #print('Σ_post',Σ_post)
    μ_post=Σ_post @ (Λ_prior @ μ_prior+N*Λ_data_generating @ Xbar_data)
    #print(f"μ_prior={μ_prior}")
    #print(f"μ_data_generating={μ_data_generating}")
    #print(f"X'={Xp}")
    #print(f"Xbar_data={Xbar_data}")
    #print(f"μ_post={μ_post}")
    δ_kl_ex=round(kl_bivariate_normal_normal(μ_post,Λ_post,μ_tgt,Λ_tgt,P),precision)
    δ_kl_in=round(kl_bivariate_normal_normal(μ_tgt,Λ_tgt,μ_post,Λ_post,P),precision)
    #print('******************************')
    #print(f"μ_tgt={μ_tgt}")
    #print(f"Σ_tgt={Σ_tgt}")
    #print(f"δE(X')={δ_kl_ex}; δI(X')={δ_kl_in}")
    #print(f"b1={b1} out of {N*P} entries may be permuted")
    #print(f"b2={b2} out of {P} entries in row n are permutable")
    #print(f"b3={b3}")
    #plot_bivariate_normals(μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt)
    
    δI_poisoned_list={}
    solve_time_list={}
    num_changed_rows_list={}
    num_changed_list={}
    Eucl_dist_list={}
    solvers={'SCIP'}    
    for solver_name in sorted(solvers):    
        #print('******************************')
        print(f"P2L Results with {solver_name}")
        solution_P2L=solve_P2L_δI(N,P,μ_prior,Σ_prior,Λ_prior,Σ_data_generating,Λ_data_generating,Xp,Σ_post,Λ_post,μ_tgt,Σ_tgt,Λ_tgt,b1,b2,b3,precision,MP_time_limit,solver_name)
        X=solution_P2L['X']
        Xbar=np.mean(X,axis=0)
        μ_poisoned=Σ_post @ (Λ_prior @ μ_prior+N*Λ_data_generating @ Xbar)
        #print(f"μ_poisoned={μ_poisoned}")
        δI_poisoned=round(kl_bivariate_normal_normal(μ_tgt,Λ_tgt,μ_poisoned,Λ_post,P),precision)
        #print(f"δI(X)={δI_poisoned}")
        #print(f"time={solution_P2L['time']} sec")
        title='P2L-'+solver_name[0]
        #plot_bivariate_normals_with_poisoning(μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,μ_poisoned,title)
        plot_bivariate_normals_with_poisoningXpX(i,b1_name,b2max,b3max,μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,μ_poisoned,title,Xp,X)
        δI_poisoned_list[f"{title} KL"]=float(δI_poisoned)
        solve_time_list[f"{title} time"]=solution_P2L['time']
        (num_changed_rows,num_changed,Eucl_dist)=compute_change_metrics(Xp,X)
        num_changed_rows_list[f"{title} rows chng"]=num_changed_rows
        num_changed_list[f"{title} entries chng"]=num_changed
        Eucl_dist_list[f"{title} L2"]=float(round(Eucl_dist,precision))
    solvers={'CPLEX'}            
    for solver_name in sorted(solvers):    
        #print('******************************')
        print(f"Q2L Results with {solver_name}")
        (solution_Q2L,Q2L_trace)=solve_Q2L(N,P,Xp,μ_prior,Λ_prior,Λ_data_generating,Σ_post,Λ_post,μ_tgt,Λ_tgt,b1,b2,b3,MP_time_limit,precision,QA_max_time,QA_max_iterations,QA_termination_tolerance,solver_name)
        X=solution_Q2L['X']
        Xbar=np.mean(X,axis=0)
        μ_poisoned=Σ_post @ (Λ_prior @ μ_prior+N*Λ_data_generating @ Xbar)
        δI_poisoned=round(kl_bivariate_normal_normal(μ_tgt,Λ_tgt,μ_poisoned,Λ_post,P),precision)
        #print(f"μ_poisoned={μ_poisoned}")
        #print(f"δI(X)={δI_poisoned}")
        #print(f"time={solution_Q2L['time']} sec")
        title='Q2L-'+solver_name[0]
        #plot_bivariate_normals_with_poisoning(μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,μ_poisoned,title)
        plot_bivariate_normals_with_poisoningXpX(i,b1_name,b2max,b3max,μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,μ_poisoned,title,Xp,X)
        δI_poisoned_list[f"{title} KL"]=float(δI_poisoned)
        solve_time_list[f"{title} time"]=solution_Q2L['time']
        (num_changed_rows,num_changed,Eucl_dist)=compute_change_metrics(Xp,X)
        num_changed_rows_list[f"{title} rows chng"]=num_changed_rows
        num_changed_list[f"{title} entries chng"]=num_changed
        Eucl_dist_list[f"{title} L2"]=float(round(Eucl_dist,precision))
    '''    
    for solver_name in sorted(solvers):    
        #print('******************************')
        print(f"Q2L1 Results with {solver_name}")
        #solution_Q2L1=solve_Q2L1(N,P,Xp,μ_prior,Λ_prior,Λ_data_generating,Σ_post,Λ_post,μ_tgt,Λ_tgt,b1,b2,b3,MP_time_limit,precision,solver_name)
        solution_Q2L1=solve_Q2L1_for_Q2L(N,P,Xp,Xp,μ_prior,Λ_prior,Λ_data_generating,Σ_post,Λ_post,μ_tgt,Λ_tgt,b1,b2,b3,MP_time_limit,precision,solver_name)
        X=solution_Q2L1['X']
        Xbar=np.mean(X,axis=0)
        μ_poisoned=Σ_post @ (Λ_prior @ μ_prior+N*Λ_data_generating @ Xbar)
        #print(f"μ_poisoned={μ_poisoned}")
        δI_poisoned=round(kl_bivariate_normal_normal(μ_tgt,Λ_tgt,μ_poisoned,Λ_post,P),precision)
        #print(f"δI(X)={δI_poisoned}")
        #print(f"time={solution_Q2L1['time']} sec")
        title='Q2L1-'+solver_name[0]
        #plot_bivariate_normals_with_poisoning(μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,μ_poisoned,title)
        plot_bivariate_normals_with_poisoningXpX(i,b1_name,b2max,b3max,μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,μ_poisoned,title,Xp,X)
        δI_poisoned_list[f"{title} KL"]=float(δI_poisoned)
        solve_time_list[f"{title} time"]=solution_Q2L1['time']
        (num_changed_rows,num_changed,Eucl_dist)=compute_change_metrics(Xp,X)
        num_changed_rows_list[f"{title} rows chng"]=num_changed_rows
        num_changed_list[f"{title} entries chng"]=num_changed
        Eucl_dist_list[f"{title} L2"]=float(round(Eucl_dist,precision))
    '''    
    for solver_name in sorted(solvers):    
        #print('******************************')
        print(f"2M2 Results with {solver_name}")
        solution_2M2=solve_2M2(N,P,μ_prior,Σ_prior,Λ_prior,Σ_data_generating,Λ_data_generating,Xp,Σ_post,Λ_post,μ_tgt,Σ_tgt,Λ_tgt,b1,b2,b3,precision,MP_time_limit,solver_name)
        X=solution_2M2['X']
        Xbar=np.mean(X,axis=0)
        μ_poisoned=Σ_post @ (Λ_prior @ μ_prior+N*Λ_data_generating @ Xbar)
        δI_poisoned=round(kl_bivariate_normal_normal(μ_tgt,Λ_tgt,μ_poisoned,Λ_post,P),precision)
        #print(f"μ_poisoned={μ_poisoned}")
        #print(f"2M2(X)={np.sqrt(solution_2M2['2M(X)'])}")
        #print(f"δI(X)={δI_poisoned}")
        #print(f"time={solution_2M2['time']} sec")
        title='2M2-'+solver_name[0]
        #plot_bivariate_normals_with_poisoning(μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,μ_poisoned,title)
        plot_bivariate_normals_with_poisoningXpX(i,b1_name,b2max,b3max,μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,μ_poisoned,title,Xp,X)
        δI_poisoned_list[f"{title} KL"]=float(δI_poisoned)
        solve_time_list[f"{title} time"]=solution_2M2['time']
        (num_changed_rows,num_changed,Eucl_dist)=compute_change_metrics(Xp,X)
        num_changed_rows_list[f"{title} rows chng"]=num_changed_rows
        num_changed_list[f"{title} entries chng"]=num_changed
        Eucl_dist_list[f"{title} L2"]=float(round(Eucl_dist,precision))        
    '''
    #print('******************************')
    print('PGD Results')
    data_dict, model, clean_posterior, target_posterior, constraints_pgd, _ = generate_config(μ_prior,Σ_prior,Σ_data_generating,μ_tgt,Σ_tgt,Xp,b1,b2,b3)
    # Initialize PGD optimizer
    pgd_optimizer = ProjectedGradientDescent(model, 
                     target_posterior, 
                     kl_function, 
                     lr=PGD_lr, 
                     max_iter=PGD_max_iter,
                     tolerance=PGD_tolerance, 
                     no_change_steps=PGD_no_change_steps, 
                     optimizer_class=torch.optim.Adam, 
                     verbose=PGD_verbose)
    X_clean = data_dict["X"]
    X_ref = X_clean.clone()
    reference_data = {"X": X_ref}
    
    X_data = torch.tensor(X_clean + PGD_init_noise * torch.randn_like(X_clean), requires_grad=True, dtype=torch.float32)
    X_data.requires_grad = True
    data = {"X": X_data}
    # Run optimization
    start_time = time.time()  # Start the clock
    optimized_data = pgd_optimizer.minimize_kl(data_dict=data, 
                        reference_data_dict=reference_data,
                        constraints=constraints_pgd,
                        random_init=False,
                        init_range=(-1.0, 1.0)
    )
    end_time = time.time()  # Stop the clock
    elapsed_time = round(end_time - start_time,precision)  # Calculate the elapsed time

    X=optimized_data["X"].detach().cpu().numpy()
    Xbar=np.mean(X,axis=0)
    μ_poisoned=Σ_post @ (Λ_prior @ μ_prior+N*Λ_data_generating @ Xbar)
    #print(f"μ_poisoned={μ_poisoned}")
    δI_poisoned=round(kl_bivariate_normal_normal(μ_tgt,Λ_tgt,μ_poisoned,Λ_post,P),precision)
    #print(f"δI(X)={δI_poisoned}")
    #print(f"time={elapsed_time} sec")
    title='PGD2'
    plot_bivariate_normals_with_poisoningXpX(i,b1_name,b2max,b3max,μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,μ_poisoned,title,Xp,X)
    δI_poisoned_list[f"{title} KL"]=float(δI_poisoned)
    solve_time_list[f"{title} time"]=elapsed_time
    (num_changed_rows,num_changed,Eucl_dist)=compute_change_metrics(Xp,X)
    num_changed_rows_list[f"{title} rows chng"]=num_changed_rows
    num_changed_list[f"{title} entries chng"]=num_changed
    Eucl_dist_list[f"{title} L2"]=float(round(Eucl_dist,precision))
    
    #print('******************************')
    print('ε-SCD2 Results')
    data_dict, model, clean_posterior, target_posterior, _, constraints_iscd = generate_config(μ_prior,Σ_prior,Σ_data_generating,μ_tgt,Σ_tgt,Xp,b1,b2,b3)
    X_clean = data_dict["X"]
    X_clean.requires_grad = True
    reference_data = {"X": X_clean}
    # Initialize PGD optimizer
    fgsm_optimizer = ISCD(
        model=model,
        target_posterior=target_posterior,
        kl_direction=kl_function,
        epsilon=SCD_epsilon, 
        constraints=constraints_iscd,
        max_oscillations=SCD_max_oscillations,
        second_order=SCD_second_order,
        verbose=SCD_verbose
    )
    # Run optimization
    start_time = time.time()  # Start the clock
    optimized_data = fgsm_optimizer.minimize_kl(reference_data_dict=reference_data)
    end_time = time.time()  # Stop the clock
    elapsed_time = round(end_time - start_time,precision)  # Calculate the elapsed time

    X=optimized_data["X"].detach().cpu().numpy()
    Xbar=np.mean(X,axis=0)
    μ_poisoned=Σ_post @ (Λ_prior @ μ_prior+N*Λ_data_generating @ Xbar)
    #print(f"μ_poisoned={μ_poisoned}")
    δI_poisoned=round(kl_bivariate_normal_normal(μ_tgt,Λ_tgt,μ_poisoned,Λ_post,P),precision)
    #print(f"δI(X)={δI_poisoned}")
    #print(f"time={elapsed_time} sec")
    title='ε-SCD2'
    plot_bivariate_normals_with_poisoningXpX(i,b1_name,b2max,b3max,μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,μ_poisoned,title,Xp,X)
    δI_poisoned_list[f"{title} KL"]=float(δI_poisoned)
    solve_time_list[f"{title} time"]=elapsed_time   
    (num_changed_rows,num_changed,Eucl_dist)=compute_change_metrics(Xp,X)
    num_changed_rows_list[f"{title} rows chng"]=num_changed_rows
    num_changed_list[f"{title} entries chng"]=num_changed
    Eucl_dist_list[f"{title} L2"]=float(round(Eucl_dist,precision))
    '''
    #print('******************************')
    print('FGA2 Results')
    solution_FGA2=FGA2(N,P,Xp,b1,b2,b3,μ_prior,Λ_prior,Λ_data_generating,Λ_post,μ_tgt,Λ_tgt,precision)
    # Output optimized matrix and function history
    X=solution_FGA2['X']
    Xbar=np.mean(X,axis=0)
    μ_poisoned=Σ_post @ (Λ_prior @ μ_prior+N*Λ_data_generating @ Xbar)
    #print(f"μ_poisoned={μ_poisoned}")
    δI_poisoned=round(kl_bivariate_normal_normal(μ_tgt,Λ_tgt,μ_poisoned,Λ_post,P),precision)
    #print(f"δI(X)={δI_poisoned}")
    #print('time=',solution_FGA2['time'],'sec')
    title='FGA2'
    #plot_bivariate_normals_with_poisoning(μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,μ_poisoned,title)
    plot_bivariate_normals_with_poisoningXpX(i,b1_name,b2max,b3max,μ_prior,Σ_prior,μ_post,Σ_post,μ_tgt,Σ_tgt,μ_poisoned,title,Xp,X)
    δI_poisoned_list[f"{title} KL"]=float(δI_poisoned)
    solve_time_list[f"{title} time"]=solution_FGA2['time']
    (num_changed_rows,num_changed,Eucl_dist)=compute_change_metrics(Xp,X)
    num_changed_rows_list[f"{title} rows chng"]=num_changed_rows
    num_changed_list[f"{title} entries chng"]=num_changed
    Eucl_dist_list[f"{title} L2"]=float(round(Eucl_dist,precision))
    
    #print('******************************')    
    #print(δI_poisoned_list)
    #print(solve_time_list)  
    results={
        'N':N,
        'P':P,
        'b1 rho':b1_rho,
        'b2 max':b2max,
        'b3 max':b3max,        
        'iter':i+1,
        'prior mean':μ_prior,
        'prior cov':Σ_prior,
        'data_generating mean':μ_data_generating,
        'data_generating cov':Σ_data_generating,
        'X_data':Xp,
        'Xbar_data':Xbar_data,
        'b1':b1,
        'b2':b2,
        'b3':b3,
        'tgt mean':μ_tgt,
        'tgt cov':Σ_tgt,
        'KL_poisoned_list':δI_poisoned_list,
        'solve_time_list':solve_time_list,
        'num_changed_rows_list':num_changed_rows_list,
        'num_changed_list':num_changed_list,
        'Eucl_dist_list':Eucl_dist_list
        }
    # flatten the nested dictionaries
    δE_poisoned_list = results.pop("KL_poisoned_list")
    results.update(δE_poisoned_list)
    solve_time_data = results.pop("solve_time_list")
    results.update(solve_time_data)
    num_changed_rows_list = results.pop("num_changed_rows_list")
    results.update(num_changed_rows_list)
    num_changed_list = results.pop("num_changed_list")
    results.update(num_changed_list)
    Eucl_dist_list = results.pop("Eucl_dist_list")
    results.update(Eucl_dist_list)
    return results

# MAIN PROGRAM BEGINS HERE 
# Specify user-determined parameters
precision=6 # number of decimal places for selected calculations
N=100 # of independent observations, indexed on n
P=2 # of elements within an observation, indexed on p
b1_rho=0.5 # approximate percentage of elements that can change
b1_name=str(round(b1_rho*100,0)) # Used for file creation
b2max=2 
b3max=3 # maximum magnitude of any change in an entry (n,p); for now, we've assumed it's 3, for simplicity
k = 30 # iterations

# Solver related parameters 
MP_time_limit=15 # time limit on all solvers for any math program
QA_max_time=60 # max time for Q2L process
QA_max_iterations=10
QA_termination_tolerance=0 # terminate based on lack of eps improvement in objective function value between iterations

# Roi-coded method parameters
kl_function = "inclusive"
SCD_epsilon = 0.001
SCD_verbose = False    
SCD_max_oscillations=10
SCD_second_order=False

# PGD Parameters
PGD_lr = 0.01
PGD_init_noise = 0.1
PGD_max_iter = 50000
PGD_verbose = False
PGD_tolerance = 1e-10  # Stopping criterion: minimum change in KL loss
PGD_no_change_steps = 5000  # Maximum number of steps without significant change before stopping

b1_rho_values=[0.33,0.67,1]
b2max_values=[1,2,3]
b3max_values=[1,3,5]

for b1_rho_counter,b1_rho in enumerate(b1_rho_values, 1):
    b1_name=str(round(b1_rho*100,0)) # Used for file creation
    for b2max_counter,b2max in enumerate(b2max_values,1):  
        for b3max_counter,b3max in enumerate(b3max_values,1):      
            stored_aggregate_results={}
            for i in range(k):
                ### Generate the instance data ###
                (μ_prior,Σ_prior,μ_data_generating,Σ_data_generating,μ_tgt,Σ_tgt,Xp,Xbar_data,b1,b2,b3)=generate_bivariate_instance(N,P,i,b1_rho,b2max,b3max,precision)
                #print(f"*** Iteration {i+1} with μ_prior={μ_prior} ***")
                Λ_prior=np.linalg.inv(Σ_prior)
                Λ_data_generating=np.linalg.inv(Σ_data_generating)
                Λ_tgt=np.linalg.inv(Σ_tgt)
                    
                ### Run the solution methods ###
                iteration_results=run_methods(i,b1_rho,b1_name,b2max,b3max,μ_prior,Σ_prior,Λ_prior,N,P,μ_data_generating,Σ_data_generating,Λ_data_generating,Xp,Xbar_data,b1,b2,b3,μ_tgt,Σ_tgt,Λ_tgt,MP_time_limit,QA_max_time,QA_max_iterations,QA_termination_tolerance,kl_function,SCD_epsilon,SCD_verbose,SCD_max_oscillations,SCD_second_order,PGD_lr,PGD_init_noise,PGD_max_iter,PGD_verbose,PGD_tolerance,PGD_no_change_steps)
                stored_aggregate_results[i]=iteration_results 
                
            # Save the aggregate results to a file
            file_path_aggregate=f"C:/Users/brian/Documents/PA/NN_IPA_N{N}_b1_{b1_name}_b2_{b2max}_b3_{b3max}_k{k}.csv"
            dump_aggregate_data(stored_aggregate_results,file_path_aggregate)