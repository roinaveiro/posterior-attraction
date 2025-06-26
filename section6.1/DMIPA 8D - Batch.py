import numpy as np
from scipy.special import digamma,gammaln
#from scipy.special import gamma # used in previous function 'dirichlet_kl_divergence' that would call gamma directly and often overload
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
import time
import pyomo.environ as pyo
import copy
import csv
import seaborn as sns

def plot_heatmap(x,title,i):
    """
    Plots a heatmap for a 10x8 NumPy array, standardized with a color scale from -10 to 10.
    Rows and columns are labeled starting from 1 instead of 0.

    Parameters:
    x (np.ndarray): A 10x8 NumPy array.
    """
    if i>9:
        return
    if x.shape != (10, 8):
        raise ValueError("Input array must be of shape (10, 8)")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        x,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        vmin=-3,  # use -(b3max-1)
        vmax=3, # use b3max-1
        xticklabels=np.arange(1, 9),  # Columns 1 to 8
        yticklabels=np.arange(1, 11)  # Rows 1 to 10
    )
    plt.title(f"Changes to X' by {title}")
    plt.xlabel("Data Point (p)")
    plt.ylabel("Observation (n)")
    plt.tight_layout()
    plt.savefig(f"C:/Users/brian/Documents/PA/Heatmaps/DM_IPA_iter{i+1}_delta_X_{title}.png", dpi=300, bbox_inches='tight')    
    plt.show()
    
def generate_dirichlet_instance(N,P,i,R,concentration_parameter,density_of_b1,b2maxfactor,b3max):
    np.random.seed(N+P+i+R) # fix the pseudorandom seed,if desired
    α_prior=np.random.permutation(np.arange(1,P+1))
    p=np.random.dirichlet([concentration_parameter] * P)
    Xp=np.array([np.random.multinomial(R,p) for n in range(N)])
    α_post=α_prior+np.sum(Xp, axis=0)
    α_tgt=create_alpha_tgt(α_post)
    b1=np.minimum(N*P,round(density_of_b1*N*P,0))
    b2max=max(1,round(b2maxfactor*b1/N,0)) ### maximum # of changes in each row n, assumes 1.5 multiplier so either b1 or b_2n-values might limit poisoning solution
    b2=np.zeros(N,dtype=int)
    while np.sum(b2)==0:  # Ensure sum(b2) > 0
        b2 = np.minimum(P,np.random.randint(0,b2max+1,size=N))  # Generate N integers in range [0,b2max]
    b3=np.random.randint(0,b3max,size=(N,P))
    return (Xp,b1,b2,b3,α_prior,α_tgt)

def random_integer_partition_strictly_positive(total, parts):
    """
    Randomly partition a strictly positive integer `total` into `parts` strictly positive integers
    that sum exactly to `total`.
    """
    # We subtract `parts` from the total to ensure each part has at least 1
    total_adjusted = total - parts
    # Partition the adjusted total into `parts` non-negative integers
    partition = random_integer_partition_nonnegative(total_adjusted, parts)
    # Add 1 to each part to ensure they are strictly positive
    return partition + 1

def random_integer_partition_nonnegative(total, parts):
    """
    Randomly partition a non-negative integer `total` into `parts` non-negative integers
    that sum exactly to `total`.
    """
    if parts == 1:
        return np.array([total])
    # Choose `parts - 1` unique cut points from range(1, total + parts)
    cuts = np.sort(np.random.choice(range(1, total + parts), parts - 1, replace=False))
    full_cuts = np.concatenate(([0], cuts, [total + parts]))
    return full_cuts[1:] - full_cuts[:-1] - 1

def create_alpha_tgt(alpha_post):
    alpha_post = np.array(alpha_post, dtype=int)
    K = len(alpha_post)
    # Step 1: Compute the total sum of the first 3 entries
    total = int(np.sum(alpha_post[:3]))
    # Step 2: Generate new first 3 entries that are strictly positive
    new_first3 = random_integer_partition_strictly_positive(total, 3)
    # Step 3: Construct the new vector
    if K == 3:
        alpha_tgt = new_first3
    else:
        alpha_tgt = np.concatenate([new_first3, alpha_post[3:]])
    return alpha_tgt

def dirichlet_kl_divergence(alpha, α_given):
    # Compute the α_given functions B(α) and B(α_given)
    alpha_0 = np.sum(alpha)
    α_given_0 = np.sum(α_given)
    term1 = gammaln(alpha_0) - np.sum(gammaln(alpha))
    term2 = np.sum(gammaln(α_given)) - gammaln(α_given_0)
    term3 = np.sum((alpha - α_given) * (digamma(alpha) - digamma(alpha_0)))
    return term1 + term2 + term3

def kl_dirichlet_multinomial_given_X(α,X,α_given):
    return dirichlet_kl_divergence(α_given,α+np.sum(X,axis=0)) # Reversed order for IPA ###

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

def plot_dirichlet_distributions(α_prior, α_post, α_tgt,i):
    if i>9:
        return
    def format_alpha(alpha):
        return r"$\alpha = [" + ", ".join(map(str, alpha)) + "]$"

    alphas = [α_prior, α_post, α_tgt]
    labels = [
        f"Marginal Prior {format_alpha(α_prior)}",
        f"Marginal Posterior {format_alpha(α_post)}",
        f"Marginal Target {format_alpha(α_tgt)}",
    ]
    cmap = plt.cm.coolwarm

    def barycentric_to_cartesian(x, y, z):
        return 0.5 * (2 * x + y), (np.sqrt(3) / 2) * y

    def create_simplex_grid(num_points=500):
        x = np.linspace(0, 1, num_points)
        y = np.linspace(0, 1, num_points)
        X, Y = np.meshgrid(x, y)
        Z = 1 - X - Y
        mask = (Z >= 0) & (Z <= 1)
        return X[mask], Y[mask], Z[mask]

    X, Y, Z = create_simplex_grid()
    U, V = barycentric_to_cartesian(X, Y, Z)
    points = np.stack([X, Y, Z], axis=1)

    for alpha, label in zip(alphas, labels):
        density = dirichlet.pdf(points.T, alpha)

        fig, ax = plt.subplots(figsize=(5, 5))
        contour = ax.tricontourf(U, V, density, levels=100, cmap=cmap)
        ax.set_title(label)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, np.sqrt(3) / 2 + 0.05)
        ax.set_aspect("equal")
        ax.axis('off')

        triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
        ax.plot(triangle[:, 0], triangle[:, 1], 'k-', lw=1)

        cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label("Density")
        plt.savefig(f"C:/Users/brian/Documents/PA/Plots/DM_IPA_iter{i+1}_{label[9:14]}.png", dpi=300, bbox_inches='tight')     
        plt.show()

def plot_poisoned_dirichlet_distributions(α_poisoned, title,i):
    if i>9:
        return
    def format_alpha(alpha):
        return r"$\alpha = [" + ", ".join(map(str, alpha)) + "]$"
    alphas = [α_poisoned]
    labels = [
        f"Marginal {title} Poisoned {format_alpha(α_poisoned)}"
    ]
    cmap = plt.cm.coolwarm

    def barycentric_to_cartesian(x, y, z):
        return 0.5 * (2 * x + y), (np.sqrt(3) / 2) * y

    def create_simplex_grid(num_points=500):
        x = np.linspace(0, 1, num_points)
        y = np.linspace(0, 1, num_points)
        X, Y = np.meshgrid(x, y)
        Z = 1 - X - Y
        mask = (Z >= 0) & (Z <= 1)
        return X[mask], Y[mask], Z[mask]

    X, Y, Z = create_simplex_grid()
    U, V = barycentric_to_cartesian(X, Y, Z)
    points = np.stack([X, Y, Z], axis=1)

    for alpha, label in zip(alphas, labels):
        density = dirichlet.pdf(points.T, alpha)

        fig, ax = plt.subplots(figsize=(5, 5))
        contour = ax.tricontourf(U, V, density, levels=100, cmap=cmap)
        ax.set_title(label)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, np.sqrt(3) / 2 + 0.05)
        ax.set_aspect("equal")
        ax.axis('off')

        triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
        ax.plot(triangle[:, 0], triangle[:, 1], 'k-', lw=1)

        cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label("Density")
        plt.savefig(f"C:/Users/brian/Documents/PA/Plots/DM_IPA_iter{i+1}_{title}.png", dpi=300, bbox_inches='tight')     
        plt.show()

def approximate_gradient(α_prior, Xp, α_tgt, precision):
    X = np.array(Xp, dtype=float)
    N, P = X.shape
    gradient = np.zeros_like(X)
    epsilon = 0.5#1e-5  # Small perturbation for numerical stability

    # Compute KL divergence at the original X once
    kl_base = kl_dirichlet_multinomial_given_X(α_prior, X, α_tgt)

    for n in range(N):
        for p in range(P):
            X_perturbed = X.copy()
            X_perturbed[n, p] += epsilon  # Small perturbation
            kl_perturbed = kl_dirichlet_multinomial_given_X(α_prior, X_perturbed, α_tgt)
            gradient[n, p] = (kl_perturbed - kl_base) / epsilon  # Finite difference approximation

    return np.round(gradient, precision)

def approximate_hessian(α_prior, Xp, α_tgt, precision):
    X = np.array(Xp, dtype=float)
    N, P = X.shape
    H = np.zeros((N, P, N, P))  # Hessian tensor
    epsilon = 0.5#1e-5  # Small perturbation for numerical stability

    # Compute KL divergence at the unperturbed X
    kl_base = kl_dirichlet_multinomial_given_X(α_prior, X, α_tgt)

    for n in range(N):
        for p in range(P):
            for k in range(N):
                for l in range(P):
                    # Create perturbed versions of X
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
                    kl_np_plus = kl_dirichlet_multinomial_given_X(α_prior, X_np_plus, α_tgt)
                    kl_np_minus = kl_dirichlet_multinomial_given_X(α_prior, X_np_minus, α_tgt)
                    kl_kl_plus = kl_dirichlet_multinomial_given_X(α_prior, X_kl_plus, α_tgt)
                    kl_kl_minus = kl_dirichlet_multinomial_given_X(α_prior, X_kl_minus, α_tgt)
                    kl_np_kl_plus = kl_dirichlet_multinomial_given_X(α_prior, X_np_kl_plus, α_tgt)
                    kl_np_kl_minus = kl_dirichlet_multinomial_given_X(α_prior, X_np_kl_minus, α_tgt)

                    # Compute second derivative using central difference approximation
                    H[n, p, k, l] = (kl_np_kl_plus - kl_np_plus - kl_kl_plus + 2 * kl_base - kl_np_minus - kl_kl_minus + kl_np_kl_minus) / (epsilon ** 2)

    return np.round(H, precision)

def solve_Q2L1_for_Q2L(N,P,Xest,α_prior,Xprime,b1,b2,b3,α_tgt,precision,MP_time_limit,solver_name):
    start_time = time.time() # Start the clock
    ### The next two lines construct a quadratic approximation based on the point Xest rather than Xp ###
    grad_list = approximate_gradient(α_prior,Xest,α_tgt,precision)
    H_list = approximate_hessian(α_prior,Xest,α_tgt,precision)
    # Create a Pyomo model
    model = pyo.ConcreteModel()
    # Sets
    model.N = pyo.Set(initialize=create_list_from_integer(N))
    model.P = pyo.Set(initialize=create_list_from_integer(P))
    # Parameters
    model.Xp=pyo.Param(model.N,model.P,initialize=matrix_to_dict(Xprime))
    model.Xest=pyo.Param(model.N,model.P,initialize=matrix_to_dict(Xest))
    model.grad=pyo.Param(model.N,model.P,initialize=matrix_to_dict(grad_list))
    model.H=pyo.Param(model.N,model.P,model.N,model.P,initialize=matrix_of_matrices_to_dict(H_list,N,P),domain=pyo.Reals)
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
    model.f=pyo.Param(initialize=kl_dirichlet_multinomial_given_X(α_prior,Xest,α_tgt))
    # Decision Variables
    def initialize_X(model,n,p):
        return model.Xest[n,p]
    model.X=pyo.Var(model.N,model.P,domain=pyo.NonNegativeIntegers,initialize=initialize_X) 
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
            model.DVbounds.add(model.X[n,p]>=0)
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
            'X':[[int(pyo.value(model.X[n,p])) for p in model.P] for n in model.N],
            'Ω':[[int(pyo.value(model.Ω[n,p])) for p in model.P] for n in model.N],
            'q(X)':round(pyo.value(model.objfnvalue),precision),
            'time':round(elapsed_time,precision)
            }
    else:
        solution={
            'X':[[int(pyo.value(model.Xest[n,p])) for p in model.P] for n in model.N],
            'Ω':[[int(0) for p in model.P] for n in model.N],
            'q(X)':999999,
            'time':round(elapsed_time,precision)
            }
    return solution

def solve_Q2L(N,P,α_prior,Xprime,b1,b2,b3,α_tgt,precision,MP_time_limit,solver_name,QA_max_time,QA_max_iterations,QA_termination_tolerance):
    stored_solutions={}
    k_iter=1
    termination_check=QA_termination_tolerance+1
    Xest=Xp #construct quadratic approximation about the point Xest
    f_k=1 #temporary initialization to define f_k
    start_time = time.time() # Start the clock
    elapsed_time=0
    
    while (k_iter<=QA_max_iterations and elapsed_time<QA_max_time and termination_check>QA_termination_tolerance):
        solution_Q2L1=solve_Q2L1_for_Q2L(N,P,Xest,α_prior,Xprime,b1,b2,b3,α_tgt,precision,MP_time_limit,solver_name)
        stored_solutions[k_iter]=solution_Q2L1
        if k_iter==1: 
            f_kminus1=kl_dirichlet_multinomial_given_X(α_prior,solution_Q2L1['X'],α_tgt)+QA_termination_tolerance+1 
        else:
            f_kminus1=f_k
        f_k=kl_dirichlet_multinomial_given_X(α_prior,solution_Q2L1['X'],α_tgt)
        termination_check=f_kminus1-f_k
        if f_k<f_kminus1: # only accept a better solution
            print(f"-Accepting a better solution on iteration {k_iter}")
            Xest=solution_Q2L1['X']
            solution=solution_Q2L1
        else:
            print(f"-Rejecting a worse solution on iteration {k_iter}")
        k_iter=k_iter+1
        end_time = time.time() # Stop the clock
        elapsed_time = end_time - start_time # Calculate the elapsed time    
    # make sure the reported time is the cumulative time,not the time of the last Q2L1 solution
    solution['time']=round(elapsed_time,precision)
    solution['iterations']=k_iter-1
    return (solution,stored_solutions)   

def solve_2M2(N,P,α_prior,Xprime,b1,b2,b3,α_tgt,precision,MP_time_limit,solver_name):
    start_time = time.time() # Start the clock
    # Create a Pyomo model
    model = pyo.ConcreteModel()
    # Sets
    model.N = pyo.Set(initialize=create_list_from_integer(N))
    model.P = pyo.Set(initialize=create_list_from_integer(P))
    # Parameters
    model.α_prior=pyo.Param(model.P,initialize=list_to_dict(α_prior))
    model.α_tgt=pyo.Param(model.P,initialize=list_to_dict(α_tgt))
    model.Xp=pyo.Param(model.N,model.P,initialize=matrix_to_dict(Xprime))
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
    model.X=pyo.Var(model.N,model.P,domain=pyo.NonNegativeIntegers,initialize=initialize_X) 
    model.Ω=pyo.Var(model.N,model.P,domain=pyo.Binary,initialize=0) # 1 of |X(n,p)-X'(n,p)|>0 and 0 otherwise
    # Formulate model
    # Objective function 
    model.objfnvalue = pyo.Objective(expr =sum((model.α_prior[p]+sum(model.X[n,p] for n in model.N)-model.α_tgt[p])**2 for p in model.P),sense=pyo.minimize)
    
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
            model.DVbounds.add(model.X[n,p]<=model.BigM)  #bounds that will not cut off an optimal solution
            model.DVbounds.add(model.X[n,p]>=0)
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
            'X':[[int(pyo.value(model.X[n,p])) for p in model.P] for n in model.N],
            'Ω':[[int(pyo.value(model.Ω[n,p])) for p in model.P] for n in model.N],
            '2M(X)':round(pyo.value(model.objfnvalue),precision),           
            'time':round(elapsed_time,precision)
            }    
    else:
        solution={
            'X':[[int(pyo.value(model.Xp[n,p])) for p in model.P] for n in model.N],
            'Ω':[[int(0) for p in model.P] for n in model.N],
            '2M(X)':999999,           
            'time':round(elapsed_time,precision)
            }    
    return solution

def FGA2(N,P,α_prior,Xp,b1,b2,b3,α_tgt,precision):
    start_time = time.time()  # Start the clock
    Xprime = copy.deepcopy(Xp)
    X = Xprime
    N,P = len(X),len(X[0])  # Dimensions of the matrix
    modified_entries = set()  # To track which entries have been modified
    row_changes = [0 for _ in range(N)]  # Track row-specific changes
    # Compute the gradient
    grad = approximate_gradient(α_prior,X,α_tgt,precision)
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
        X[i][j] = max(0, X[i][j] + delta) # Ensure X[k] is non-negative after update
    end_time = time.time()  # Stop the clock
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    solution = {
        'X':[[int(X[n][p]) for p in range(P)] for n in range(N)],
        'δE(X)': float(kl_dirichlet_multinomial_given_X(α_prior,X,α_tgt)),
        'time': round(elapsed_time,precision)
    }

    return solution

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

def run_methods(α_prior,N,P,i,R,Xp,b1,b2,b3,α_tgt,precision,MP_time_limit,QA_max_time,QA_max_iterations,QA_termination_tolerance):
    solvers={'CPLEX'}#{'CPLEX','SCIP','MindtPy'}    
    print('******************************')
    print(f"Iteration {i+1}")
    print(f"α_prior={α_prior}")
    #print(f"X'={Xp}")
    α_post=α_prior+np.sum(Xp, axis=0)
    print(f"α_post={α_post}")
    δ_kl_in = round(dirichlet_kl_divergence(α_tgt,α_post),precision)
    print(f"α_tgt={α_tgt}")
    print(f"δI(X')={δ_kl_in}")    
    α_prior3 = α_prior[:3]
    α_post3 = α_post[:3]
    α_tgt3 = α_tgt[:3]
    plot_dirichlet_distributions(α_prior3,α_post3,α_tgt3,i)
    
    δI_poisoned_list={}
    delta_X_list={}
    α_poisoned_list={}
    α_last5_poisoned_indicator_list={} # Equal 1 if any of last 5 elements are perturbed, 0 otherwise
    α_poisoned_percentage_list={} # proportion of changes made to last 5 elements
    solve_time_list={}
    δI_poisoned_list["Unpoisoned KL"]=δ_kl_in    

    for solver_name in solvers: 
        print('******************************')
        print(f"Q2L Results with {solver_name}")
        (solution_Q2L,Q2L_trace)=solve_Q2L(N,P,α_prior,Xp,b1,b2,b3,α_tgt,precision,MP_time_limit,solver_name,QA_max_time,QA_max_iterations,QA_termination_tolerance)
        X=np.array(solution_Q2L['X'])
        #print(f"X={X}")
        α_poisoned=α_prior+np.sum(X,axis=0)
        print(f"α_poisoned={α_poisoned}")
        δI_poisoned=round(dirichlet_kl_divergence(α_tgt,α_poisoned),precision)
        print(f"δI(X)={δI_poisoned}")
        print(f"time={solution_Q2L['time']} sec")
        title='Q2L-'+solver_name[0]
        α_poisoned3 = α_poisoned[:3]
        plot_poisoned_dirichlet_distributions(α_poisoned3, title,i)
        δI_poisoned_list[f"{title} KL"]=float(δI_poisoned)
        delta_X_list[f"{title} delta X"]=X-Xp
        plot_heatmap(X-Xp,title,i)
        α_poisoned_list[f"{title} alpha"]=α_poisoned
        if P>3:
            α_last5_poisoned_indicator_list[f"{title} alpha_5_indicator"]=int(np.any(α_poisoned[4:] != α_post[4:]))
            α_temp=np.abs(α_poisoned-α_post)
            α_poisoned_percentage_list[f"{title} alpha_5_percent"]=np.sum(α_temp[4:]) / np.sum(α_temp) if  np.sum(α_temp) != 0 else 0
        solve_time_list[f"{title} time"]=solution_Q2L['time']

    for solver_name in solvers: 
        print('******************************')
        print(f"2M2 Results with {solver_name}")
        solution_2M2=solve_2M2(N,P,α_prior,Xp,b1,b2,b3,α_tgt,precision,MP_time_limit,solver_name)
        X=np.array(solution_2M2['X'])
        #print(f"X={X}")
        α_poisoned=α_prior+np.sum(X,axis=0)
        print(f"α_poisoned={α_poisoned}")
        δI_poisoned=round(dirichlet_kl_divergence(α_tgt,α_poisoned),precision)
        print(f"δI(X)={δI_poisoned}")
        print(f"time={solution_2M2['time']} sec")
        title='2M2-'+solver_name[0]
        α_poisoned3 = α_poisoned[:3]
        plot_poisoned_dirichlet_distributions(α_poisoned3, title,i)
        δI_poisoned_list[f"{title} KL"]=float(δI_poisoned)
        delta_X_list[f"{title} delta X"]=X-Xp
        plot_heatmap(X-Xp,title,i)
        α_poisoned_list[f"{title} alpha"]=α_poisoned
        if P>3:
            α_last5_poisoned_indicator_list[f"{title} alpha_5_indicator"]=int(np.any(α_poisoned[4:] != α_post[4:]))
            α_temp=np.abs(α_poisoned-α_post)
            α_poisoned_percentage_list[f"{title} alpha_5_percent"]=np.sum(α_temp[4:]) / np.sum(α_temp) if  np.sum(α_temp) != 0 else 0
        solve_time_list[f"{title} time"]=solution_2M2['time']
        
    print('******************************')
    print('FGA2 Results')
    solution_FGA2=FGA2(N,P,α_prior,Xp,b1,b2,b3,α_tgt,precision)
    X=np.array(solution_FGA2['X'])
    #print(f"X={X}")
    α_poisoned=α_prior+np.sum(X,axis=0)
    print(f"α_poisoned={α_poisoned}")
    δI_poisoned=round(dirichlet_kl_divergence(α_tgt,α_poisoned),precision)
    print(f"δI(X)={δI_poisoned}")
    print(f"time={solution_FGA2['time']} sec")
    title='FGA2'
    α_poisoned3 = α_poisoned[:3]
    plot_poisoned_dirichlet_distributions(α_poisoned3, title,i)
    δI_poisoned_list[f"{title} KL"]=float(δI_poisoned)
    delta_X_list[f"{title} delta X"]=X-Xp
    plot_heatmap(X-Xp,title,i)
    α_poisoned_list[f"{title} alpha"]=α_poisoned
    if P>3:
        α_last5_poisoned_indicator_list[f"{title} alpha_5_indicator"]=int(np.any(α_poisoned[4:] != α_post[4:]))
        α_temp=np.abs(α_poisoned-α_post)
        α_poisoned_percentage_list[f"{title} alpha_5_percent"]=np.sum(α_temp[4:]) / np.sum(α_temp) if  np.sum(α_temp) != 0 else 0
    solve_time_list[f"{title} time"]=solution_FGA2['time']
    
    results={
        'N':N,
        'P':P,   
        'iter':i+1,
        'X_data':Xp,
        'b1':b1,
        'b2':b2,
        'b3':b3,
        'alpha_prior':α_prior,
        'alpha_post':α_post,
        'alpha_tgt':α_tgt,
        'KL_poisoned':δI_poisoned_list,
        'delta_X':delta_X_list,
        'alpha_poisoned':α_poisoned_list,
        'alpha_last5_indicator':α_last5_poisoned_indicator_list,
        'alpha_poisoned_percentage':α_poisoned_percentage_list,
        'solve_time':solve_time_list
        }
    # flatten the nested dictionaries
    δI_poisoned_list = results.pop("KL_poisoned")
    results.update(δI_poisoned_list)
    delta_X_list = results.pop("delta_X")
    results.update(delta_X_list)
    α_poisoned_list = results.pop("alpha_poisoned")
    results.update(α_poisoned_list)
    α_last5_poisoned_indicator_list = results.pop("alpha_last5_indicator")
    results.update(α_last5_poisoned_indicator_list)
    α_poisoned_percentage_list = results.pop("alpha_poisoned_percentage")
    results.update(α_poisoned_percentage_list)
    solve_time_data = results.pop("solve_time")
    results.update(solve_time_data)
   
    return results

# MAIN PROGRAM BEGINS HERE
# Specify user-determined parameters
precision=6 # number of decimal places for selected calculations
N=10 # Number of observations
P=8 # Number of categories for the Dirichlet distribution
R=1 # Number of replications within a single multinomial observations (R choose P with pmf p)
concentration_parameter=0.25 # concentration parameter for generating a pmf p, where >>1 is close to a uniform distribution
k = 100 # iterations

# Poisoning-related parameters
density_of_b1=0.75 # approximate percentage of elements that can change
b2maxfactor=2 # when generating b2-values,imposes allows up to round(b2maxfactor*b1/K,0) or 1 change,whichever is larger
b3max=4 # maximum magnitude of any change in an entry k; for now,we've assumed it's 5,for simplicity

# Solver related parameters 
MP_time_limit=15 # time limit on all solvers for any math program
QA_max_time=120 # max time for Q2L process
QA_max_iterations=20
QA_termination_tolerance=0 # terminate based on lack of eps improvement in objective function value between iterations

stored_aggregate_results={}
for i in range(k):
    ### Generate the instance data ###
    (Xp,b1,b2,b3,α_prior,α_tgt)=generate_dirichlet_instance(N,P,i,R,concentration_parameter,density_of_b1,b2maxfactor,b3max)
    iteration_results=run_methods(α_prior,N,P,i,R,Xp,b1,b2,b3,α_tgt,precision,MP_time_limit,QA_max_time,QA_max_iterations,QA_termination_tolerance)
    stored_aggregate_results[i]=iteration_results 
    
# Save the aggregate results to a file
file_path_aggregate=f"C:/Users/brian/Documents/PA/DM_IPA_N{N}_P{P}_k{k}.csv"
dump_aggregate_data(stored_aggregate_results,file_path_aggregate)