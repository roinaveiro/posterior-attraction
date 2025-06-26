import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D



def plot_location_changes(
    original_locations: torch.Tensor,
    new_locations: torch.Tensor,
    change_threshold: float = 1e-7,
    draw_lines: bool = True,
    figsize: tuple = (8, 6),
    text: bool = False,
    save_path: str = None
):
    """
    Plot a comparison of original vs. new 2D coordinates with index labels and visual markers
    for changes, using seaborn styling and high-quality settings.

    Args:
        original_locations (Tensor): Shape (n, 2). The original 2D points.
        new_locations (Tensor): Same shape (n, 2). The updated 2D points.
        change_threshold (float): L1 distance threshold to consider a point as changed.
        draw_lines (bool): Draw dashed lines from old to new locations for changed points.
        figsize (tuple): Size of the matplotlib figure.
        save_path (str): Optional path to save the figure in high resolution.
    """

    sns.set(style="whitegrid")
    
    # Prepare data
    orig = original_locations.detach().cpu()
    new = new_locations.detach().cpu()

    moved = (orig - new).abs().sum(dim=1) > change_threshold
    stayed = ~moved

    # Initialize figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot unchanged points
    sns.scatterplot(
        x=orig[stayed, 0].numpy(),
        y=orig[stayed, 1].numpy(),
        color="#9e9e9e",
        alpha=0.6,
        s=80,
        label="Unchanged",
        ax=ax
    )
    for i in torch.where(stayed)[0]:
        if text:
            ax.text(
                orig[i, 0].item(),
                orig[i, 1].item(),
                str(i.item()),
                fontsize=10,
                color="gray",
                ha="center",
                va="bottom"
            )

    # Plot changed points
    if moved.any():
        sns.scatterplot(
            x=orig[moved, 0].numpy(),
            y=orig[moved, 1].numpy(),
            color="#0072B2",
            alpha=0.6,
            marker="o",
            s=100,
            label="Changed (original)",
            ax=ax
        )
        sns.scatterplot(
            x=new[moved, 0].numpy(),
            y=new[moved, 1].numpy(),
            color="#D55E00",
            alpha=0.9,
            marker="X",
            s=100,
            label="Changed (new)",
            ax=ax
        )
        for i in torch.where(moved)[0]:
            if text:
                ax.text(
                    orig[i, 0].item(),
                    orig[i, 1].item(),
                    str(i.item()),
                    fontsize=10,
                    color="#0072B2",
                    ha="center",
                    va="bottom"
                )
        if draw_lines:
            for i in torch.where(moved)[0]:
                ax.plot(
                    [orig[i, 0], new[i, 0]],
                    [orig[i, 1], new[i, 1]],
                    color="#E69F00",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.6
                )

    # Final touches
    ax.set_title("Change in 2D Locations", fontsize=18, fontweight='bold')
    ax.set_xlabel("X Coordinate", fontsize=16)
    ax.set_ylabel("Y Coordinate", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=12, frameon=True)
    sns.despine(trim=True)
    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig, ax



def visualize_posteriors_and_data_2D(clean_posterior, attacked_posterior, target_posterior, clean_data, attacked_data):
    """
    Visualize the densities of the posteriors and the evolution of data points from clean to attacked.

    Args:
        clean_posterior: Torch distribution representing the posterior under clean data.
        attacked_posterior: Torch distribution representing the posterior under attacked data.
        target_posterior: Torch distribution representing the target posterior.
        clean_data: Tensor of clean data points.
        attacked_data: Tensor of attacked data points.
    """
    # Helper function to compute density
    def compute_density(distribution, grid_x, grid_y):
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        density = torch.exp(distribution.log_prob(grid))
        density[density < 1e-10] = np.nan
        return density.view(grid_x.shape)

    # Determine grid bounds based on data ranges
    all_data = torch.cat([clean_data, attacked_data], dim=0)
    x_min, y_min = all_data.min(dim=0).values - 1
    x_max, y_max = all_data.max(dim=0).values + 1

    # Create grid for density visualization
    x = torch.linspace(x_min.item(), x_max.item(), 100)
    y = torch.linspace(y_min.item(), y_max.item(), 100)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

    # Compute densities
    clean_density = compute_density(clean_posterior, grid_x, grid_y).detach().numpy()
    attacked_density = compute_density(attacked_posterior, grid_x, grid_y).detach().numpy()
    target_density = compute_density(target_posterior, grid_x, grid_y).detach().numpy()

    # Convert tensors to numpy
    clean_data_np = clean_data.detach().numpy()
    attacked_data_np = attacked_data.detach().numpy()

    # Plot
    plt.figure(figsize=(14, 10))
    sns.set(style="whitegrid")

    # Contour plots with seaborn
    sns.kdeplot(x=grid_x.numpy().flatten(), y=grid_y.numpy().flatten(),
                weights=clean_density.flatten(), levels=10, cmap='Blues', alpha=0.8, label='Clean Posterior')
    sns.kdeplot(x=grid_x.numpy().flatten(), y=grid_y.numpy().flatten(),
                weights=attacked_density.flatten(), levels=10, cmap='Oranges', alpha=0.8, linestyle='--', label='Attacked Posterior')
    sns.kdeplot(x=grid_x.numpy().flatten(), y=grid_y.numpy().flatten(),
                weights=target_density.flatten(), levels=10, cmap='Greens', alpha=0.8, linestyle='-.', label='Target Posterior')

    # Plot clean and attacked data points
    plt.scatter(clean_data_np[:, 0], clean_data_np[:, 1],
                c='dodgerblue', s=70, edgecolors='black', linewidth=0.8, alpha=0.8, label='Clean Data')
    plt.scatter(attacked_data_np[:, 0], attacked_data_np[:, 1],
                c='crimson', s=70, edgecolors='black', linewidth=0.8, alpha=0.8, label='Attacked Data')

    # Highlight points that changed significantly
    changed_mask = np.linalg.norm(attacked_data_np - clean_data_np, axis=1) > 1e-3
    plt.scatter(attacked_data_np[changed_mask, 0], attacked_data_np[changed_mask, 1],
                edgecolor='gold', linewidth=2, facecolor='crimson', s=90, zorder=4, label="Significantly Changed Points")

    # Draw arrows for data evolution
    for i in range(clean_data.size(0)):
        delta = attacked_data_np[i] - clean_data_np[i]
        if np.linalg.norm(delta) > 1e-3:  # Check if there is a significant change
            plt.quiver(clean_data_np[i, 0], clean_data_np[i, 1], delta[0], delta[1],
                       angles='xy', scale_units='xy', scale=1, color='gray',
                       alpha=0.7, width=0.002, headwidth=6, headlength=8, zorder=3)

    # Legends and labels
    plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True, fancybox=True)
    plt.title("Posteriors and Data Evolution", fontsize=18, weight='bold')
    plt.xlabel(r"$x_1$", fontsize=14)  # LaTeX style label
    plt.ylabel(r"$x_2$", fontsize=14)  # LaTeX style label
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_covariate_changes_vs_response(
    X_orig: torch.Tensor,
    X_attack: torch.Tensor,
    y: torch.Tensor,
    covariate_index: int,
    covariate_name: str = None,
    change_threshold: float = 1e-7,
    figsize: tuple = (8, 6),
    save_path: str = None
):
    """
    Plot changes in a single covariate vs. the response y.

    Args:
        X_orig (Tensor): Original design matrix, shape (n, p).
        X_attack (Tensor): Modified design matrix, shape (n, p).
        y (Tensor): Response vector, shape (n,).
        covariate_index (int): Which covariate column to visualize (0-based).
        covariate_name (str): Optional name to label the x-axis.
        change_threshold (float): Threshold for deciding if a covariate entry changed.
        figsize (tuple): Figure size.
        save_path (str): Optional path to save the figure in high resolution.
    """

    sns.set(style="whitegrid")

    # Ensure on CPU
    x0 = X_orig[:, covariate_index].detach().cpu()
    x1 = X_attack[:, covariate_index].detach().cpu()
    y = y.detach().cpu()

    # Detect changes
    changed = (x0 - x1).abs() > change_threshold
    unchanged = ~changed

    # Initialize figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot unchanged
    sns.scatterplot(
        x=x0[unchanged].numpy(),
        y=y[unchanged].numpy(),
        color="#9e9e9e",
        alpha=0.5,
        s=80,
        label="Unchanged",
        ax=ax
    )

    # Plot changed (original)
    sns.scatterplot(
        x=x0[changed].numpy(),
        y=y[changed].numpy(),
        color="#0072B2",
        alpha=0.6,
        s=100,
        label="Changed (original)",
        ax=ax
    )

    # Plot changed (new)
    sns.scatterplot(
        x=x1[changed].numpy(),
        y=y[changed].numpy(),
        color="#D55E00",
        alpha=0.9,
        marker="X",
        s=100,
        label="Changed (new)",
        ax=ax
    )

    # Draw horizontal lines to show change
    for i in torch.where(changed)[0]:
        ax.plot(
            [x0[i].item(), x1[i].item()],
            [y[i].item(), y[i].item()],
            color="#E69F00",
            linestyle="--",
            linewidth=1.2,
            alpha=0.6
        )

    # Labels and aesthetics
    covariate_label = covariate_name or f"Covariate {covariate_index}"
    ax.set_xlabel(covariate_label, fontsize=16)
    ax.set_ylabel("Response", fontsize=16)
    ax.set_title(f"Changes to '{covariate_label}' vs. Response", fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=12, frameon=True)
    sns.despine(trim=True)
    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig, ax



def plot_beta_posteriors(
    posterior_beta_given_sigma2,
    posterior_sigma2,
    num_samples=1000,
    bins=30,
    figsize=(14, 5),
    palette="crest",
    save_path=None
):
    """
    Plot histograms of beta parameters sampled from the Normal-Inverse-Gamma posterior,
    highlighting 95% credible intervals and the posterior means.

    Args:
        posterior_beta_given_sigma2: Function mapping sigma2 to a MultivariateNormal(mean, cov)
        posterior_sigma2: An InverseGamma distribution for sigma^2
        num_samples: Number of posterior samples to draw.
        bins: Number of histogram bins.
        figsize: Figure size.
        palette: Color palette for plots.
    """

    # Sample from the posterior
    sigma2_samples = posterior_sigma2.sample((num_samples,))
    beta_samples = torch.stack([
        posterior_beta_given_sigma2(sigma2).sample() 
        for sigma2 in sigma2_samples
    ])  # shape: (num_samples, dim_beta)

    dim_beta = beta_samples.shape[1]

    # Shorter, clearer names for beta coefficients
    beta_names = [
        "$\\beta_{0}$",
        "$\\beta_{SQ\,DIST}$",
        "$\\beta_{ELEV}$",
        "$\\beta_{OM}$",
        "$\\beta_{FLOOD}$"
    ]

    # Create figure and axes
    fig, axs = plt.subplots(1, dim_beta, figsize=figsize, constrained_layout=True)

    if dim_beta == 1:
        axs = [axs]

    # Use seaborn color palette
    colors = sns.color_palette(palette, dim_beta)

    for i, ax in enumerate(axs):
        data = beta_samples[:, i].cpu().numpy()
        mean = np.mean(data)
        ci_lower, ci_upper = np.percentile(data, [2.5, 97.5])

        sns.histplot(
            data, bins=bins, kde=True, color=colors[i], alpha=0.7, edgecolor="black", ax=ax
        )

        # Plot mean and credible interval without legends
        ax.axvline(mean, color="black", linestyle="--", linewidth=2)
        ax.axvline(ci_lower, color="red", linestyle="--", linewidth=2)
        ax.axvline(ci_upper, color="red", linestyle="--", linewidth=2)

        ax.set_title(beta_names[i], fontsize=16, fontweight='bold')
        ax.set_xlabel("Value", fontsize=14)
        if i == 0:
            ax.set_ylabel("Frequency", fontsize=14)
        else:
            ax.set_ylabel("")

        ax.tick_params(axis='both', which='major', labelsize=12)

    sns.despine(trim=True)
    plt.show()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axs




def plot_location_changes_response(
    original_locations: torch.Tensor,
    new_locations: torch.Tensor,
    ffreq,                       # 0/1 indicator – array-like length n
    response,                    # values mapped to marker size
    *,
    xlim=None, ylim=None,        # global axes limits (tuple of two floats)
    change_threshold: float = 1e-7,
    draw_lines: bool = True,
    figsize: tuple = (6, 3.6),   # identical canvas for every call
    palette = {0: "#0072B2", 1: "#D55E00"},
    size_min = 40, size_max = 200,
    save_path: str = None,
    text: bool = False           # optional index labels
):
    """Scatter map of original vs. altered sampling sites.

    • colour  = flood class (ffreq)            • size = response magnitude
    • shape   = unchanged (grey) / old (○) / new (✕)"""
    sns.set(style="whitegrid")

    # tensors → CPU numpy
    orig = original_locations.detach().cpu()
    new  = new_locations.detach().cpu()
    ffreq = torch.as_tensor(ffreq, dtype=torch.int).cpu()
    response = torch.as_tensor(response, dtype=torch.float32).cpu()

    moved  = (orig - new).abs().sum(dim=1) > change_threshold
    stayed = ~moved
    n = orig.shape[0]

    # response → marker sizes
    r_min, r_max = response.min().item(), response.max().item()
    scale = (response - r_min) / (r_max - r_min + 1e-9)
    sizes = size_min + scale * (size_max - size_min)

    fig, ax = plt.subplots(figsize=figsize)

    # unchanged points
    if stayed.any():
        sns.scatterplot(
            x=orig[stayed, 0], y=orig[stayed, 1],
            s=sizes[stayed],
            color="grey", alpha=0.4, linewidth=0,
            label="Unchanged", ax=ax
        )

    # changed points
    if moved.any():
        for cls in (0, 1):
            sel = moved & (ffreq == cls)
            if sel.any():
                # old positions
                sns.scatterplot(
                    x=orig[sel, 0], y=orig[sel, 1],
                    s=sizes[sel],
                    color=palette[cls], edgecolor="black",
                    linewidth=0.6, alpha=0.7, marker="o",
                    label=f"Changed (orig, ffreq={cls})", ax=ax
                )
                # new positions
                sns.scatterplot(
                    x=new[sel, 0], y=new[sel, 1],
                    s=sizes[sel],
                    color=palette[cls], edgecolor="black",
                    linewidth=0.6, alpha=1.0, marker="X",
                    label=f"Changed (new, ffreq={cls})", ax=ax
                )

        # dashed arrows
        if draw_lines:
            for i in torch.where(moved)[0]:
                cls = ffreq[i].item()
                ax.plot(
                    [orig[i, 0], new[i, 0]],
                    [orig[i, 1], new[i, 1]],
                    color=palette[cls],
                    linestyle="--", linewidth=1.0, alpha=0.5, zorder=0
                )

    # optional numeric labels
    if text:
        for i in range(n):
            ax.text(orig[i, 0].item(), orig[i, 1].item(),
                    str(i), fontsize=8, ha="center", va="bottom")

    # identical window for every plot
    if xlim and ylim:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # cosmetics
    ax.set_title("Original vs. Altered Locations",
                 fontsize=18, fontweight='bold')
    ax.set_xlabel("X (km)", fontsize=16)
    ax.set_ylabel("Y (km)", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_aspect("equal")
    sns.despine(trim=True)

    # proxy-artist legend (always the same)
    legend_elements = [
        Line2D([0], [0], marker='o', color='grey',
               markersize=8, linestyle='', label='Unchanged'),
        Line2D([0], [0], marker='o', color=palette[0], markeredgecolor='black',
               markersize=8, linestyle='', label='Changed (orig, ffreq=0)'),
        Line2D([0], [0], marker='X', color=palette[0], markeredgecolor='black',
               markersize=8, linestyle='', label='Changed (new, ffreq=0)'),
        Line2D([0], [0], marker='o', color=palette[1], markeredgecolor='black',
               markersize=8, linestyle='', label='Changed (orig, ffreq=1)'),
        Line2D([0], [0], marker='X', color=palette[1], markeredgecolor='black',
               markersize=8, linestyle='', label='Changed (new, ffreq=1)')
    ]
    ax.legend(handles=legend_elements, fontsize=12, frameon=True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.show()
    return fig, ax

