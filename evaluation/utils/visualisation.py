import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import torch
import os

COLORS = ["black", "blue", "green", "red", "purple", "orange", "yellow"]
from itertools import cycle

BASE_DIR = os.getenv("BASE_DIR")


def _visualize_log_odds(
    title,
    log_odds: np.ndarray,
    mean: np.ndarray,
    max: np.ndarray,
    min: np.ndarray,
    random_references_mean: np.ndarray,
    apply_log: bool,
) -> None:
    """
    Serves the IntegratedGradientEvaluator class for visualization.
    """
    x_dims = len(mean)
    plt.figure(figsize=(7,6))
    x = np.linspace(0, 1, x_dims)
    ax = plt.plot(x, mean, label="Mean", linestyle="-", color="black")
    plt.plot(x, max, label="Max", linestyle="--", color="gray")
    plt.plot(x, min, label="Min", linestyle=":", color="gray")
    plt.plot(x, random_references_mean, label="Mean of Random Reference")

    plt.xlabel("Fraction of masked features", fontsize=16)

    if apply_log:
        plt.ylabel("Log Odds Ratio", fontsize=16)
    else: 
        plt.ylabel("Model output f(x)", fontsize=16)

    plt.title(title, fontsize=20)

    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig("log_odds_ig.png", format="png")
    # plt.show()

def _visualize_completeness_deltas_comparison(
        completeness_deltas_list : np.ndarray,
        n_steps_arr: list[int]
    ) -> None:

    plt.plot(n_steps_arr, completeness_deltas_list)
    plt.xlabel("Anzahl der Approximationsschritte m")
    plt.ylabel("Delta")

    plt.ticklabel_format(style="sci",axis="y",scilimits=(0,0))

    plt.tight_layout()
    plt.show()


def _visualize_completeness_deltas_boxplots(
        completeness_deltas : list[np.ndarray],
        n_steps_arr: list[int]
    ) -> None:

    medians = [np.median(deltas) for deltas in completeness_deltas]
    vertical_offset = 0.001

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    sb.set_style("whitegrid")

    boxplot = sb.boxplot(data=completeness_deltas, ax=ax1, boxprops={"facecolor": "white"})
    ax1.set_title(r"|$\delta$| Boxplots je $m$", fontsize=18)
    ax1.set_xticklabels(n_steps_arr)
    ax1.set_xlabel(r"Anzahl Approximationsschritte $m$",fontsize=14)
    ax1.set_ylabel(r"|$\delta$|",fontsize=14)
    ax1.ticklabel_format(style="sci",axis="y",scilimits=(0,0))
    ax1.yaxis.get_offset_text().set_fontsize(14)

    for xtick in boxplot.get_xticks():
        boxplot.text(xtick,medians[xtick] + vertical_offset,f"{medians[xtick] : .1e}", 
                horizontalalignment='center',fontsize=12,color='black')

    ax2.set_title(r"Median von |$\delta$| je $m$",fontsize=18)
    ax2.plot(n_steps_arr, [np.median(deltas) for deltas in completeness_deltas], "--", color="black")
    ax2.plot(n_steps_arr,[np.median(deltas) for deltas in completeness_deltas], "x", color="black")
    ax2.set_xlabel(r"Anzahl Approximationsschritte $m$",fontsize=14)
    ax2.set_ylabel(r"Median von |$\delta$|",fontsize=14)
    ax2.set_xticks(n_steps_arr)
    ax2.ticklabel_format(style="sci",axis="y",scilimits=(0,0))
    ax2.yaxis.get_offset_text().set_fontsize(14)



    plt.tight_layout()
    plt.savefig("./figures/approx_ig.eps", format="eps")


def visualize_baselines_output(
        zero_baseline_output : torch.Tensor,
        uniform_output_baseline_output : torch.Tensor
    ) -> None:

    x = np.arange(len(zero_baseline_output))

    fig, ax = plt.subplots(figsize=(5,2.2))
    ax.bar(x-0.1,zero_baseline_output.detach().numpy(),width=0.2, color="black", label="Zero Baseline")
    ax.bar(x+0.1, uniform_output_baseline_output.detach().numpy(),width=0.2, color="royalblue", label="Uniform Output Baseline")

    ax.plot(x,np.ones_like(x)*1/len(x),linestyle="dashed", color="gray", label="Uniforme Verteilung")

    ax.set_title("Modellausgabe der Zero Baseline und Uniform Output Baseline", fontsize=12)
    ax.set_xlabel(r"Klassenindex $j$")
    ax.set_ylabel(r"Modellausgabe $f(x')_j$")
    ax.set_ylim(0.0,0.4)

    #plt.tight_layout()
    lgd = plt.legend(bbox_to_anchor=(0.5, -0.28), loc='upper center', ncols=3)
    plt.savefig("./figures/uniform_zero_baseline_output.eps", format="eps",bbox_extra_artists=(lgd,),bbox_inches='tight')


def _visualize_log_odds_comparison(
    log_odds_mean_uniform_output_baseline : np.ndarray,
    certainty_mean_uniform_output_baseline : np.ndarray,
    log_odds_mean_zero_baseline : np.ndarray,
    certainty_mean_zero_baseline : np.ndarray,
    log_odds_mean_random_masking : np.ndarray,
    certainty_mean_random_masking : np.ndarray
    ) -> None:

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,2.7))

    x_dims = len(mean)
    x = np.linspace(0, 1, x_dims)


    ax1.plot(x, certainty_mean_zero_baseline, color="black", label="Zero Baseline")
    ax1.plot(x, certainty_mean_uniform_output_baseline, color="royalblue", label="Uniform Output Baseline")
    ax1.plot (x, certainty_mean_random_masking, linestyle="dashed", color="gray", label="Random Reference")
    ax1.set_title("Certainty Kurven", fontsize=12)
    ax1.set_ylabel(r"Mean von $f(x)$")
    ax1.set_xlabel("Anteil der maskierten Features")
    ax1.set_xlim(0.0,0.2)
    handles, labels = ax1.get_legend_handles_labels()

    ax2.plot(x, log_odds_mean_zero_baseline, color="black")
    ax2.plot(x, log_odds_mean_uniform_output_baseline, color="royalblue")
    ax2.plot(x, log_odds_mean_random_masking, linestyle="dashed", color="gray")
    ax2.set_title("Log Odds Kurven", fontsize=12)
    ax2.set_ylabel(r"Mean von $log(f(x)/(1-f(x)))$")
    ax2.set_xlabel("Anteil der maskierten Features")
    ax2.set_xlim(0.0,0.2)

    lgd = plt.legend(handles, labels=labels,bbox_to_anchor =(0,-0.28), loc='upper center',ncols=3)
    plt.subplots_adjust(wspace=0.3)
    plt.savefig("./figures/ig_log_odds.eps", format="eps",bbox_extra_artists=(lgd,),bbox_inches='tight')


def visualize_log_odds_of_attribution_methods(
        zero_baseline_log_odds_mean : np.ndarray,
        zero_uniform_output_baseline_log_odds_mean : np.ndarray,
        mean_baseline_log_odds_mean : np.ndarray,
        furthest_baseline_log_odds_mean : np.ndarray,
        nearest_baseline_log_odds_mean : np.ndarray,
        nearest_uniform_output_baseline_log_odds_mean : np.ndarray,
        furthest_uniform_output_baseline_log_odds_mean : np.ndarray,
        apply_log: bool = False,
        title: str = "",
        save_fig: bool = False
    ):
    fig, axs =  plt.subplots(1,2, figsize=(13,6.5))

    if title != "":
        fig.suptitle(title, fontsize=20)

    x_dims = len(zero_baseline_log_odds_mean)
    x = np.linspace(0,1, x_dims)

    axs[0].plot(x, zero_baseline_log_odds_mean, label="Zero Baseline", color='black', linestyle="--")
    axs[0].plot(x, zero_uniform_output_baseline_log_odds_mean, label="Zero Uniform Output Baseline", color='blue', linestyle="--")
    axs[0].plot(x, mean_baseline_log_odds_mean, label="Mean Baseline", color='green', linestyle="--")
    axs[0].plot(x, furthest_baseline_log_odds_mean, label="Furthest Baseline", color='red')
    axs[0].plot(x, nearest_baseline_log_odds_mean, label="Nearest Baseline", color='purple')
    axs[0].plot(x, nearest_uniform_output_baseline_log_odds_mean, label="Nearest Uniform Output Baseline", color='orange')
    axs[0].plot(x, furthest_uniform_output_baseline_log_odds_mean, label="Furthest Uniform Output Baseline", color='yellow')

    for ax in axs:
        ax.set_xlabel("Fraction of masked features", fontsize=16)
        if apply_log:
            ax.set_ylabel("Log Odds Ratio", fontsize=16)
        else: 
            ax.set_ylabel("Model output f(x)", fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)


    axs[0].set_title("local vs global", fontsize=16)
    axs[0].legend(fontsize=14, loc="upper right")

    axs[1].set_title("contrastive vs non-contrastive", fontsize=16)
    axs[1].plot(x, furthest_baseline_log_odds_mean, label="Furthest Baseline", color='red', linestyle="--")
    axs[1].plot(x, furthest_uniform_output_baseline_log_odds_mean, label="Furthest Uniform Output Baseline", color='yellow', linestyle="--")
    axs[1].plot(x, nearest_baseline_log_odds_mean, label="Nearest Baseline", color='purple')
    axs[1].plot(x, nearest_uniform_output_baseline_log_odds_mean, label="Nearest Uniform Output Baseline", color='orange')
    axs[1].legend(fontsize=14, loc="upper right")

    plt.tight_layout()


    if save_fig:
        path = os.path.join(BASE_DIR, "figures", f"log_odds_comp_attr_methods.png")
        plt.savefig(path, dpi=300)
    plt.show()

def visualize_logs_odds_with_different_masking_baselines(
    dict_of_log_odds: dict[str, np.ndarray],
    black_line: str,
    title: str="",
    apply_log: bool = False,
    save_fig: bool = False
):
    
    colors_cycle = cycle(COLORS)
    next(colors_cycle)

    plt.figure(figsize=(10,6))
    x_dims = len(dict_of_log_odds[list(dict_of_log_odds.keys())[0]])
    x = np.linspace(0,1, x_dims)
    for key, value in dict_of_log_odds.items():
        if key == black_line:
            plt.plot(x, value, label=key, color='black')
        else:
            plt.plot(x, value, label=key, color=next(colors_cycle))
    
    plt.xlabel("Fraction of masked features", fontsize=16)
    if apply_log:
        plt.ylabel("Log Odds Ratio", fontsize=16)
    else: 
        plt.ylabel("Model output f(x)", fontsize=16)

    plt.title(title, fontsize=16)
    plt.legend(fontsize=14, loc="upper right")
    plt.tight_layout()
    if save_fig:
        path = os.path.join(BASE_DIR, "figures", f"log_odds_comp_different_masking_baselines_attr_{black_line}.png")
        plt.savefig(path, dpi=300)
    plt.show()

def visualize_feature_agreement_matrices(
    k: int,
    fa_mean: np.ndarray,
    fa_variance: np.ndarray,
    labels: list[str],
    rank_agreement: bool = False,
    save_fig: bool = False 
)-> None:
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(15,15))
    ax1.set_title(f"Feature Agreement Matrix of IG (k={k})", fontsize=16)
    sb.heatmap(fa_mean[0:7,:], cmap='viridis', annot=True, ax=ax1, yticklabels=labels[0:7], xticklabels=labels, annot_kws={"fontsize":14}, vmin=0, vmax=1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=14)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=14)
    cbar = ax1.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)

    ax2.set_title(f"Feature Agreement Matrix of LIME (k={k})", fontsize=16)
    sb.heatmap(fa_mean[7:14,:], cmap='viridis', annot=True, ax=ax2, yticklabels=labels[7:14], xticklabels=labels, annot_kws={"fontsize":14}, vmin=0, vmax=1)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=14)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=14)
    cbar = ax2.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)

    ax3.set_title(f"Feature Agreement Matrix of BShap (k={k})", fontsize=16)
    sb.heatmap(fa_mean[14:,:], cmap='viridis', annot=True, ax=ax3, yticklabels=labels[14:], xticklabels=labels, annot_kws={"fontsize":14}, vmin=0, vmax=1)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=14)
    ax3.set_yticklabels(ax3.get_yticklabels(), fontsize=14)
    cbar = ax3.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)

    plt.subplots_adjust(hspace=0.9)
    plt.subplots_adjust(left=0.1)

    plt.tight_layout()

    if save_fig:
        path = os.path.join(BASE_DIR, "figures", f"feature_agreement_matrix_k_{k}_rank_agreement_{rank_agreement}.png")
        plt.savefig(path, dpi=300)
    plt.show()

