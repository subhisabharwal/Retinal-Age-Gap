import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def generate_attractive_distributions():
    # 1. Load Data
    print("Loading BRSET dataset...")
    brset_df = pd.read_csv("data/BRSET/labels_brset.csv")
    
    print("Loading ODIR-5K dataset...")
    odir_df = pd.read_excel("data/ODIR-5K/data.xlsx")
    
    # Extract and clean ages
    odir_ages = odir_df[['Patient Age']].dropna().copy()
    odir_ages.columns = ['Age']
    
    brset_ages = brset_df[['patient_age']].dropna().copy()
    brset_ages.columns = ['Age']

    os.makedirs("outputs/presentation", exist_ok=True)
    sns.set_theme(style="whitegrid", rc={"axes.spines.top": False, "axes.spines.right": False})

    # =========================================================================
    # PLOT 1 & 2: Separate KDE Plots
    # =========================================================================
    
    # ODIR-5K KDE
    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=odir_ages, x="Age", fill=True, color="#3498db", alpha=0.6, linewidth=2.5)
    plt.title("ODIR-5K: Patient Age Density", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Age (Years)", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    plt.tight_layout()
    odir_kde_path = "outputs/presentation/odir_kde.png"
    plt.savefig(odir_kde_path, dpi=300, bbox_inches='tight')
    plt.close()

    # BRSET KDE
    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=brset_ages, x="Age", fill=True, color="#e74c3c", alpha=0.6, linewidth=2.5)
    plt.title("BRSET: Patient Age Density", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Age (Years)", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    plt.tight_layout()
    brset_kde_path = "outputs/presentation/brset_kde.png"
    plt.savefig(brset_kde_path, dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # PLOT 3 & 4: Stylish Histogram + KDE (Replaces the Violin Plot)
    # =========================================================================
    
    # ODIR-5K Histogram
    plt.figure(figsize=(9, 5))
    sns.histplot(data=odir_ages, x="Age", bins=30, kde=True, color="#2ecc71", alpha=0.5, edgecolor="white", linewidth=1.5)
    plt.title("ODIR-5K: Detailed Age Distribution", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Age (Years)", fontsize=13)
    plt.ylabel("Number of Patients", fontsize=13)
    plt.tight_layout()
    odir_hist_path = "outputs/presentation/odir_hist.png"
    plt.savefig(odir_hist_path, dpi=300, bbox_inches='tight')
    plt.close()

    # BRSET Histogram
    plt.figure(figsize=(9, 5))
    sns.histplot(data=brset_ages, x="Age", bins=40, kde=True, color="#9b59b6", alpha=0.5, edgecolor="white", linewidth=1.5)
    plt.title("BRSET: Detailed Age Distribution", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Age (Years)", fontsize=13)
    plt.ylabel("Number of Patients", fontsize=13)
    plt.tight_layout()
    brset_hist_path = "outputs/presentation/brset_hist.png"
    plt.savefig(brset_hist_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("\nGeneration Complete!")
    print(f"Saved: {odir_kde_path}")
    print(f"Saved: {brset_kde_path}")
    print(f"Saved: {odir_hist_path}")
    print(f"Saved: {brset_hist_path}")

if __name__ == "__main__":
    generate_attractive_distributions()
