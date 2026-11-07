"""
Exploratory analysis + feature-independence checks for the Adult dataset
Usage: python eda.py [adult.csv]   ->  prints a summary, saves plots to eda/
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from common import CATEGORICAL, FEATURES, NUMERIC, TARGET, clean, encode_target

OUT = Path(__file__).parent / "eda"
OUT.mkdir(exist_ok=True)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b)
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def main(path: str) -> None:
    df = clean(pd.read_csv(path))
    df[TARGET] = encode_target(df[TARGET])

    print("== Missing values (%) ==")
    print((df[FEATURES].isna().mean() * 100).round(2)[lambda s: s > 0].to_string())
    print(f"\nExact duplicate rows: {df.duplicated().sum()}")
    print(f"Target balance (share >50K): {df[TARGET].mean():.3f}")

    print("\n== Skewness of numeric features ==")
    print(df[NUMERIC].skew().round(2).to_string())
    print("Share of zeros: capital_gain "
          f"{(df.capital_gain == 0).mean():.1%}, capital_loss {(df.capital_loss == 0).mean():.1%}")

    # numeric correlation
    corr = df[NUMERIC + [TARGET]].corr()
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Pearson correlation - numeric features")
    plt.tight_layout(); plt.savefig(OUT / "numeric_correlation.png", dpi=130); plt.close()
    print("\n== Numeric correlation with target ==")
    print(corr[TARGET].drop(TARGET).round(3).to_string())

    # categorical association (Cramér's V), incl. education vs education_num
    cats = CATEGORICAL
    v = pd.DataFrame(index=cats, columns=cats, dtype=float)
    d = df[cats].fillna("Missing")
    for a in cats:
        for b in cats:
            v.loc[a, b] = 1.0 if a == b else cramers_v(d[a], d[b])
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(v, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1)
    plt.title("Cramér's V - categorical feature association")
    plt.tight_layout(); plt.savefig(OUT / "categorical_association.png", dpi=130); plt.close()
    pairs = (v.where(np.triu(np.ones(v.shape), 1).astype(bool)).stack()
             .sort_values(ascending=False).head(5))
    print("\n== Strongest categorical associations (Cramér's V) ==")
    print(pairs.round(3).to_string())
    print(f"education vs education_num (Cramér's V): "
          f"{cramers_v(df['education'], df['education_num'].astype(str)):.3f}  -> same information")

    # association of each categorical feature with target
    print("\n== Cramér's V with target ==")
    tv = {c: cramers_v(d[c], df[TARGET]) for c in cats}
    print(pd.Series(tv).sort_values(ascending=False).round(3).to_string())

    # distributions
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), NUMERIC):
        sns.histplot(df[col], bins=40, ax=ax)
        ax.set_title(f"{col} (skew {df[col].skew():.2f})")
    plt.tight_layout(); plt.savefig(OUT / "numeric_distributions.png", dpi=130); plt.close()

    # class balance by key features
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    df[TARGET].map({0: "<=50K", 1: ">50K"}).value_counts().plot.bar(ax=axes[0], rot=0)
    axes[0].set_title("Class balance")
    df.groupby("marital_status")[TARGET].mean().sort_values().plot.barh(ax=axes[1])
    axes[1].set_title("Share >50K by marital status")
    df.groupby("education_num")[TARGET].mean().plot(ax=axes[2], marker="o")
    axes[2].set_title("Share >50K by education_num")
    plt.tight_layout(); plt.savefig(OUT / "target_relationships.png", dpi=130); plt.close()
    print(f"\nPlots saved to {OUT}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "adult.csv")
