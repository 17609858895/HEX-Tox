"""Figure S6. Paired measured and projected long endpoints and sensitivity checks."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
stem = "FigS6_independent_validation"
dest = P / "figures" / stem
(dest / "subfigures").mkdir(parents=True, exist_ok=True)
cases = pd.read_csv(P / "results/E16_matched_28d.csv")
cases = cases[cases.in_projected_658 & ~cases.possible_reprint]
metrics = pd.read_csv(P / "results/E16_validation_metrics.csv")
plt.rcParams.update({"font.family":"Arial","font.size":9,"axes.spines.top":False,
                     "axes.spines.right":False,"pdf.fonttype":42,"axes.linewidth":.7})


def endpoints(ax):
    trout = cases.latin_name.eq("Oncorhynchus mykiss")
    limits = [-4.5,3.5]
    ax.plot(limits,limits,color="#969696",ls="--",lw=.9,zorder=0)
    for keep,name,colour,marker in [(trout,"Rainbow trout","#93b2c5","o"),
                                   (~trout,"Other taxa","#d2b393","s")]:
        ax.scatter(cases.loc[keep,"observed_log"],cases.loc[keep,"projected_log"],
                   color=colour,edgecolor="#4b5256",lw=.55,s=34,marker=marker,label=name)
    assert cases[["observed_log","projected_log"]].min().min() >= limits[0]
    assert cases[["observed_log","projected_log"]].max().max() <= limits[1]
    ax.set(xlim=limits,ylim=limits,xticks=[-4,-2,0,2],yticks=[-4,-2,0,2],
           xlabel="Measured 28-day LC$_{50}$ (log$_{10}$ mg/L)",
           ylabel="Projected 28-day LC$_{50}$ (log$_{10}$ mg/L)")
    ax.set_aspect("equal",adjustable="box")
    ax.legend(loc="upper left",frameon=False,fontsize=8,handlelength=1.2)


def errors(ax):
    sets=["target_pairs","without_rainbow_trout","different_lead","matched_basis_medium"]
    labels=["All pairs","Without rainbow trout","Different lead author","Matched medium\nand concentration basis"]
    marks=[("persistence","96 h unchanged","o","#a9b1ba"),("common_2_52","96 h / 2.52","^","#cead91"),
           ("projection","Model projection","s","#86acaa")]
    for i,key in enumerate(sets):
        r=metrics[metrics.subset.eq(key)].set_index("method").rmse
        ax.plot([r.min(),r.max()],[i,i],color="#aeb4b6",lw=1.4,zorder=1)
        for method,label,marker,color in marks:
            ax.scatter(r[method],i,color=color,edgecolor="#494e52",lw=.6,s=46,marker=marker,
                       label=label if i==0 else None,zorder=2)
    ax.set(yticks=range(4),yticklabels=labels,ylim=(3.55,-.55),xlim=(.5,1.43),
           xticks=[.6,.8,1,1.2,1.4],xlabel="28-day RMSE (log$_{10}$ mg/L)")
    ax.tick_params(axis="y",length=0,pad=4)
    ax.legend(loc="lower center",bbox_to_anchor=(.49,1.01),frameon=False,ncol=2,
              handletextpad=.15,columnspacing=.5,fontsize=7.7)


def save(fig, basename):
    for ext in ("png","pdf"):
        fig.savefig(str(basename)+"."+ext,dpi=600,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    fig,axes=plt.subplots(1,2,figsize=(7.7,3.20),gridspec_kw={"width_ratios":[1,1.07]})
    fig.subplots_adjust(left=.095,right=.985,bottom=.17,top=.84,wspace=.92)
    for ax,letter,draw in zip(axes,"ab",[endpoints,errors]):
        draw(ax)
    fig.text(.018,.946,"a",fontsize=13);fig.text(.489,.946,"b",fontsize=13)
    save(fig,dest/stem)
    for letter,draw in zip("ab",[endpoints,errors]):
        fig,ax=plt.subplots(figsize=(3.9,3.2),layout="constrained")
        draw(ax); save(fig,dest/"subfigures"/(stem+"_"+letter))
