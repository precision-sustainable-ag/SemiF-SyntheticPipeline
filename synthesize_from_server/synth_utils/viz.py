from pathlib import Path
from textwrap import wrap

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec


def compare_og_w_cutout(ogimg,
                        cutout,
                        save=False,
                        show=False,
                        save_path="./compare_og_w_cutout.png",
                        facecolor="black",
                        figsize=(12, 8)):

    fig, ax = plt.subplots(1, 3, figsize=figsize, facecolor=facecolor)

    ax[0, 0].imshow(ogimg)
    ax[0, 1].imshow(cutout)
    if save == True:
        fig.savefig(save_path,
                    dpi=400,
                    transparent=False,
                    bbox_inches='tight',
                    facecolor=facecolor,
                    pad_inches=0)
        fig.savefig(save_path.replace(".png", "_transparent.png"),
                    dpi=400,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
    if not show:
        plt.close(fig)
    else:
        plt.show()


def display_images(images,
                   nrow=1,
                   ncol=1,
                   save=False,
                   show=False,
                   save_path="./display_image.png",
                   facecolor="black",
                   figsize=(12, 8)):

    fig, ax = plt.subplots(nrows=nrow,
                           ncols=ncol,
                           figsize=figsize,
                           facecolor=facecolor)

    if len(images) > 1:
        for ind, img in enumerate(images):
            ax.ravel()[ind].imshow(img)
            ax.ravel()[ind].set_axis_off()
            plt.tight_layout()
    else:
        ax.imshow(images[0])
        ax.set_axis_off()

        # plt.title(Path(path).stem)
        plt.tight_layout()

    if save == True:
        fig.savefig(save_path,
                    dpi=400,
                    transparent=False,
                    bbox_inches='tight',
                    facecolor=facecolor,
                    pad_inches=0)
        fig.savefig(save_path.replace(".png", "_transparent.png"),
                    dpi=400,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
    if not show:
        plt.close(fig)
    else:
        plt.show()


def plot_cutouts(subplot,
                 df,
                 y,
                 title,
                 xlabel,
                 ylabel,
                 color,
                 figsize,
                 text_wrapping,
                 rotation,
                 tick_bottom,
                 tick_left,
                 save_path,
                 save,
                 show=True):
    # df['common_name'] = pd.Categorical(df['common_name'],
    #    sorted(df['common_name'].unique()))
    copy_df = df.copy()
    copy_df["common_name"] = pd.Categorical(df["common_name"],
                                            sorted(df["common_name"].unique()))

    fig, ax = plt.subplots(figsize=figsize)
    fig.tight_layout(pad=5)

    ax = plt.subplot(1, 2, subplot)
    yplot = sns.histplot(y=y,
                         data=copy_df,
                         hue="is_primary",
                         multiple="dodge",
                         shrink=.85,
                         discrete=True,
                         palette=sns.diverging_palette(240, 16, s=92, n=2))
    yplot.bar_label(yplot.containers[0], padding=5)
    yplot.bar_label(yplot.containers[1], padding=5)

    plt.title(title, fontsize=22)
    plt.xlabel(xlabel, fontsize=12)
    plt.xticks(fontsize=12, rotation=rotation)
    plt.ylabel(ylabel, fontsize=12)

    labels = copy_df[y].unique()
    labels = np.sort(labels)
    labels = ['\n'.join(wrap(l, text_wrapping)) for l in labels]

    # y_pos = np.arange(len(labels))
    y_pos = np.argsort(labels)
    ax.set_yticks(y_pos, labels=labels)
    sns.despine(bottom=False, left=True)

    ax.grid(False)
    ax.tick_params(bottom=tick_bottom,
                   left=tick_left,
                   axis='both',
                   which='major',
                   labelsize=10)

    # ax.tick_params(axis='both', which='minor', labelsize=8)

    font = font_manager.FontProperties(weight='bold', style='normal', size=16)
    legend_labels = ['unique', 'duplicates']
    plt.legend(labels=legend_labels, title='', loc='lower right', prop=font)
    plt.tight_layout()
    if save == True:
        fig.savefig(save_path,
                    dpi=400,
                    transparent=False,
                    bbox_inches='tight',
                    facecolor='white',
                    pad_inches=0)
        fig.savefig(save_path.replace(".png", "_transparent.png"),
                    dpi=400,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
    if not show:
        plt.close(fig)
    else:
        plt.show()
    return None
