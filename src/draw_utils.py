"""
Utility functions to draw some common graphs.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


# helper function for the drawing functions ############################################################################

def get_groups_for_confusion_matrix(confusion: pd.DataFrame):
    """
    Get the group counts and percentages of a confusion matrix for a classification model.

    Args:
        confusion: (DataFrame): Confusion matrix from the metric of a classification model.

    Returns:

    """

    group_counts = ['{0:0.0f}'.format(value) for value in confusion.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in confusion.flatten() / np.sum(confusion)]

    return group_counts, group_percentages


# drawing functions ####################################################################################################

def draw_corr_matrix(corr: pd.DataFrame, fig_size: tuple[int, int], output_format: str=".2%") -> None:
    """
    Draw a correlation matrix using seaborn.

    Args:
        corr (DataFrame): The correlation matrix to draw.
        fig_size (tuple[int, int]): Size of the image to draw.
        output_format (): Output format of the correlation values.

    Returns:
        None
    """

    # generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # set up the matplotlib figure
    plt.subplots(figsize=fig_size)

    # generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        annot=True,
        mask=mask,
        cmap=cmap,
        fmt=output_format,
        vmax=.3,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )

    return None


def draw_confusion_matrix(confusion: pd.DataFrame) -> None:
    """
    Draw a confusion matrix using seaborn.

    Args:
        confusion (DataFrame): The confusion matrix to draw.

    Returns:
        None
    """

    # create the groups to display
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts, group_percentages = get_groups_for_confusion_matrix(confusion=confusion)

    # labels to display
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure(figsize=(5, 5))

    sns.heatmap(confusion, annot=labels, fmt='')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()
    plt.show()

    return None


def draw_comparison_confusion_matrices(
        confusion_1: pd.DataFrame,
        confusion_2: pd.DataFrame,
        confusion_matrix_1_name: str,
        confusion_matrix_2_name: str,
) -> None:
    """
    Draw a confusion matrix using seaborn.

    Args:
        confusion_1 (DataFrame): The confusion matrix of the first model.
        confusion_2 (DataFrame): The confusion matrix of the second model.
        confusion_matrix_1_name (str): Label to put on the heatmap of the first confusion matrix.
        confusion_matrix_2_name (str): Label to put on the heatmap of the second confusion matrix.

    Returns:
        None
    """

    _, axis = plt.subplots(1, 2, figsize=(20, 7))

    # create the groups to display
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    # first confusion values
    conf_1_group_counts, conf_1_group_percentages = get_groups_for_confusion_matrix(confusion=confusion_1)

    # second confusion values
    conf_2_group_counts, conf_2_group_percentages = get_groups_for_confusion_matrix(confusion=confusion_2)

    # labels to display of the first confusion matrix
    conf_1_labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
                     zip(group_names, conf_1_group_counts, conf_1_group_percentages)]
    conf_1_labels = np.asarray(conf_1_labels).reshape(2, 2)

    # labels to display of the second confusion matrix
    conf_2_labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
                     zip(group_names, conf_2_group_counts, conf_2_group_percentages)]
    conf_2_labels = np.asarray(conf_2_labels).reshape(2, 2)

    plt.figure(figsize=(10, 5))

    # first heatmap
    sns.heatmap(ax=axis[0], data=confusion_1, annot=conf_1_labels, fmt='').set(
        xlabel=f'{confusion_matrix_1_name} - True label', ylabel='Predicted label'
        )
    # second heatmap
    sns.heatmap(ax=axis[1], data=confusion_2, annot=conf_2_labels, fmt='').set(
        xlabel=f'{confusion_matrix_2_name} - True label', ylabel='Predicted label'
        )

    plt.tight_layout()
    plt.show()

    return None


def draw_roc_auc(
        y_test: pd.Series,
        y_prob: pd.Series,
        g_title: str
) -> None:
    """
    Draw the ROC AUC of a classification model.

    Args:
        y_test (DataFrame): Y test values.
        y_prob (DataFrame): Y predicted probabilities.
        g_title (str): Title of the graph.

    Returns:
        None
    """

    # calculate the metrics
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # graph the curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label=f"Random guess") # random plot
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(g_title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def draw_pr_auc(
        y_test: pd.Series,
        y_prob: pd.Series,
        g_title: str
) -> None:
    """
    Draw the PR AUC of a classification model.

    Args:
        y_test (DataFrame): Y test values.
        y_prob (DataFrame): Y predicted probabilities.
        g_title (str): Title of the graph.

    Returns:
        None
    """

    # calculate the metrics
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    # Graficar la curva
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f"PR AUC = {pr_auc:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc='lower left')
    plt.show()