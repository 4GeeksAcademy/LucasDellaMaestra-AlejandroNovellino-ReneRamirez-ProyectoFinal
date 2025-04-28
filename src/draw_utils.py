"""
Utility functions to draw some common graphs.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

def draw_corr_matrix(corr: pd.DataFrame, ) -> None:
    """
    Draw a correlation matrix using seaborn.

    Args:
        corr (DataFrame): The correlation matrix to draw.

    Returns:
        None

    """

    # generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # set up the matplotlib figure
    plt.subplots(figsize=(11, 9))

    # generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        annot=True,
        mask=mask,
        cmap=cmap,
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
