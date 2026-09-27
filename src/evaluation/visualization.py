import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)


def create_visualizations(
    evaluation_df,
    metrics,
    output_directory
):

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # ---------------------------------
    # 1. Confusion Matrix
    # ---------------------------------

    cm = confusion_matrix(
        evaluation_df["true_label"],
        evaluation_df["prediction"]
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Non-Match",
            "Match"
        ]
    )

    disp.plot()

    plt.title(
        "Entity Resolution Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_directory,
            "confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()

    # ---------------------------------
    # 2. Model Metrics
    # ---------------------------------

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f0.5"
    ]

    values = [
        metrics.get(
            name,
            0
        )
        for name in metric_names
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        metric_names,
        values
    )

    plt.ylim(
        0,
        1
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Entity Resolution Model Performance"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_directory,
            "model_metrics.png"
        ),
        dpi=300
    )

    plt.close()

    # ---------------------------------
    # 3. Probability Distribution
    # ---------------------------------

    if "match_probability" in evaluation_df.columns:

        plt.figure(
            figsize=(8, 5)
        )

        plt.hist(
            evaluation_df[
                evaluation_df["true_label"] == 1
            ]["match_probability"].dropna(),
            bins=20,
            alpha=0.7,
            label="True Match"
        )

        plt.hist(
            evaluation_df[
                evaluation_df["true_label"] == 0
            ]["match_probability"].dropna(),
            bins=20,
            alpha=0.7,
            label="True Non-Match"
        )

        plt.xlabel(
            "Match Probability"
        )

        plt.ylabel(
            "Number of Candidate Pairs"
        )

        plt.title(
            "Match Probability Distribution"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_directory,
                "probability_distribution.png"
            ),
            dpi=300
        )

        plt.close()

    print(
        "\nVisualizations saved to:",
        output_directory
    )