import os
import pandas as pd


def perform_error_analysis(
    evaluation_df,
    output_directory
):

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # False positives
    false_positives = evaluation_df[
        (evaluation_df["true_label"] == 0)
        &
        (evaluation_df["prediction"] == 1)
    ].copy()

    # False negatives
    false_negatives = evaluation_df[
        (evaluation_df["true_label"] == 1)
        &
        (evaluation_df["prediction"] == 0)
    ].copy()

    # High-confidence false positives
    if "match_probability" in false_positives.columns:

        high_conf_fp = false_positives[
            false_positives["match_probability"] >= 0.90
        ]

        high_conf_fp.to_csv(
            os.path.join(
                output_directory,
                "high_confidence_false_positives.csv"
            ),
            index=False
        )

    # Low-confidence true matches
    if "match_probability" in false_negatives.columns:

        low_conf_fn = false_negatives[
            false_negatives["match_probability"] < 0.50
        ]

        low_conf_fn.to_csv(
            os.path.join(
                output_directory,
                "low_confidence_false_negatives.csv"
            ),
            index=False
        )

    print("\n========== ERROR ANALYSIS ==========")

    print(
        f"False Positives : {len(false_positives)}"
    )

    print(
        f"False Negatives : {len(false_negatives)}"
    )

    if len(false_positives) > 0:
        print("\nSample False Positives:")
        print(
            false_positives.head(10).to_string(
                index=False
            )
        )

    if len(false_negatives) > 0:
        print("\nSample False Negatives:")
        print(
            false_negatives.head(10).to_string(
                index=False
            )
        )

    return false_positives, false_negatives