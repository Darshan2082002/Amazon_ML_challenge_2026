import os

from src.evaluation.evaluate import (
    evaluate_predictions
)

from src.evaluation.error_analysis import (
    perform_error_analysis
)

from src.evaluation.final_mapping import (
    create_final_mapping
)

from src.evaluation.visualization import (
    create_visualizations
)


# ==========================================================
# PATHS
# ==========================================================

PREDICTION_FILE = (
    "output/prediction_details.tsv"
)

MATCHING_RESULTS_FILE = (
    "output/matching_results.tsv"
)

# IMPORTANT:
# Change this after finding the actual ground-truth file.
GROUND_TRUTH_FILE = None

RESULTS_DIR = (
    "results/member3"
)


def main():

    print("=" * 60)
    print("MEMBER 3 - ENTITY RESOLUTION EVALUATION")
    print("=" * 60)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Check Member 2 outputs
    # ------------------------------------------------------

    if not os.path.exists(
        PREDICTION_FILE
    ):

        raise FileNotFoundError(
            "\nMember 2 prediction file not found:\n"
            f"{PREDICTION_FILE}\n\n"
            "Run Member 2's pipeline first."
        )

    if not os.path.exists(
        MATCHING_RESULTS_FILE
    ):

        raise FileNotFoundError(
            "\nMember 2 matching result not found:\n"
            f"{MATCHING_RESULTS_FILE}\n\n"
            "Run Member 2's pipeline first."
        )

    # ------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------

    if GROUND_TRUTH_FILE is None:

        print("\n" + "=" * 60)
        print("GROUND TRUTH REQUIRED")
        print("=" * 60)

        print(
            "\nSet GROUND_TRUTH_FILE in run_member3.py "
            "to the actual challenge ground-truth file."
        )

        print(
            "\nFor example:"
        )

        print(
            'GROUND_TRUTH_FILE = '
            '"data/student_resource/ground_truth.tsv"'
        )

        print(
            "\nThe final mapping can still be generated "
            "without ground truth."
        )

        create_final_mapping(
            MATCHING_RESULTS_FILE,
            RESULTS_DIR
        )

        return

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------

    evaluation_df, metrics = (
        evaluate_predictions(
            PREDICTION_FILE,
            GROUND_TRUTH_FILE,
            RESULTS_DIR
        )
    )

    # ------------------------------------------------------
    # Error analysis
    # ------------------------------------------------------

    perform_error_analysis(
        evaluation_df,
        RESULTS_DIR
    )

    # ------------------------------------------------------
    # Final entity mapping
    # ------------------------------------------------------

    create_final_mapping(
        MATCHING_RESULTS_FILE,
        RESULTS_DIR
    )

    # ------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------

    create_visualizations(
        evaluation_df,
        metrics,
        RESULTS_DIR
    )

    # ------------------------------------------------------
    # Final summary
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("MEMBER 3 PROCESS COMPLETED")
    print("=" * 60)

    print("\nGenerated files:")

    for filename in sorted(
        os.listdir(RESULTS_DIR)
    ):
        print(
            f"  ✓ {filename}"
        )

    print(
        "\nResults directory:",
        RESULTS_DIR
    )


if __name__ == "__main__":
    main()