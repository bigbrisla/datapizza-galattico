import argparse
from pathlib import Path

from .pipeline import MvpAssistantPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVP GenAI assistant for Datapizza technical test")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to test-tecnico-ai-engineer repository root",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Path to questions CSV (default: Dataset/domande.csv)",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Path to dish_mapping.json (default: Dataset/ground_truth/dish_mapping.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: submissions/mvp_submission.csv)",
    )
    parser.add_argument(
        "--mode",
        choices=["structured", "lexical"],
        default="structured",
        help="Retrieval strategy to use",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    repo_root = args.repo_root
    questions_path = args.questions or repo_root / "Dataset" / "domande.csv"
    mapping_path = args.mapping or repo_root / "Dataset" / "ground_truth" / "dish_mapping.json"
    output_path = args.output or repo_root / "submissions" / "mvp_submission.csv"

    pipeline = MvpAssistantPipeline()
    if args.mode == "lexical":
        generated = pipeline.run_lexical(
            repo_root=repo_root,
            questions_path=questions_path,
            mapping_path=mapping_path,
            output_path=output_path,
        )
    else:
        generated = pipeline.run(
            repo_root=repo_root,
            questions_path=questions_path,
            mapping_path=mapping_path,
            output_path=output_path,
        )

    print(f"Submission generated: {generated}")


if __name__ == "__main__":
    main()
