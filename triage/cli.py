import argparse
from triage.data import download
from triage.model import train, evaluate
from triage.briefs import build_demo, evaluate_summaries
from triage.experiments import improve, nonlinear_study
from triage.study import improve_budget


def main():
    parser = argparse.ArgumentParser(description="Reproducible SECOM retrospective triage")
    parser.add_argument("command", choices=["download", "train", "evaluate", "build-demo", "evaluate-summaries", "improve", "compare-nonlinear", "improve-budget"])
    parser.add_argument("--live", action="store_true", help="Generate locally with Ollama; default is labeled templates")
    args = parser.parse_args()
    if args.command == "download":
        download()
    elif args.command == "train":
        train()
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "build-demo":
        build_demo(live=args.live)
    elif args.command == "improve":
        improve()
    elif args.command == "compare-nonlinear":
        nonlinear_study()
    elif args.command == "improve-budget":
        improve_budget()
    else:
        print(evaluate_summaries(live=args.live))


if __name__ == "__main__":
    main()
