"""Run with python evaluate.py; optionally save metrics and a holdout plot."""
import argparse
import json
from pathlib import Path
from modeling import load_data, train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    _, report, actual, predicted = train_and_evaluate(load_data())
    rendered = json.dumps(report, indent=2, allow_nan=False)
    print(rendered)
    if args.output_dir:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'metrics.json').write_text(rendered + '\n')
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(actual / 100000, predicted / 100000, alpha=0.65)
        bounds = [min(actual.min(), predicted.min()) / 100000,
                  max(actual.max(), predicted.max()) / 100000]
        ax.plot(bounds, bounds, '--', color='gray', label='Perfect prediction')
        ax.set(xlabel='Actual price (INR lakh)', ylabel='Predicted price (INR lakh)',
               title='Linear regression: unseen holdout houses')
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / 'holdout_predictions.png', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    main()
