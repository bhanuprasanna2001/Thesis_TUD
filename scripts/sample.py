"""Generate samples from a trained checkpoint."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import torch

from src.models import DiffusionModel
from src.utils import save_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--output", type=str, default="samples.png")
    parser.add_argument("--steps", type=int, default=None, help="Sampling steps")
    args = parser.parse_args()

    model = DiffusionModel.load_from_checkpoint(args.checkpoint)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    print(f"Device: {device}")
    print(f"Generating {args.n_samples} samples...")

    with torch.no_grad():
        samples = model.sample(args.n_samples, steps=args.steps)

    save_samples(samples, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
