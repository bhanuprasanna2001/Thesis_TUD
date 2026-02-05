"""Train Diffusion on MNIST Dataset for Generation."""

import sys
import yaml
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import get_scheduler


def main():
    config = {
        # Scheduler
        "start": 0.0001,
        "end": 0.02,
        "timesteps": 1000,
        "type": "linear",
    }

    betas = get_scheduler()
    print(betas.size())
    print(betas)


if __name__ == "__main__":
    main()