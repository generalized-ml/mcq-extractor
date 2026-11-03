"""Compatibility entry point; evaluation now lives in evaluation.evaluate."""
from evaluation.evaluate import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
