"""Execute stage train explicitly; use run_analysis.py for configured stages."""
import sys
from module.pipeline import main
if __name__ == "__main__":
    raise SystemExit(main(["--stage", "train", *sys.argv[1:]]))
