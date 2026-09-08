#!/usr/bin/env python3
"""Generate the fixed-random-feature control from the adjacent PC cell library.
MIT License; Copyright (c) 2026 Thomas Dybdahl Ahle and contributors.
"""
import importlib.util
from pathlib import Path

library=Path(__file__).resolve().parent.parent / "predictive-coding" / "generate.py"
spec=importlib.util.spec_from_file_location("pc_submission",library)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if __name__ == "__main__":
    module.main(default_frozen=True)
