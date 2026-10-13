"""Standalone frozen-model worker evaluation.

Evaluates frozen language models as candidates for future worker A/B roles:
  1. evidence-based local QA with abstention (MuSiQue local task)
  2. mathematical reasoning checking (ProcessBench)

This project does NOT train, does not load LoRA, does not run a coordinator,
and does not simulate inter-agent communication.
"""

__version__ = "0.1.0"
