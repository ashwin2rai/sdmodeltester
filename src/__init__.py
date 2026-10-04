"""Stable Diffusion image test UI (SD 1.5 / SDXL single-file checkpoints).

Importing this package must never import torch or diffusers; the real
backend imports them lazily so the mock path runs on any machine.
"""

__version__ = "0.1.0"
