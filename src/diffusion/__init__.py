"""
diffusion/__init__.py
─────────────────────────────────────────────────────────────────────────────
Public interface of the diffusion package (Phase 3).

Exports only TabDDPM — the rest of the pipeline never needs to import
internal modules directly. Internal structure can change freely without
affecting any other part of the project.

Usage from main.py or Phase 4:
    from diffusion import TabDDPM

Part of:
    Synthetic Tabular Data Engine — Auto-Adaptive Diffusion System
"""

from .tabddpm import TabDDPM

__all__ = ["TabDDPM"]