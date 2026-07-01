"""Shared pytest fixtures/config. Forces a non-interactive matplotlib
backend so the test suite runs headless (CI, no display)."""
import matplotlib

matplotlib.use("Agg")
