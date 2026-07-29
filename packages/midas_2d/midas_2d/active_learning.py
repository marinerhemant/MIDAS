"""Experiment design -- now provided by the shared ``midas_invert`` leaf.

Thin re-export so existing ``midas_2d`` imports are unchanged.
"""
from midas_invert.design import (
    fisher_information,
    next_best_measurement,
    rank_measurements,
    sensitivity,
)

__all__ = ["sensitivity", "fisher_information", "rank_measurements",
           "next_best_measurement"]
