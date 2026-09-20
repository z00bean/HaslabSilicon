"""Reference-model exceptions."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan


class ArithmeticOverflowError(ArithmeticError):
    """An exact v0 operation exceeded its required accumulator range."""


class UnsupportedOperationError(NotImplementedError):
    """The requested behavior is intentionally absent from the target profile."""
