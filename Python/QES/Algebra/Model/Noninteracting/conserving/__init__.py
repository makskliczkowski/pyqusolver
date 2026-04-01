"""
Conserving Non-Interacting Models Module
========================================

This module contains non-interacting quantum models with particle number conservation.

Modules:
--------
- free_fermions: Free fermionic models
- Majorana: Majorana fermion models (Kitaev, Kitaev-Gamma, etc.)

Author: Maksymilian Kliczkowski
Email: maksymilian.kliczkowski@pwr.edu.pl
"""

from . import free_fermions
from . import Majorana

__all__ = ["free_fermions", "Majorana"]
