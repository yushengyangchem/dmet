"""
DMET - Density Matrix Embedding Theory

A minimal implementation of one-shot DMET (Density Matrix Embedding Theory)
built on top of PySCF for quantum chemistry calculations.

This package provides:
- RHF-based DMET solver
- Automatic fragment generation from molecular structure
- FCI embedding solver for accurate fragment calculations

Key Features:
- RHF reference state
- Lowdin-orthogonalized AO local basis
- Environment 1-RDM based bath/core orbital construction
- Fragment embedding Hamiltonian construction
- One-shot DMET total energy estimation

Example:
    >>> from pyscf import gto
    >>> from dmet import RHFDMET, make_atom_fragments
    >>>
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 0.8", basis="sto-3g")
    >>> dmet = RHFDMET(mol)
    >>> result = dmet.kernel(make_atom_fragments(mol))
    >>> print(f"DMET energy: {result.total_energy}")

References:
    - Knizia, G., & Chan, G. K. L. (2012). Density matrix embedding:
      A simple alternative to dynamical mean-field theory.
      Physical Review Letters, 109(26), 263001.
"""

__version__ = "0.1.0"

from dmet.core import (
    RHFDMET,
    DMETFragmentResult,
    DMETResult,
    Fragment,
    make_atom_fragments,
)

__all__ = [
    "DMETFragmentResult",
    "DMETResult",
    "Fragment",
    "RHFDMET",
    "make_atom_fragments",
]
