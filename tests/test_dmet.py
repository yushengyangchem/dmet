"""
Test suite for DMET (Density Matrix Embedding Theory) implementation.

This module contains tests for the DMET package, serving both as
verification of the implementation and as usage examples.
"""

from pyscf import gto

from dmet import RHFDMET, make_atom_fragments


def test_h4_chain_runs_and_builds_fragment_embeddings() -> None:
    """Test DMET calculation on a 4-atom hydrogen chain.

    This test verifies that:
    1. The DMET calculation completes without errors
    2. Four fragments are created (one per atom)
    3. The total energy is negative and below the mean-field energy
    4. The mean-field energy is negative
    5. Each fragment has valid embedding dimensions and energies

    This test also serves as the minimal end-to-end usage example,
    demonstrating how to set up and run a DMET calculation using
    atomic fragments.
    """
    mol = gto.M(
        atom="""
        H 0.0 0.0 0.0
        H 0.0 0.0 0.8
        H 0.0 0.0 1.6
        H 0.0 0.0 2.4
        """,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )

    result = RHFDMET(mol).kernel(make_atom_fragments(mol))

    assert len(result.fragments) == 4
    assert result.total_energy < 0.0
    assert result.total_energy < result.mean_field_energy
    assert result.mean_field_energy < 0.0

    for fragment in result.fragments:
        assert fragment.embedding_dimension >= len(fragment.fragment.ao_indices)
        assert fragment.bath_dimension <= len(fragment.fragment.ao_indices)
        assert fragment.embedding_energy < 0.0
