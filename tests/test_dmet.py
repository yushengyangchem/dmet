"""
Test suite for DMET (Density Matrix Embedding Theory) implementation.

This module contains tests for the DMET package, serving both as
verification of the implementation and as usage examples.
"""

from types import SimpleNamespace

from pyscf import gto

from dmet import RHFDMET, DMETFragmentResult, Fragment, make_atom_fragments


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
        assert fragment.mean_field_fragment_energy < 0.0


def test_kernel_accumulates_fragment_correlation_corrections(monkeypatch) -> None:
    """Kernel should add fragment correlation corrections on top of RHF."""
    dmet = RHFDMET.__new__(RHFDMET)
    dmet.mf = SimpleNamespace(e_tot=-10.0)

    fragments = [
        Fragment("frag_a", (0,)),
        Fragment("frag_b", (1,)),
    ]
    fragment_results = {
        "frag_a": DMETFragmentResult(
            fragment=fragments[0],
            embedding_dimension=2,
            bath_dimension=1,
            core_dimension=0,
            nelec_active=2,
            fragment_electron_count=1.0,
            embedding_energy=-100.0,
            fragment_energy=-3.5,
            mean_field_fragment_energy=-3.0,
            bath_occupations=(0.5,),
        ),
        "frag_b": DMETFragmentResult(
            fragment=fragments[1],
            embedding_dimension=2,
            bath_dimension=1,
            core_dimension=0,
            nelec_active=2,
            fragment_electron_count=1.0,
            embedding_energy=-200.0,
            fragment_energy=-4.0,
            mean_field_fragment_energy=-4.25,
            bath_occupations=(0.5,),
        ),
    }

    def fake_solve_fragment(fragment: Fragment) -> DMETFragmentResult:
        return fragment_results[fragment.name]

    monkeypatch.setattr(dmet, "_solve_fragment", fake_solve_fragment)

    result = dmet.kernel(fragments)

    assert result.total_energy == -10.25
