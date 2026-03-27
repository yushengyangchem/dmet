"""
Core DMET implementation module.

This module provides the core classes and functions for performing Density Matrix
Embedding Theory (DMET) calculations using PySCF as the underlying quantum chemistry
framework.

Key Components:
    - Fragment: Represents a spatial fragment of the molecular system
    - RHFDMET: Main DMET solver class using RHF reference
    - make_atom_fragments: Utility to create one fragment per atom
    - DMETResult: Container for DMET calculation results
    - DMETFragmentResult: Container for individual fragment results

The implementation follows a one-shot DMET approach with:
    1. RHF mean-field calculation for reference
    2. Lowdin orthogonalization of AO basis
    3. Bath construction from environment 1-RDM
    4. FCI solving of embedding Hamiltonian
    5. Energy estimation from fragment averages
"""

from dataclasses import dataclass

import numpy as np
from pyscf import ao2mo, fci, gto, scf


@dataclass(frozen=True)
class Fragment:
    """Represents a fragment of the molecular system for DMET calculation.

    A fragment defines a subset of atomic orbitals that will be treated
    with high-level quantum chemistry methods while the rest of the system
    is represented at a lower level.

    Attributes:
        name: A descriptive identifier for the fragment.
        ao_indices: Tuple of atomic orbital indices belonging to this fragment.

    Example:
        >>> fragment = Fragment("H1", ao_indices=(0, 1))
        >>> print(fragment.name)
        'H1'
    """

    name: str
    ao_indices: tuple[int, ...]


@dataclass(frozen=True)
class DMETFragmentResult:
    """Results from solving a single fragment in DMET.

    Contains all information about the embedding calculation for one fragment,
    including dimensions, electron counts, energies, and bath orbital occupations.

    Attributes:
        fragment: The Fragment object that was solved.
        embedding_dimension: Total number of orbitals in the embedding space
            (impurity + bath orbitals).
        bath_dimension: Number of bath orbitals in the embedding.
        core_dimension: Number of core (doubly occupied environment) orbitals.
        nelec_active: Number of active electrons in the embedding problem.
        fragment_electron_count: Electron count on the impurity orbitals from
            the embedded wavefunction.
        embedding_energy: Total energy of the embedding problem (including
            core contribution).
        fragment_energy: Energy contribution attributed to the impurity orbitals.
        bath_occupations: Occupation numbers of bath orbitals, indicating
            their entanglement with the impurity.
    """

    fragment: Fragment
    embedding_dimension: int
    bath_dimension: int
    core_dimension: int
    nelec_active: int
    fragment_electron_count: float
    embedding_energy: float
    fragment_energy: float
    bath_occupations: tuple[float, ...]


@dataclass(frozen=True)
class DMETResult:
    """Complete results from a DMET calculation.

    Contains the overall DMET calculation results including energies and
    individual fragment results.

    Attributes:
        mean_field_energy: Total energy from the reference RHF calculation.
        total_energy: Estimated total energy from DMET (averaged over fragments).
        fragments: Tuple of results for each fragment that was solved.

    Note:
        The total_energy is currently computed as a simple average of fragment
        embedding energies. More sophisticated energy partitioning schemes
        are not yet implemented.
    """

    mean_field_energy: float
    total_energy: float
    fragments: tuple[DMETFragmentResult, ...]


def make_atom_fragments(mol: gto.Mole) -> list[Fragment]:
    """Create one fragment for each atom in the molecule.

    This utility function automatically partitions the molecular system
    into atomic fragments based on the AO-to-atom mapping from PySCF.
    Each atom gets its own fragment containing all AOs centered on that atom.

    Args:
        mol: PySCF Mole object representing the molecular system.

    Returns:
        A list of Fragment objects, one per atom, with AO indices assigned
        according to the atomic centers.

    Example:
        >>> from pyscf import gto
        >>> mol = gto.M(atom="H 0 0 0; H 0 0 0.8", basis="sto-3g")
        >>> fragments = make_atom_fragments(mol)
        >>> print(len(fragments))
        2

    Note:
        The AO indices are determined by mol.aoslice_by_atom(), which
        provides the starting and ending AO index for each atom.
    """
    slices = mol.aoslice_by_atom()
    fragments: list[Fragment] = []
    for atom_index, (_, _, p0, p1) in enumerate(slices):
        fragments.append(
            Fragment(
                name=f"atom_{atom_index}",
                ao_indices=tuple(range(p0, p1)),
            )
        )
    return fragments


def _symmetric_orthogonalizer(
    overlap: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute symmetric (Lowdin) orthogonalization matrices.

    Given an overlap matrix S, computes the matrices S^{-1/2} and S^{1/2}
    that can be used to transform between orthogonal and non-orthogonal
    basis sets.

    Args:
        overlap: The overlap matrix S (n x n) in the AO basis.
        threshold: Minimum eigenvalue to keep. Eigenvectors with eigenvalues
            below this threshold are discarded to avoid numerical instability.

    Returns:
        A tuple (S^{-1/2}, S^{1/2}) where:
        - S^{-1/2} transforms from AO to orthogonal basis
        - S^{1/2} transforms from orthogonal back to AO basis

    Raises:
        ValueError: If no eigenvalues exceed the threshold, indicating
            no linearly independent AO functions.

    Note:
        The threshold should be set appropriately for the basis set being
        used. Default value of 1e-10 is usually sufficient for most cases.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    keep = eigenvalues > threshold
    if not np.any(keep):
        raise ValueError(
            "No linearly independent AO functions survived the overlap threshold."
        )

    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    inv_sqrt = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T
    sqrt = eigenvectors @ np.diag(eigenvalues**0.5) @ eigenvectors.T
    return inv_sqrt, sqrt


def _projector_columns(indices: tuple[int, ...], dimension: int) -> np.ndarray:
    """Create a projector matrix that selects specific basis functions.

    Constructs a matrix that projects onto a subspace spanned by the
    specified basis function indices. The resulting matrix can be used
    to embed a smaller basis into a larger one.

    Args:
        indices: Tuple of basis function indices to project onto.
        dimension: Total dimension of the full basis space.

    Returns:
        A (dimension x len(indices)) matrix where each column is a unit
        vector selecting one of the specified basis functions.

    Example:
        >>> P = _projector_columns((1, 3), 4)
        >>> print(P)
        [[0. 0.]
         [1. 0.]
         [0. 0.]
         [0. 1.]]
    """
    basis = np.zeros((dimension, len(indices)))
    for column, index in enumerate(indices):
        basis[index, column] = 1.0
    return basis


class RHFDMET:
    """A minimal one-shot DMET implementation on top of PySCF RHF.

    This class implements Density Matrix Embedding Theory using a restricted
    Hartree-Fock reference state. It performs the following steps:

    1. Runs or accepts an RHF calculation to obtain mean-field orbitals
    2. Orthogonalizes the AO basis using Lowdin symmetric orthogonalization
    3. For each fragment:
       - Constructs bath orbitals from the environment 1-RDM
       - Builds an embedding Hamiltonian in the impurity+bath space
       - Solves the embedding problem with FCI
       - Computes fragment energies
    4. Estimates total energy by averaging fragment embedding energies

    The total energy returned by :meth:`kernel` is a simple estimator formed by
    averaging the embedded fragment energies. This keeps the first version
    numerically stable before adding the usual DMET correlation-potential
    self-consistency and a more careful fragment energy partition.

    Attributes:
        mol: PySCF Mole object for the molecular system.
        mf: PySCF RHF object containing the mean-field solution.
        occupancy_tolerance: Threshold for treating occupations as 0 or 1.
        overlap_tolerance: Threshold for linear dependence in overlap matrix.
        orthogonalizer: Matrix S^{-1/2} for orthogonalization.
        deorthogonalizer: Matrix S^{1/2} for inverse transformation.
        occupied_mo_coeff: MO coefficients for occupied orbitals.
        occupied_orth_coeff: Occupied orbitals in orthogonal basis.
        reference_density_orth: 1-RDM in orthogonal basis (idempotent).

    Example:
        >>> from pyscf import gto
        >>> from dmet import RHFDMET, make_atom_fragments
        >>>
        >>> mol = gto.M(atom="H 0 0 0; H 0 0 0.8", basis="sto-3g", verbose=0)
        >>> dmet = RHFDMET(mol)
        >>> result = dmet.kernel(make_atom_fragments(mol))
        >>> print(f"RHF energy: {result.mean_field_energy:.6f}")
        >>> print(f"DMET energy: {result.total_energy:.6f}")

    References:
        - Knizia, G., & Chan, G. K. L. (2012). Physical Review Letters, 109(26).
    """

    def __init__(
        self,
        mol: gto.Mole,
        *,
        mf: scf.hf.RHF | None = None,
        occupancy_tolerance: float = 1e-8,
        overlap_tolerance: float = 1e-10,
    ) -> None:
        """Initialize the DMET solver with a molecular system.

        Args:
            mol: PySCF Mole object representing the molecular system.
            mf: Optional pre-computed RHF solution. If None, a new RHF
                calculation will be performed automatically.
            occupancy_tolerance: Numerical threshold for treating occupations
                as exactly 0 or 1. Used in bath/core orbital selection.
                Default: 1e-8.
            overlap_tolerance: Threshold for removing linearly dependent
                AOs during orthogonalization. Default: 1e-10.

        Note:
            The tolerance parameters should be adjusted based on the
            numerical precision of your system. For large basis sets,
            you may need to increase overlap_tolerance.
        """
        self.mol = mol
        self.mf = mf or scf.RHF(mol).run()
        self.occupancy_tolerance = occupancy_tolerance
        self.overlap_tolerance = overlap_tolerance

        overlap = self.mf.get_ovlp()
        self.orthogonalizer, self.deorthogonalizer = _symmetric_orthogonalizer(
            overlap,
            overlap_tolerance,
        )
        occupied = self.mf.mo_occ > occupancy_tolerance
        self.occupied_mo_coeff = self.mf.mo_coeff[:, occupied]
        self.occupied_orth_coeff = self.deorthogonalizer @ self.occupied_mo_coeff
        self.reference_density_orth = (
            self.occupied_orth_coeff @ self.occupied_orth_coeff.T
        )

    def kernel(self, fragments: list[Fragment]) -> DMETResult:
        """Run the DMET calculation for all fragments.

        Performs the complete DMET calculation by solving each fragment's
        embedding problem and combining the results.

        Args:
            fragments: List of Fragment objects defining the partitioning
                of the molecular system. Each fragment will be solved
                independently with its own embedding space.

        Returns:
            DMETResult object containing:
            - mean_field_energy: RHF reference energy
            - total_energy: DMET energy estimate (average over fragments)
            - fragments: Detailed results for each fragment

        Raises:
            ValueError: If fragments list is empty.

        Example:
            >>> dmet = RHFDMET(mol)
            >>> result = dmet.kernel(make_atom_fragments(mol))
            >>> print(f"DMET energy: {result.total_energy:.6f}")

        Note:
            The current implementation uses a simple average of fragment
            embedding energies. For more accurate energies, consider
            implementing a proper energy partitioning scheme.
        """
        if not fragments:
            raise ValueError("At least one fragment is required.")

        fragment_results = tuple(
            self._solve_fragment(fragment) for fragment in fragments
        )
        total_energy = float(
            np.mean([result.embedding_energy for result in fragment_results])
        )
        return DMETResult(
            mean_field_energy=float(self.mf.e_tot),
            total_energy=float(total_energy),
            fragments=fragment_results,
        )

    def _solve_fragment(self, fragment: Fragment) -> DMETFragmentResult:
        """Solve the embedding problem for a single fragment.

        This is the core DMET routine that:
        1. Constructs bath orbitals from environment 1-RDM eigenvectors
        2. Identifies and factor out core (doubly occupied) orbitals
        3. Builds the embedding Hamiltonian (1e + 2e + core potential)
        4. Solves with FCI to get embedded wavefunction
        5. Extracts fragment energy from reduced density matrices

        Args:
            fragment: The fragment to solve, specifying impurity AO indices.

        Returns:
            DMETFragmentResult containing all computed properties for this fragment.

        Raises:
            ValueError: If fragment has no orbitals or negative active electron count.
            NotImplementedError: If odd number of active electrons (open shell not supported).

        Note:
            The bath construction follows the standard DMET procedure:
            - Diagonalize environment 1-RDM
            - Select orbitals with occupations between (0, 1) as bath
            - Limit bath size to impurity size by entanglement ordering
            - Core orbitals (occupation ~1) are treated as frozen
        """
        n_orbitals = self.reference_density_orth.shape[0]
        impurity = np.array(fragment.ao_indices, dtype=int)
        if impurity.size == 0:
            raise ValueError(
                f"Fragment {fragment.name!r} does not contain any orbitals."
            )

        env_mask = np.ones(n_orbitals, dtype=bool)
        env_mask[impurity] = False
        env_indices = np.flatnonzero(env_mask)

        density_env = self.reference_density_orth[np.ix_(env_indices, env_indices)]
        occupations, eigenvectors = np.linalg.eigh(density_env)
        order = np.argsort(occupations)[::-1]
        occupations = occupations[order]
        eigenvectors = eigenvectors[:, order]

        bath_selector = (occupations > self.occupancy_tolerance) & (
            occupations < 1.0 - self.occupancy_tolerance
        )
        bath_occupations = occupations[bath_selector]
        bath_vectors_env = eigenvectors[:, bath_selector]
        if bath_vectors_env.shape[1] > impurity.size:
            entanglement_order = np.argsort(np.abs(bath_occupations - 0.5))
            entanglement_order = entanglement_order[: impurity.size]
            bath_occupations = bath_occupations[entanglement_order]
            bath_vectors_env = bath_vectors_env[:, entanglement_order]

        core_selector = occupations >= 1.0 - self.occupancy_tolerance
        core_vectors_env = eigenvectors[:, core_selector]

        impurity_basis = _projector_columns(fragment.ao_indices, n_orbitals)
        bath_basis = np.zeros((n_orbitals, bath_vectors_env.shape[1]))
        bath_basis[np.ix_(env_indices, np.arange(bath_vectors_env.shape[1]))] = (
            bath_vectors_env
        )

        core_basis = np.zeros((n_orbitals, core_vectors_env.shape[1]))
        core_basis[np.ix_(env_indices, np.arange(core_vectors_env.shape[1]))] = (
            core_vectors_env
        )

        embedding_basis_orth = np.concatenate((impurity_basis, bath_basis), axis=1)
        embedding_basis_ao = self.orthogonalizer @ embedding_basis_orth
        core_basis_ao = self.orthogonalizer @ core_basis

        hcore = self.mf.get_hcore()
        if core_basis_ao.shape[1] > 0:
            dm_core = 2.0 * core_basis_ao @ core_basis_ao.T
            core_potential = scf.hf.get_veff(self.mol, dm_core)
            core_energy = (
                self.mol.energy_nuc()
                + np.einsum("ij,ji->", hcore, dm_core)
                + 0.5 * np.einsum("ij,ji->", core_potential, dm_core)
            )
        else:
            core_potential = np.zeros_like(hcore)
            core_energy = self.mol.energy_nuc()

        h1_emb = embedding_basis_ao.T @ (hcore + core_potential) @ embedding_basis_ao
        n_active = embedding_basis_ao.shape[1]
        nelec_active = int(round(self.mol.nelectron - 2 * core_basis_ao.shape[1]))
        if nelec_active < 0:
            raise ValueError(
                f"Fragment {fragment.name!r} produced a negative active electron count."
            )
        if nelec_active % 2 != 0:
            raise NotImplementedError(
                "This minimal implementation currently assumes a closed-shell active space."
            )

        eri_emb = ao2mo.restore(
            1,
            ao2mo.kernel(self.mol, embedding_basis_ao, compact=False),
            n_active,
        )

        solver = fci.direct_spin1.FCI(self.mol)
        electrons = (nelec_active // 2, nelec_active // 2)
        embedding_energy, ci_vector = solver.kernel(
            h1_emb,
            eri_emb,
            n_active,
            electrons,
            ecore=core_energy,
        )
        rdm1, rdm2 = solver.make_rdm12(ci_vector, n_active, electrons)

        impurity_slice = slice(0, impurity.size)
        fragment_energy = float(
            np.einsum("ij,ji->", h1_emb[impurity_slice, :], rdm1[:, impurity_slice])
            + 0.5
            * np.einsum(
                "ijkl,ijkl->",
                eri_emb[impurity_slice, :, :, :],
                rdm2[impurity_slice, :, :, :],
            )
        )

        return DMETFragmentResult(
            fragment=fragment,
            embedding_dimension=n_active,
            bath_dimension=bath_basis.shape[1],
            core_dimension=core_basis.shape[1],
            nelec_active=nelec_active,
            fragment_electron_count=float(
                np.trace(rdm1[impurity_slice, impurity_slice])
            ),
            embedding_energy=float(embedding_energy),
            fragment_energy=float(fragment_energy),
            bath_occupations=tuple(float(value) for value in bath_occupations),
        )
