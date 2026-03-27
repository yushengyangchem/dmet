# DMET

A minimal DMET prototype based on PySCF, currently implementing:

- RHF reference state
- Lowdin-orthogonalized AO local basis
- Environment 1-RDM based bath/core orbital construction
- Fragment embedding Hamiltonian
- FCI as embedding solver
- One-shot DMET total energy estimation

This version focuses on "minimum runnable", and does not yet include DMET correlation potential self-consistency, chemical potential fitting, or more rigorous fragment energy partitioning.

## Quick Start

```python
from pyscf import gto

from dmet import RHFDMET, make_atom_fragments

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

dmet = RHFDMET(mol)
result = dmet.kernel(make_atom_fragments(mol))

print("RHF energy:", result.mean_field_energy)
print("DMET energy estimate:", result.total_energy)
for fragment in result.fragments:
    print(
        fragment.fragment.name,
        "bath=",
        fragment.bath_dimension,
        "embed=",
        fragment.embedding_dimension,
        "E_emb=",
        fragment.embedding_energy,
    )
```

## API

- `Fragment(name, ao_indices)`: Manually define a fragment
- `make_atom_fragments(mol)`: One fragment per atom
- `RHFDMET(mol, mf=None)`: Build a one-shot DMET solver
- `RHFDMET.kernel(fragments)`: Returns `DMETResult`

For minimal end-to-end usage, you can also check the test file [test_dmet.py](/home/yangys/projects/dmet/tests/test_dmet.py).

## Development

```bash
just test
```

## License

MIT
