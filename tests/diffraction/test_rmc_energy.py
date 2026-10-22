import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

from quantem.diffraction.rmc_energy import ClusterEnergy, RelaxedSwapEnergy

A = 3.37
R_CUT = 4.0


class LocalManyBody(Calculator):
    """Species-dependent pair term plus an embedding term, cut off sharply at ``rc``: each
    per-atom energy depends nonlinearly on every neighbour within ``rc`` and on nothing else."""

    implemented_properties = ["energy", "energies"]

    def __init__(self, rc):
        super().__init__()
        self.rc = rc

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        i, j, d = neighbor_list("ijd", self.atoms, self.rc)
        z = self.atoms.numbers
        eps = 0.02 * (1 + (z[i] % 7) * (z[j] % 7) / 10)
        pair = 4 * eps * ((2.6 / d) ** 12 - (2.6 / d) ** 6)
        rho = np.bincount(i, (1 + z[j] % 5) * np.exp(-d), minlength=len(atoms))
        e = 0.5 * np.bincount(i, pair, minlength=len(atoms)) - 0.3 * np.sqrt(rho)
        self.results = {"energies": e, "energy": float(e.sum())}


def bcc(cells, seed):
    rng = np.random.default_rng(seed)
    idx = np.stack(np.meshgrid(*(np.arange(cells),) * 3, indexing="ij"), -1).reshape(-1, 3)
    frac = np.concatenate([idx, idx + 0.5]) / cells
    numbers = rng.choice([41, 23, 40], size=len(frac), p=[0.33, 0.22, 0.45])
    pos = frac * A * cells + rng.normal(0, 0.05, frac.shape)
    return numbers, pos, A * cells


def setup(cells, seed=0):
    numbers, pos, L = bcc(cells, seed)
    return ClusterEnergy(LocalManyBody(R_CUT), R_CUT, numbers, pos, L)


def full_delta(ce, index, numbers=None, positions=None):
    num, pos = ce.numbers.copy(), ce.positions.copy()
    if numbers is not None:
        num[index] = numbers
    if positions is not None:
        pos[index] = positions
    return float(ce.full_energies(num, pos).sum() - ce.energies.sum())


# 3 cells (10 A) is smaller than the 8.5 A cluster radius, so atoms enter as several images
@pytest.mark.parametrize("cells", [3, 6])
def test_swap_matches_full_recompute(cells):
    ce = setup(cells)
    rng = np.random.default_rng(1)
    for _ in range(10):
        i, j = rng.choice(len(ce.numbers), 2, replace=False)
        if ce.numbers[i] == ce.numbers[j]:
            continue
        idx, num = [i, j], [ce.numbers[j], ce.numbers[i]]
        dE, _, _ = ce.delta(idx, numbers=num)
        assert dE == pytest.approx(full_delta(ce, idx, numbers=num), abs=1e-9)


@pytest.mark.parametrize("cells", [3, 6])
def test_displacements_match_full_recompute(cells):
    ce = setup(cells)
    rng = np.random.default_rng(2)
    for _ in range(10):
        i = int(rng.integers(len(ce.numbers)))
        new = ce.positions[i] + rng.normal(0, 0.2, 3)
        dE, _, _ = ce.delta(i, positions=new)
        assert dE == pytest.approx(full_delta(ce, [i], positions=new[None]), abs=1e-9)


def test_three_atom_move_across_the_boundary():
    """An omega-like (0, +v, -v) move on three atoms of a <111> row through the cell edge."""
    ce = setup(6)
    L = ce.cell[0]
    corner = np.argmin(np.linalg.norm((ce.positions + 0.5) % L - 0.5, axis=1))
    row = [corner]
    for t in (1, 2):
        target = (ce.positions[corner] + t * A / 2 * np.array([1, 1, 1])) % L
        d = ce.positions - target
        d -= L * np.round(d / L)
        row.append(int(np.argmin(np.linalg.norm(d, axis=1))))
    v = A / 12 * np.array([1, 1, 1])
    new = ce.positions[row] + np.stack([0 * v, v, -v])
    dE, _, _ = ce.delta(row, positions=new)
    assert dE == pytest.approx(full_delta(ce, row, positions=new), abs=1e-9)


def test_applied_moves_keep_the_cache_exact():
    ce = setup(5)
    rng = np.random.default_rng(3)
    for k in range(20):
        if k % 2:
            i, j = rng.choice(len(ce.numbers), 2, replace=False)
            idx, num, pos = [i, j], [ce.numbers[j], ce.numbers[i]], None
        else:
            idx, num = [int(rng.integers(len(ce.numbers)))], None
            pos = ce.positions[idx] + rng.normal(0, 0.1, (1, 3))
        dE, affected, new_e = ce.delta(idx, numbers=num, positions=pos)
        total = ce.total
        ce.apply(idx, numbers=num, positions=pos, affected=affected, new_energies=new_e)
        assert ce.total == pytest.approx(total + dE, abs=1e-9)
    np.testing.assert_allclose(ce.energies, ce.full_energies(), atol=1e-9)
    assert abs(ce.rebuild()) < 1e-9


def test_distant_atoms_give_separate_clusters():
    ce = setup(6)
    L = ce.cell[0]
    i = 0
    d = ce.positions - ce.positions[i]
    d -= L * np.round(d / L)
    j = int(np.argmax(np.linalg.norm(d, axis=1)))  # about half a cell (17 A) away
    assert len(ce._groups(ce.positions[[i, j]])) == 2
    num = [ce.numbers[j], ce.numbers[i]]
    dE, _, _ = ce.delta([i, j], numbers=num)
    assert dE == pytest.approx(full_delta(ce, [i, j], numbers=num), abs=1e-9)
    assert ce.separation == ce.radius


class SmoothManyBody(Calculator):
    """Species-dependent Morse pairs plus an embedding term (as in EAM), all going smoothly to
    zero at ``rc``, with forces. Through the embedding, the force on an atom depends on atoms up
    to ``2 rc`` away, as for a many-body potential such as Allegro."""

    implemented_properties = ["energy", "energies", "forces"]

    def __init__(self, rc):
        super().__init__()
        self.rc = rc

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(self.atoms)
        i, j, d, D = neighbor_list("ijdD", self.atoms, self.rc)
        z = self.atoms.numbers
        cut, dcut = (1 - (d / self.rc) ** 2) ** 2, -4 * d / self.rc**2 * (1 - (d / self.rc) ** 2)
        eps = 0.05 + 0.002 * (z[i] % 7) * (z[j] % 7)
        x = d - (2.7 + 0.03 * ((z[i] + z[j]) % 5))
        morse, dmorse = (
            np.exp(-4 * x) - 2 * np.exp(-2 * x),
            -4 * np.exp(-4 * x) + 4 * np.exp(-2 * x),
        )
        phi, dphi = eps * morse * cut, eps * (dmorse * cut + morse * dcut)
        w = 1 + (z % 5) / 4  # each species' contribution to its neighbours' density
        g, dg = np.exp(-d) * cut, np.exp(-d) * (dcut - cut)
        rho = np.bincount(i, w[j] * g, minlength=n)
        embed, dembed = -0.3 * np.sqrt(1 + rho), -0.15 / np.sqrt(1 + rho)
        e = 0.5 * np.bincount(i, phi, minlength=n) + embed
        c = dphi + (dembed[i] * w[j] + dembed[j] * w[i]) * dg  # dE / d r_ij over both ends
        f = np.zeros((n, 3))
        np.add.at(f, i, (c / d)[:, None] * D)  # D = r_j - r_i
        self.results = {"energies": e, "energy": float(e.sum()), "forces": f}


def relaxed_setup(relax_radius=3.0, cells=6, seed=4):
    numbers, pos, L = bcc(cells, seed)
    return RelaxedSwapEnergy(
        SmoothManyBody(R_CUT), R_CUT, numbers, pos, L, relax_radius, fmax=1e-4, max_steps=1000
    )


def energy_of(numbers, positions, cell, calc):
    atoms = Atoms(numbers=numbers, positions=positions, cell=np.diag(np.full(3, cell)), pbc=True)
    atoms.calc = calc
    return float(atoms.get_potential_energy())


def test_smooth_many_body_forces_are_the_energy_gradient():
    numbers, pos, L = bcc(3, 5)
    calc = SmoothManyBody(R_CUT)
    atoms = Atoms(numbers=numbers, positions=pos, cell=np.diag(np.full(3, L)), pbc=True)
    atoms.calc = calc
    f = atoms.get_forces()[7, 1]
    h = 1e-5
    e = []
    for s in (h, -h):
        moved = pos.copy()
        moved[7, 1] += s
        e.append(energy_of(numbers, moved, L, calc))
    assert f == pytest.approx(-(e[0] - e[1]) / (2 * h), rel=1e-5, abs=1e-8)


def test_relaxed_swap_matches_relaxing_the_same_atoms_in_the_whole_cell():
    ce = relaxed_setup()
    L = ce.cell[0]
    rng = np.random.default_rng(6)
    for _ in range(3):
        i, j = (int(k) for k in rng.choice(len(ce.numbers), 2, replace=False))
        if ce.numbers[i] == ce.numbers[j]:
            continue
        num = [ce.numbers[j], ce.numbers[i]]
        dE, affected, (new_positions, _) = ce.delta([i, j], numbers=num)
        # the same relaxation in the periodic cell: only atoms within relax_radius of the pair move
        d = np.stack([ce.positions - ce.positions[k] for k in (i, j)])
        d -= L * np.round(d / L)
        free = np.linalg.norm(d, axis=-1).min(0) <= ce.relax_radius
        swapped = ce.numbers.copy()
        swapped[[i, j]] = num
        atoms = Atoms(numbers=swapped, positions=ce.positions, cell=np.diag(ce.cell), pbc=True)
        e_after = ce._relax(atoms, free)
        e_before = energy_of(ce.numbers, ce.positions, L, ce.calc)
        assert dE == pytest.approx(e_after - e_before, abs=1e-7)
        assert sorted(affected) == sorted(np.nonzero(free)[0])
        moved = new_positions - np.mod(atoms.positions[affected], L)
        moved -= L * np.round(moved / L)
        np.testing.assert_allclose(moved, 0, atol=1e-6)


def test_applied_relaxed_swaps_keep_the_total_exact():
    ce = relaxed_setup()
    rng = np.random.default_rng(7)
    for _ in range(6):
        i, j = (int(k) for k in rng.choice(len(ce.numbers), 2, replace=False))
        if ce.numbers[i] == ce.numbers[j]:
            continue
        num = [ce.numbers[j], ce.numbers[i]]
        dE, affected, relaxed = ce.delta([i, j], numbers=num)
        total = ce.total
        ce.apply([i, j], numbers=num, affected=affected, new_energies=relaxed)
        assert ce.total == pytest.approx(total + dE, abs=1e-12)
    assert ce.total == pytest.approx(
        energy_of(ce.numbers, ce.positions, ce.cell[0], ce.calc), abs=1e-7
    )


def test_relaxed_swaps_refuse_moves_and_small_cells():
    ce = relaxed_setup()
    assert ce.separation == pytest.approx(2 * ce.relax_radius + 2 * R_CUT + 0.5)
    with pytest.raises(NotImplementedError, match="swaps only"):
        ce.delta(0, positions=ce.positions[0] + 0.1)
    with pytest.raises(ValueError, match="must be larger"):
        relaxed_setup(relax_radius=3.0, cells=4)
