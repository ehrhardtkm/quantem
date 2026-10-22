"""Exact energy changes of local Monte Carlo moves from a strictly local interatomic potential.

A strictly local potential writes the total energy as a sum of per-atom energies, each a
function only of the atoms within a cut-off ``r_cut`` of that atom. This holds for Allegro, NEP,
ACE and EAM, but not for message-passing models, whose range is the number of layers times the
cut-off. Moving a few atoms (or exchanging their species) then changes the energy of the atoms
within ``r_cut`` of an old or new position of a moved atom, the affected atoms, and of no other.
Their new energies depend only on atoms within ``r_cut`` of them, so a cluster of every atom
within ``2 r_cut + margin`` of a moved atom, cut from the moved configuration and evaluated in
vacuum, gives them exactly. The energy change of the move is their new energies minus their
cached ones; the cache holds every atom's energy from one evaluation of the whole periodic cell
and is patched whenever a move is applied.

Periodic images are enumerated explicitly, so the cell may be smaller than the cluster (an atom
can then enter a cluster more than once, as distinct images). Moved atoms far apart give
separate clusters, evaluated in one call side by side in vacuum.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from ase import Atoms


class ClusterEnergy:
    """Per-atom energy cache of a periodic orthorhombic cell, and exact cluster energy changes.

    Parameters
    ----------
    calculator : ase.calculators.calculator.Calculator
        Reports per-atom ``energies`` (eV) and is strictly local within ``r_cut``.
    r_cut : float
        Cut-off (A) beyond which no per-atom energy depends on any position.
    numbers : (N,) int
        Atomic numbers.
    positions : (N, 3) float
        Cartesian positions (A).
    cell : float or (3,) float
        Edge lengths (A) of the periodic orthorhombic cell.
    margin : float
        Extra cluster radius (A), insurance against an understated ``r_cut``.
    """

    def __init__(self, calculator, r_cut, numbers, positions, cell, margin: float = 0.5):
        self.calc = calculator
        self.r_cut = float(r_cut)
        self.radius = 2 * self.r_cut + float(margin)
        self.numbers = np.array(numbers, dtype=int)
        self.positions = np.array(positions, dtype=float)
        self.cell = np.broadcast_to(np.asarray(cell, dtype=float), (3,)).copy()
        self.energies = np.zeros(len(self.numbers))
        self.rebuild()

    @property
    def total(self) -> float:
        """Total energy (eV) of the cell from the cache."""
        return float(self.energies.sum())

    @property
    def separation(self) -> float:
        """Moves farther apart than this (A) change disjoint sets of atom energies, so their
        energy changes add exactly."""
        return self.radius

    def _evaluate(self, atoms: Atoms) -> np.ndarray:
        self.calc.calculate(atoms, ["energies"])
        return np.asarray(self.calc.results["energies"], dtype=np.float64)

    def full_energies(self, numbers=None, positions=None) -> np.ndarray:
        """Per-atom energies (eV) of the whole periodic cell, by default the current one."""
        atoms = Atoms(
            numbers=self.numbers if numbers is None else numbers,
            positions=self.positions if positions is None else positions,
            cell=np.diag(self.cell),
            pbc=True,
        )
        return self._evaluate(atoms)

    def rebuild(self) -> float:
        """Recompute every per-atom energy; returns the drift (eV) of the cached total."""
        new = self.full_energies()
        drift = float(new.sum() - self.energies.sum())
        self.energies = new
        return drift

    def _mic(self, d: np.ndarray) -> np.ndarray:
        return d - self.cell * np.round(d / self.cell)

    def _groups(self, centres: np.ndarray) -> list[np.ndarray]:
        """Centres split into groups whose clusters do not overlap; each group in one frame."""
        groups: list[list[np.ndarray]] = []
        for c in centres:
            near = [
                k
                for k, g in enumerate(groups)
                if min(np.linalg.norm(self._mic(c - h)) for h in g) < 2 * self.radius
            ]
            if not near:
                groups.append([c])
                continue
            merged = groups[near[0]]
            for k in near[1:][::-1]:
                merged += [merged[0] + self._mic(h - merged[0]) for h in groups.pop(k)]
            merged.append(merged[0] + self._mic(c - merged[0]))
        return [np.array(g) for g in groups]

    def _gather(self, positions: np.ndarray, centres: np.ndarray):
        """Every periodic image of every atom within ``radius`` of a centre.

        Returns the atom index of each image (M,), its position (M, 3) and its distance to the
        nearest centre (M,). Each (atom, lattice shift) appears once.
        """
        mid = centres.mean(0)
        spread = float(np.max(np.linalg.norm(centres - mid, axis=1)))
        base = mid + self._mic(positions - mid)
        reach = np.floor((self.radius + spread) / self.cell + 0.5).astype(int)
        idx, xyz, dist = [], [], []
        everyone = np.arange(len(positions))
        for shift in product(*(range(-n, n + 1) for n in reach)):
            x = base + np.array(shift) * self.cell
            d = np.min(np.linalg.norm(x[:, None, :] - centres[None], axis=-1), axis=1)
            keep = d <= self.radius
            idx.append(everyone[keep])
            xyz.append(x[keep])
            dist.append(d[keep])
        return np.concatenate(idx), np.concatenate(xyz), np.concatenate(dist)

    def delta(self, index, numbers=None, positions=None):
        """Exact energy change (eV) of moving atoms ``index`` to ``positions`` and/or giving
        them atomic numbers ``numbers``. Nothing is changed; see :meth:`apply`.

        Returns
        -------
        dE : float
        affected : (n,) int
            Atoms whose per-atom energy may have changed.
        new_energies : (n,) float
            Their per-atom energies after the move.
        """
        index = np.atleast_1d(np.asarray(index, dtype=int))
        num, pos = self.numbers, self.positions
        if numbers is not None:
            num = num.copy()
            num[index] = numbers
        if positions is not None:
            pos = pos.copy()
            pos[index] = np.asarray(positions, dtype=float).reshape(len(index), 3)
        centres = self.positions[index]
        if positions is not None:
            centres = np.concatenate([centres, pos[index]])

        # gather each group's cluster from the moved configuration; pack them in one vacuum box
        packed_pos, packed_num, parts = [], [], []
        offset, n0 = 0.0, 0
        for g in self._groups(centres):
            idx, xyz, dist = self._gather(pos, g)
            near = np.nonzero(dist <= self.r_cut)[0]
            order = near[np.lexsort((dist[near], idx[near]))]
            first = np.ones(len(order), dtype=bool)
            first[1:] = idx[order][1:] != idx[order][:-1]
            chosen = order[first]  # the image of each affected atom nearest a centre
            xyz = xyz - xyz.min(0) + [offset, 0.0, 0.0]
            offset = xyz[:, 0].max() + self.r_cut + 1.0
            packed_pos.append(xyz)
            packed_num.append(num[idx])
            parts.append((idx[chosen], n0 + chosen))
            n0 += len(idx)
        xyz = np.concatenate(packed_pos) + self.r_cut + 1.0
        box = xyz.max(0) + self.r_cut + 1.0
        atoms = Atoms(numbers=np.concatenate(packed_num), positions=xyz, cell=np.diag(box))
        e = self._evaluate(atoms)

        affected = np.concatenate([a for a, _ in parts])
        new_energies = np.concatenate([e[rows] for _, rows in parts])
        if len(np.unique(affected)) != len(affected):
            raise AssertionError("an atom is affected through two separate clusters")
        dE = float((new_energies - self.energies[affected]).sum())
        return dE, affected, new_energies

    def apply(self, index, numbers=None, positions=None, affected=None, new_energies=None):
        """Apply a move (the arguments of :meth:`delta` and the last two values it returned)."""
        index = np.atleast_1d(np.asarray(index, dtype=int))
        if numbers is not None:
            self.numbers[index] = numbers
        if positions is not None:
            self.positions[index] = np.asarray(positions, dtype=float).reshape(len(index), 3)
        if affected is not None:
            self.energies[affected] = new_energies


class RelaxedSwapEnergy(ClusterEnergy):
    """Relaxed energy changes of species swaps, from a strictly local potential with forces.

    Keeps its own relaxed copy of the cell's positions, separate from any other: the whole periodic
    cell is relaxed once at the start, and again by :meth:`rebuild`. A swap's energy change is
    taken between relaxed states: every atom within ``relax_radius + 2 r_cut + margin`` of the
    swapped pair is cut from the relaxed cell (as :class:`ClusterEnergy` cuts its clusters), the
    pair swapped, and the atoms within ``relax_radius`` of it relaxed (LBFGS until the largest force
    is below ``fmax``) with the rest of the cluster held. The forces on the relaxing atoms are then
    exact, and the cluster's energy changes by exactly as much as the cell's, since atoms farther
    than ``relax_radius + r_cut`` from the pair see nothing change. Applying a swap stores the
    relaxed positions. Only swaps are scored: moving atoms is refused.

    Parameters
    ----------
    calculator : ase.calculators.calculator.Calculator
        Reports ``energy`` and ``forces`` and is strictly local within ``r_cut``, e.g. from
        :func:`nequip_force_calculator`.
    r_cut : float
        Cut-off (A) beyond which no per-atom energy depends on any position.
    numbers : (N,) int
        Atomic numbers.
    positions : (N, 3) float
        Cartesian positions (A) to relax from.
    cell : float or (3,) float
        Edge lengths (A) of the periodic orthorhombic cell.
    relax_radius : float
        Atoms within this distance (A) of a swapped atom relax.
    fmax : float
        A relaxation stops once the largest force on a relaxing atom is below this (eV/A).
    max_steps : int
        Most LBFGS steps per relaxation.
    margin : float
        Extra cluster radius (A), insurance against an understated ``r_cut``.
    """

    swaps_only = True

    def __init__(
        self,
        calculator,
        r_cut,
        numbers,
        positions,
        cell,
        relax_radius: float = 6.0,
        fmax: float = 0.05,
        max_steps: int = 200,
        margin: float = 0.5,
    ):
        self.calc = calculator
        self.r_cut = float(r_cut)
        self.relax_radius = float(relax_radius)
        self.fmax = float(fmax)
        self.max_steps = int(max_steps)
        self.radius = self.relax_radius + 2 * self.r_cut + float(margin)  # cluster cut per swap
        self._separation = 2 * self.relax_radius + 2 * self.r_cut + float(margin)
        self.numbers = np.array(numbers, dtype=int)
        self.positions = np.array(positions, dtype=float)
        self.cell = np.broadcast_to(np.asarray(cell, dtype=float), (3,)).copy()
        if self.cell.min() <= self._separation:
            # a relaxing atom's periodic images must stay out of every changed atom's reach
            raise ValueError(
                f"The cell ({self.cell.min():.1f} A) must be larger than "
                f"2 relax_radius + 2 r_cut + margin ({self._separation:.1f} A)."
            )
        self._total = 0.0
        self.rebuild()

    @property
    def total(self) -> float:
        """Total energy (eV) of the relaxed cell."""
        return self._total

    @property
    def separation(self) -> float:
        """Swaps farther apart than this (A) relax independently: neither's relaxing atoms are in
        the other's cluster, so their energy changes add exactly."""
        return self._separation

    def _relax(self, atoms: Atoms, free=None) -> float:
        """Relax ``atoms`` in place (only the ``free`` ones, if given); returns its energy."""
        from ase.constraints import FixAtoms
        from ase.optimize import LBFGS

        if free is not None:
            atoms.set_constraint(FixAtoms(indices=np.nonzero(~free)[0]))
        atoms.calc = self.calc
        LBFGS(atoms, logfile=None, maxstep=0.1).run(fmax=self.fmax, steps=self.max_steps)
        return float(atoms.get_potential_energy())

    def rebuild(self) -> float:
        """Relax the whole periodic cell from the stored positions; returns the change (eV) of
        the stored total."""
        atoms = Atoms(
            numbers=self.numbers, positions=self.positions, cell=np.diag(self.cell), pbc=True
        )
        e = self._relax(atoms)
        self.positions = np.mod(atoms.positions, self.cell)
        drift, self._total = e - self._total, e
        return drift

    def delta(self, index, numbers=None, positions=None):
        """Relaxed energy change (eV) of giving the atoms ``index`` (a swapped pair) the atomic
        numbers ``numbers``. Nothing is changed; see :meth:`apply`.

        Returns
        -------
        dE : float
        affected : (n,) int
            The atoms that relaxed.
        relaxed : tuple
            Their relaxed positions (n, 3) and dE, for :meth:`apply`.
        """
        if positions is not None:
            raise NotImplementedError("Relaxed energy changes are implemented for swaps only.")
        index = np.atleast_1d(np.asarray(index, dtype=int))
        num = self.numbers.copy()
        num[index] = numbers

        # each group's cluster from the relaxed cell, packed side by side in one vacuum box
        parts, offset = [], 0.0
        for g in self._groups(self.positions[index]):
            idx, xyz, dist = self._gather(self.positions, g)
            free = dist <= self.relax_radius
            if len(np.unique(idx[free])) != free.sum():
                raise AssertionError("an atom relaxes in two images")
            shift = [offset, 0.0, 0.0] - xyz.min(0) + self.r_cut + 1.0
            offset = (xyz + shift)[:, 0].max() + 1.0
            parts.append((idx, xyz + shift, free, np.broadcast_to(shift, xyz.shape)))
        idx = np.concatenate([p[0] for p in parts])
        xyz = np.concatenate([p[1] for p in parts])
        free = np.concatenate([p[2] for p in parts])
        shift = np.concatenate([p[3] for p in parts])
        box = np.diag(xyz.max(0) + self.r_cut + 1.0)

        start = Atoms(numbers=self.numbers[idx], positions=xyz, cell=box)
        start.calc = self.calc
        e0 = float(start.get_potential_energy())
        atoms = Atoms(numbers=num[idx], positions=xyz, cell=box)
        dE = self._relax(atoms, free) - e0
        rows = np.nonzero(free)[0]
        new_positions = np.mod(atoms.positions[rows] - shift[rows], self.cell)
        return dE, idx[rows], (new_positions, dE)

    def apply(self, index, numbers=None, positions=None, affected=None, new_energies=None):
        """Apply a swap: the arguments of :meth:`delta` and the last two values it returned."""
        index = np.atleast_1d(np.asarray(index, dtype=int))
        if numbers is not None:
            self.numbers[index] = numbers
        if affected is not None:
            new_positions, dE = new_energies
            self.positions[affected] = new_positions
            self._total += dE


def nequip_energy_calculator(compiled_path, device: str = "cuda"):
    """ASE calculator for a NequIP or Allegro model compiled to output only per-atom and total
    energies (no forces), and its cut-off.

    Monte Carlo needs no forces, and leaving them out saves most of the model's time and memory.
    Requires the ``nequip`` package (and ``nequip-allegro`` for Allegro models).

    Parameters
    ----------
    compiled_path : str or Path
        A ``.nequip.pt2`` file compiled with outputs ``atomic_energy`` and ``total_energy``.
    device : str
        Device the compiled model was compiled for.

    Returns
    -------
    calculator : ase.calculators.calculator.Calculator
    r_cut : float
        The model's neighbour cut-off (A). It bounds the range of the per-atom energies only for
        strictly local models such as Allegro, not for message-passing ones such as NequIP.
    """
    from nequip.data import AtomicDataDict
    from nequip.data.transforms import NeighborListTransform
    from nequip.integrations.ase import NequIPCalculator

    class _EnergyOnlyCalculator(NequIPCalculator):
        @classmethod
        def _get_aoti_compile_target(cls):
            return {
                "input": [
                    AtomicDataDict.POSITIONS_KEY,
                    AtomicDataDict.EDGE_INDEX_KEY,
                    AtomicDataDict.ATOM_TYPE_KEY,
                    AtomicDataDict.CELL_KEY,
                    AtomicDataDict.EDGE_CELL_SHIFT_KEY,
                ],
                "output": [AtomicDataDict.PER_ATOM_ENERGY_KEY, AtomicDataDict.TOTAL_ENERGY_KEY],
            }

    calc = _EnergyOnlyCalculator.from_compiled_model(
        str(compiled_path), device=device, chemical_species_to_atom_type_map=True
    )
    r_cut = next(t.r_max for t in calc.transforms if isinstance(t, NeighborListTransform))
    return calc, float(r_cut)


def nequip_force_calculator(compiled_path, device: str = "cuda"):
    """ASE calculator for a NequIP or Allegro model compiled for ASE (energy, forces and stress),
    and its cut-off, as :class:`RelaxedSwapEnergy` needs.

    Parameters
    ----------
    compiled_path : str or Path
        A ``.nequip.pt2`` file compiled with ``--target ase``.
    device : str
        Device the compiled model was compiled for.

    Returns
    -------
    calculator : ase.calculators.calculator.Calculator
    r_cut : float
        The model's neighbour cut-off (A), as for :func:`nequip_energy_calculator`.
    """
    from nequip.data.transforms import NeighborListTransform
    from nequip.integrations.ase import NequIPCalculator

    calc = NequIPCalculator.from_compiled_model(
        str(compiled_path), device=device, chemical_species_to_atom_type_map=True
    )
    r_cut = next(t.r_max for t in calc.transforms if isinstance(t, NeighborListTransform))
    return calc, float(r_cut)
