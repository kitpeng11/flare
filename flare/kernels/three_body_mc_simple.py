import numpy as np
from flare.kernels.kernels import force_helper, force_energy_helper, \
    grad_helper, three_body_fe_perm
from numba import njit
from flare.env import AtomicEnvironment
from typing import Callable
import flare.cutoffs as cf
from math import exp


class ThreeBodyKernel:
    def __init__(self, hyperparameters: 'ndarray', cutoff: float,
                 cutoff_func: Callable = cf.quadratic_cutoff):
        self.hyperparameters = hyperparameters
        self.signal_variance = hyperparameters[0]
        self.length_scale = hyperparameters[1]
        self.cutoff = cutoff
        self.cutoff_func = cutoff_func

    def energy_energy(self, env1: AtomicEnvironment, env2: AtomicEnvironment):

        return energy_energy(env1.bond_array_2, env1.ctype, env1.etypes,
                             env2.bond_array_2, env2.ctype, env2.etypes,
                             env1.cross_bond_inds, env2.cross_bond_inds,
                             env1.cross_bond_dists, env2.cross_bond_dists,
                             env1.triplet_counts, env2.triplet_counts,
                             self.signal_variance, self.length_scale,
                             self.cutoff, self.cutoff_func)

    def force_energy(self, env1: AtomicEnvironment, env2: AtomicEnvironment):

        return force_energy(env1.bond_array_3, env1.ctype, env1.etypes,
                            env2.bond_array_3, env2.ctype, env2.etypes,
                            env1.cross_bond_inds, env2.cross_bond_inds,
                            env1.cross_bond_dists, env2.cross_bond_dists,
                            env1.triplet_counts, env2.triplet_counts,
                            self.signal_variance, self.length_scale,
                            self.cutoff, self.cutoff_func)

    def stress_energy(self, env1: AtomicEnvironment, env2: AtomicEnvironment):

        return stress_energy(env1.bond_array_2, env1.ctype, env1.etypes,
                             env2.bond_array_2, env2.ctype, env2.etypes,
                             self.signal_variance, self.length_scale,
                             self.cutoff, self.cutoff_func)

    def force_force(self, env1: AtomicEnvironment, env2: AtomicEnvironment):

        return force_force(env1.bond_array_2, env1.ctype, env1.etypes,
                           env2.bond_array_2, env2.ctype, env2.etypes,
                           self.signal_variance, self.length_scale,
                           self.cutoff, self.cutoff_func)

    def stress_force(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        return stress_force(env1.bond_array_2, env1.ctype, env1.etypes,
                            env2.bond_array_2, env2.ctype, env2.etypes,
                            self.signal_variance, self.length_scale,
                            self.cutoff, self.cutoff_func)

    def stress_stress(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        return stress_stress(env1.bond_array_2, env1.ctype, env1.etypes,
                             env2.bond_array_2, env2.ctype, env2.etypes,
                             self.signal_variance, self.length_scale,
                             self.cutoff, self.cutoff_func)

    def force_force_gradient(self, env1: AtomicEnvironment,
                             env2: AtomicEnvironment):
        return force_force_gradient(env1.bond_array_2, env1.ctype, env1.etypes,
                                    env2.bond_array_2, env2.ctype, env2.etypes,
                                    self.signal_variance, self.length_scale,
                                    self.cutoff, self.cutoff_func)


@njit
def energy_energy(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
                  cross_bond_inds_1, cross_bond_inds_2,
                  cross_bond_dists_1, cross_bond_dists_2,
                  triplets_1, triplets_2, sig, ls, r_cut, cutoff_func):
    """3-body multi-element kernel between two local energies accelerated
    with Numba.

    Args:
        bond_array_1 (np.ndarray): 3-body bond array of the first local
            environment.
        c1 (int): Species of the central atom of the first local environment.
        etypes1 (np.ndarray): Species of atoms in the first local
            environment.
        bond_array_2 (np.ndarray): 3-body bond array of the second local
            environment.
        c2 (int): Species of the central atom of the second local environment.
        etypes2 (np.ndarray): Species of atoms in the second local
            environment.
        cross_bond_inds_1 (np.ndarray): Two dimensional array whose row m
            contains the indices of atoms n > m in the first local
            environment that are within a distance r_cut of both atom n and
            the central atom.
        cross_bond_inds_2 (np.ndarray): Two dimensional array whose row m
            contains the indices of atoms n > m in the second local
            environment that are within a distance r_cut of both atom n and
            the central atom.
        cross_bond_dists_1 (np.ndarray): Two dimensional array whose row m
            contains the distances from atom m of atoms n > m in the first
            local environment that are within a distance r_cut of both atom
            n and the central atom.
        cross_bond_dists_2 (np.ndarray): Two dimensional array whose row m
            contains the distances from atom m of atoms n > m in the second
            local environment that are within a distance r_cut of both atom
            n and the central atom.
        triplets_1 (np.ndarray): One dimensional array of integers whose entry
            m is the number of atoms in the first local environment that are
            within a distance r_cut of atom m.
        triplets_2 (np.ndarray): One dimensional array of integers whose entry
            m is the number of atoms in the second local environment that are
            within a distance r_cut of atom m.
        sig (float): 3-body signal variance hyperparameter.
        ls (float): 3-body length scale hyperparameter.
        r_cut (float): 3-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Returns:
        float:
            Value of the 3-body local energy kernel.
    """

    kern = 0

    sig2 = sig * sig
    ls2 = 1 / (2 * ls * ls)

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        fi1, _ = cutoff_func(r_cut, ri1, 0)
        ei1 = etypes1[m]

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            fi2, _ = cutoff_func(r_cut, ri2, 0)
            ei2 = etypes1[ind1]

            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)
            fi = fi1 * fi2 * fi3

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                fj1, _ = cutoff_func(r_cut, rj1, 0)
                ej1 = etypes2[p]

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + q + 1]
                    rj2 = bond_array_2[ind2, 0]
                    fj2, _ = cutoff_func(r_cut, rj2, 0)
                    ej2 = etypes2[ind2]

                    rj3 = cross_bond_dists_2[p, p + q + 1]
                    fj3, _ = cutoff_func(r_cut, rj3, 0)
                    fj = fj1 * fj2 * fj3

                    r11 = ri1 - rj1
                    r12 = ri1 - rj2
                    r13 = ri1 - rj3
                    r21 = ri2 - rj1
                    r22 = ri2 - rj2
                    r23 = ri2 - rj3
                    r31 = ri3 - rj1
                    r32 = ri3 - rj2
                    r33 = ri3 - rj3

                    if (c1 == c2):
                        if (ei1 == ej1) and (ei2 == ej2):
                            C1 = r11 * r11 + r22 * r22 + r33 * r33
                            kern += sig2 * exp(-C1 * ls2) * fi * fj
                        if (ei1 == ej2) and (ei2 == ej1):
                            C3 = r12 * r12 + r21 * r21 + r33 * r33
                            kern += sig2 * exp(-C3 * ls2) * fi * fj
                    if (c1 == ej1):
                        if (ei1 == ej2) and (ei2 == c2):
                            C5 = r13 * r13 + r21 * r21 + r32 * r32
                            kern += sig2 * exp(-C5 * ls2) * fi * fj
                        if (ei1 == c2) and (ei2 == ej2):
                            C2 = r11 * r11 + r23 * r23 + r32 * r32
                            kern += sig2 * exp(-C2 * ls2) * fi * fj
                    if (c1 == ej2):
                        if (ei1 == ej1) and (ei2 == c2):
                            C6 = r13 * r13 + r22 * r22 + r31 * r31
                            kern += sig2 * exp(-C6 * ls2) * fi * fj
                        if (ei1 == c2) and (ei2 == ej1):
                            C4 = r12 * r12 + r23 * r23 + r31 * r31
                            kern += sig2 * exp(-C4 * ls2) * fi * fj

    return kern / 9


@njit
def force_energy(bond_array_1, c1, etypes1, bond_array_2, c2, etypes2,
                 cross_bond_inds_1, cross_bond_inds_2,
                 cross_bond_dists_1, cross_bond_dists_2,
                 triplets_1, triplets_2, sig, ls, r_cut, cutoff_func):
    """3-body multi-element kernel between a force component and a local
    energy accelerated with Numba.

    Args:
        bond_array_1 (np.ndarray): 3-body bond array of the first local
            environment.
        c1 (int): Species of the central atom of the first local environment.
        etypes1 (np.ndarray): Species of atoms in the first local
            environment.
        bond_array_2 (np.ndarray): 3-body bond array of the second local
            environment.
        c2 (int): Species of the central atom of the second local environment.
        etypes2 (np.ndarray): Species of atoms in the second local
            environment.
        cross_bond_inds_1 (np.ndarray): Two dimensional array whose row m
            contains the indices of atoms n > m in the first local
            environment that are within a distance r_cut of both atom n and
            the central atom.
        cross_bond_inds_2 (np.ndarray): Two dimensional array whose row m
            contains the indices of atoms n > m in the second local
            environment that are within a distance r_cut of both atom n and
            the central atom.
        cross_bond_dists_1 (np.ndarray): Two dimensional array whose row m
            contains the distances from atom m of atoms n > m in the first
            local environment that are within a distance r_cut of both atom
            n and the central atom.
        cross_bond_dists_2 (np.ndarray): Two dimensional array whose row m
            contains the distances from atom m of atoms n > m in the second
            local environment that are within a distance r_cut of both atom
            n and the central atom.
        triplets_1 (np.ndarray): One dimensional array of integers whose entry
            m is the number of atoms in the first local environment that are
            within a distance r_cut of atom m.
        triplets_2 (np.ndarray): One dimensional array of integers whose entry
            m is the number of atoms in the second local environment that are
            within a distance r_cut of atom m.
        sig (float): 3-body signal variance hyperparameter.
        ls (float): 3-body length scale hyperparameter.
        r_cut (float): 3-body cutoff radius.
        cutoff_func (Callable): Cutoff function.

    Returns:
        float:
            Value of the 3-body force/energy kernel.
    """
    kern = np.zeros(3)

    sig2 = sig * sig
    ls1 = 1 / (2 * ls * ls)
    ls2 = 1 / (ls * ls)

    for m in range(bond_array_1.shape[0]):
        ri1 = bond_array_1[m, 0]
        ei1 = etypes1[m]

        for n in range(triplets_1[m]):
            ind1 = cross_bond_inds_1[m, m + n + 1]
            ri2 = bond_array_1[ind1, 0]
            ei2 = etypes1[ind1]

            ri3 = cross_bond_dists_1[m, m + n + 1]
            fi3, _ = cutoff_func(r_cut, ri3, 0)

            for p in range(bond_array_2.shape[0]):
                rj1 = bond_array_2[p, 0]
                fj1, _ = cutoff_func(r_cut, rj1, 0)
                ej1 = etypes2[p]

                for q in range(triplets_2[p]):
                    ind2 = cross_bond_inds_2[p, p + q + 1]
                    rj2 = bond_array_2[ind2, 0]
                    fj2, _ = cutoff_func(r_cut, rj2, 0)
                    ej2 = etypes2[ind2]
                    rj3 = cross_bond_dists_2[p, p + q + 1]
                    fj3, _ = cutoff_func(r_cut, rj3, 0)
                    fj = fj1 * fj2 * fj3

                    r11 = ri1 - rj1
                    r12 = ri1 - rj2
                    r13 = ri1 - rj3
                    r21 = ri2 - rj1
                    r22 = ri2 - rj2
                    r23 = ri2 - rj3
                    r31 = ri3 - rj1
                    r32 = ri3 - rj2
                    r33 = ri3 - rj3

                    for d1 in range(3):
                        ci1 = bond_array_1[m, d1 + 1]
                        fi1, fdi1 = cutoff_func(r_cut, ri1, ci1)
                        ci2 = bond_array_1[ind1, d1 + 1]
                        fi2, fdi2 = cutoff_func(r_cut, ri2, ci2)
                        fi = fi1 * fi2 * fi3
                        fdi = fdi1 * fi2 * fi3 + fi1 * fdi2 * fi3

                        kern[d1] += \
                            three_body_fe_perm(r11, r12, r13, r21, r22, r23,
                                               r31, r32, r33, c1, c2, ci1, ci2,
                                               ei1, ei2, ej1, ej2, fi, fj, fdi,
                                               ls1, ls2, sig2)

    return kern / 3
