import pytest
import numpy as np
from numpy import allclose
from flare.struc import Structure
from flare.env import AtomicEnvironment
from .fake_gp import generate_mb_envs


np.random.seed(0)

cutoff_mask_list = [# (True, np.array([1]), [10]),
                    (False, np.array([1]), [16]),
                    (False, np.array([1, 0.05]), [16, 0]),
                    (False, np.array([1, 0.8]), [16, 1]),
                    (False, np.array([1, 0.9]), [16, 21]),
                    (True, np.array([1, 0.8]), [16, 9]),
                    (True, np.array([1, 0.05, 0.4]), [16, 0]),
                    (False, np.array([1, 0.05, 0.4]), [16, 0])]


@pytest.fixture(scope='module')
def structure() -> Structure:
    """
    Returns a GP instance with a two-body numba-based kernel
    """

    # list of all bonds and triplets can be found in test_files/test_env_list
    cell = np.eye(3)
    species = [1, 2, 3, 1]
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5],
                          [0.1, 0.1, 0.1], [0.75, 0.75, 0.75]])
    struc_test = Structure(cell, species, positions)

    yield struc_test
    del struc_test


@pytest.mark.parametrize('mask, cutoff, result', cutoff_mask_list)
def test_2bspecies_count(structure, mask, cutoff, result):

    if (mask is True):
        mask = generate_mask(cutoff)
    else:
        mask = None

    env_test = AtomicEnvironment(structure=structure,
                                 atom=0,
                                 cutoffs=cutoff,
                                 cutoffs_mask=mask)
    assert (len(structure.positions) == len(structure.coded_species))
    print(env_test.__dict__)

    assert (len(env_test.bond_array_2) == len(env_test.etypes))
    assert (isinstance(env_test.etypes[0], np.int8))
    assert (len(env_test.bond_array_2) == result[0])

    if (len(cutoff) > 1):
        assert (np.sum(env_test.triplet_counts) == result[1])


@pytest.mark.parametrize('mask, cutoff, result', cutoff_mask_list)
def test_env_methods(structure, mask, cutoff, result):

    if (mask is True):
        mask = generate_mask(cutoff)
    else:
        mask = None

    env_test = AtomicEnvironment(structure,
                                 atom=0,
                                 cutoffs=cutoff,
                                 cutoffs_mask=mask)

    assert str(env_test) == f'Atomic Env. of Type 1 surrounded by {result[0]} atoms' \
                            ' of Types [1, 2, 3]'

    the_dict = env_test.as_dict()
    assert isinstance(the_dict, dict)
    for key in ['positions', 'cell', 'atom', 'cutoffs', 'species']:
        assert key in the_dict.keys()

    remade_env = AtomicEnvironment.from_dict(the_dict)
    assert isinstance(remade_env, AtomicEnvironment)

    assert np.array_equal(remade_env.bond_array_2, env_test.bond_array_2)
    if (len(cutoff) > 1):
        assert np.array_equal(remade_env.bond_array_3, env_test.bond_array_3)
    if (len(cutoff) > 2):
        assert np.array_equal(remade_env.m2b_array, env_test.m2b_array)
    if (len(cutoff) > 3):
        assert np.array_equal(remade_env.m3b_array, env_test.m3b_array)


def test_mb():
    delta = 1e-4
    tol = 1e-4
    cell = 1e7 * np.eye(3)
    cutoffs = np.ones(4)*1.2

    np.random.seed(10)
    atom = 0
    d1 = 1
    env_test = generate_mb_envs(cutoffs, cell, delta, d1=d1, kern_type='mc')
    env_0 = env_test[0][atom]
    env_p = env_test[1][atom]
    env_m = env_test[2][atom]
    ctype = env_0.ctype

    # test m2b
    mb_grads_analytic = env_0.m2b_neigh_grads[:, d1-1]

    s_p = np.where(env_p.m2b_unique_species==ctype)[0][0]
    p_neigh_array = env_p.m2b_neigh_array[:, s_p]

    s_m = np.where(env_m.m2b_unique_species==ctype)[0][0]
    m_neigh_array = env_m.m2b_neigh_array[:, s_m]

    mb_grads_finitediff = (p_neigh_array - m_neigh_array) / (2 * delta)
    assert(allclose(mb_grads_analytic, mb_grads_finitediff))

    # test m3b
    mb_grads_analytic = env_0.m3b_grads[:, :, d1-1]
    mb_neigh_grads_analytic = env_0.m3b_neigh_grads[:, :, d1-1]

    s_p = np.where(env_p.m3b_unique_species==ctype)[0][0]
    p_array = env_p.m3b_array
    p_neigh_array = env_p.m3b_neigh_array[:, s_p, :]

    s_m = np.where(env_m.m3b_unique_species==ctype)[0][0]
    m_array = env_m.m3b_array
    m_neigh_array = env_m.m3b_neigh_array[:, s_m, :]

    mb_grads_finitediff = (p_array - m_array) / (2 * delta)
    assert(allclose(mb_grads_analytic, mb_grads_finitediff))
    print(mb_grads_analytic, mb_grads_finitediff)

    for n in range(p_neigh_array.shape[0]):
        mb_neigh_grads_finitediff = (p_neigh_array[n] - m_neigh_array[n]) / (2 * delta)
#        if env_p.etypes[n] == ctype:
#            mb_neigh_grads_finitediff /= 2
        assert(allclose(mb_neigh_grads_analytic[n], mb_neigh_grads_finitediff))
        print(mb_neigh_grads_analytic[n], mb_neigh_grads_finitediff)

def generate_mask(cutoff):
    ncutoff = len(cutoff)
    if (ncutoff == 1):
        # (1, 1) uses 0.5 cutoff,  (1, 2) (1, 3) (2, 3) use 0.9 cutoff
        mask = {'nspecie': 2, 'specie_mask': np.ones(118, dtype=int)}
        mask['specie_mask'][1] = 0
        mask['cutoff_2b'] = np.array([0.5, 0.9])
        mask['nbond'] = 2
        mask['bond_mask'] = np.ones(4, dtype=int)
        mask['bond_mask'][0] = 0

    elif (ncutoff == 2):
        # the 3b mask is the same structure as 2b
        nspecie = 3
        specie_mask = np.zeros(118, dtype=int)
        chem_spec = [1, 2, 3]
        specie_mask[chem_spec] = np.arange(3)

        # from type 1 to 4 is
        # (1, 1) (1, 2) (1, 3) (2, 3) (*, *)
        # correspond to cutoff 0.5, 0.9, 0.8, 0.9, 0.05
        ncut3b = 5
        tmask = np.ones(nspecie**2, dtype=int)*(ncut3b-1)
        count = 0
        for i, j in [(1, 1), (1, 2), (1, 3), (2, 3)]:
            cs1 = specie_mask[i]
            cs2 = specie_mask[j]
            tmask[cs1*nspecie+cs2] = count
            tmask[cs2*nspecie+cs1] = count
            count += 1

        mask = {'nspecie': nspecie,
                'specie_mask': specie_mask,
                'cutoff_3b': np.array([0.5, 0.9, 0.8, 0.9, 0.05]),
                'ncut3b': ncut3b,
                'cut3b_mask': tmask}

    elif (ncutoff == 3):
        # (1, 1) uses 0.5 cutoff,  (1, 2) (1, 3) (2, 3) use 0.9 cutoff
        mask = {'nspecie': 2, 'specie_mask': np.ones(118, dtype=int)}
        mask['specie_mask'][1] = 0
        mask['cutoff_mb'] = np.array([0.5, 0.9])
        mask['nmb'] = 2
        mask['mb_mask'] = np.ones(4, dtype=int)
        mask['mb_mask'][0] = 0
    return mask

