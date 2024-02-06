"""Testing for src/simple_dmrg/mps_functions.py"""
import unittest  # The test framework

import numpy as np
import numpy.testing as npt

import simple_dmrg.dmrg_functions as dmrg
import simple_dmrg.mpo_construction as make_mpo
import simple_dmrg.mpo_operations as mpo_ops
import simple_dmrg.mps_functions as mps_func

NUMPY_RTOL = 1e-10
NUMPY_ATOL = 1e-13


class TestMPSFunctions(unittest.TestCase):
    """Tests for mps_functions.py"""

    def test_occupation_numbers_to_statevector_1(self):
        # Occupation numbers
        num_sites = 4
        physical_dim = 2
        occupation_numbers = np.array([1, 0, 1, 0])

        # Expected statevector
        expected_statevector = np.zeros(physical_dim**num_sites, dtype=complex)
        expected_statevector[10] = 1

        # Convert the occupation numbers to a statevector
        statevector = mps_func.occupation_numbers_to_statevector(
            occupation_numbers=occupation_numbers,
            num_sites=num_sites,
            physical_dim=physical_dim,
        )

        # Check that the statevector is as expected
        npt.assert_allclose(
            statevector, expected_statevector, rtol=NUMPY_RTOL, atol=NUMPY_ATOL
        )

    def test_occupation_numbers_to_statevector_2(self):
        # Occupation numbers
        num_sites = 7
        physical_dim = 2
        occupation_numbers = np.array([1, 0, 0, 0, 1, 0, 1])

        # Expected statevector
        expected_statevector = np.zeros(physical_dim**num_sites, dtype=complex)
        expected_statevector[69] = 1

        # Convert the occupation numbers to a statevector
        statevector = mps_func.occupation_numbers_to_statevector(
            occupation_numbers=occupation_numbers,
            num_sites=num_sites,
            physical_dim=physical_dim,
        )

        # Check that the statevector is as expected
        npt.assert_allclose(
            statevector, expected_statevector, rtol=NUMPY_RTOL, atol=NUMPY_ATOL
        )

    def test_occupation_numbers_to_statevector_3(self):
        # Occupation numbers
        num_sites = 6
        physical_dim = 4
        occupation_numbers = np.array([3, 2, 0, 0, 3, 2])

        # Expected statevector
        expected_statevector = np.zeros(physical_dim**num_sites, dtype=complex)
        expected_statevector[3598] = 1

        # Convert the occupation numbers to a statevector
        statevector = mps_func.occupation_numbers_to_statevector(
            occupation_numbers=occupation_numbers,
            num_sites=num_sites,
            physical_dim=physical_dim,
        )

        # Check that the statevector is as expected
        npt.assert_allclose(
            statevector, expected_statevector, rtol=NUMPY_RTOL, atol=NUMPY_ATOL
        )

    def test_statevector_to_mps_1(self):
        # Occupation numbers
        num_sites = 4
        physical_dim = 2
        occupation_numbers = np.array([1, 0, 1, 0])
        bond_dim = None

        # Expected statevector
        expected_statevector = np.zeros(physical_dim**num_sites, dtype=complex)
        expected_statevector[10] = 1

        # Convert the occupation numbers to a statevector
        statevector = mps_func.occupation_numbers_to_statevector(
            occupation_numbers=occupation_numbers,
            num_sites=num_sites,
            physical_dim=physical_dim,
        )

        # Convert the statevector to an MPS
        new_mps = mps_func.statevector_to_mps(
            state_vector=statevector,
            num_sites=num_sites,
            physical_dim=physical_dim,
            right_normalize=True,
            orig_bond_dim=bond_dim,
            verbosity=1,
        )

        # Convert the MPS to a statevector
        new_statevector = mps_func.mps_to_statevector(mps=new_mps)
        # mps_to_statevector(mps: List[np.ndarray], verbosity: int = 0)
        # Check that the statevector is as expected
        npt.assert_allclose(
            -1 * new_statevector,
            expected_statevector,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
        )

        # Check that the MPS is normalized
        npt.assert_allclose(
            mps_func.mps_inner_product(mps_ket=new_mps, mps_bra=new_mps),
            1,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
        )

        # Check that particle densities for the MPS are as expected
        densities = mps_func.calculate_on_site_densities(mps=new_mps)
        npt.assert_allclose(
            densities,
            occupation_numbers,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
        )
        # Check that the sum of the particle densities is as expected
        npt.assert_allclose(
            np.sum(densities),
            np.sum(occupation_numbers),
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
        )

    def test_statevector_to_mps_2(self):
        # Occupation numbers
        num_sites = 7
        physical_dim = 2
        occupation_numbers = np.array([1, 0, 0, 0, 1, 0, 1])
        bond_dim = None

        # Expected statevector
        expected_statevector = np.zeros(physical_dim**num_sites, dtype=complex)
        expected_statevector[69] = 1

        # Convert the occupation numbers to a statevector
        statevector = mps_func.occupation_numbers_to_statevector(
            occupation_numbers=occupation_numbers,
            num_sites=num_sites,
            physical_dim=physical_dim,
        )

        # Convert the statevector to an MPS
        new_mps = mps_func.statevector_to_mps(
            state_vector=statevector,
            num_sites=num_sites,
            physical_dim=physical_dim,
            right_normalize=True,
            orig_bond_dim=bond_dim,
            verbosity=1,
        )

        # Convert the MPS to a statevector
        new_statevector = mps_func.mps_to_statevector(mps=new_mps)
        # mps_to_statevector(mps: List[np.ndarray], verbosity: int = 0)
        # Check that the statevector is as expected
        npt.assert_allclose(
            -1 * new_statevector,
            expected_statevector,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
        )

        # Check that the MPS is normalized
        npt.assert_allclose(
            mps_func.mps_inner_product(mps_ket=new_mps, mps_bra=new_mps),
            1,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
        )

        # Check that particle densities for the MPS are as expected
        densities = mps_func.calculate_on_site_densities(mps=new_mps)
        npt.assert_allclose(
            densities,
            occupation_numbers,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
        )
        # Check that the sum of the particle densities is as expected
        npt.assert_allclose(
            np.sum(densities),
            np.sum(occupation_numbers),
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
        )
