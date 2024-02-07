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

    def hami_prep_1(
        self,
        num_sites: int = 4,
        seed: int = 0,
        num_physical_dims: int = 2,
        num_particles: int = 3,
        verbosity: int = 0,
        two_body_tensor_density: float = None,
    ):
        # Parameters
        penalty = 1000

        # RNG
        rng = np.random.default_rng(seed)
        one_body_tensor = rng.standard_normal((num_sites, num_sites))
        # Symmetrize
        one_body_tensor = (one_body_tensor + one_body_tensor.T) / 2

        # Twobody tensor
        two_body_tensor = rng.standard_normal(
            (num_sites, num_sites, num_sites, num_sites)
        )
        # Symmetrize
        two_body_tensor = (
            two_body_tensor + np.transpose(two_body_tensor, (3, 2, 1, 0))
        ) / 2

        # Remove elements from the two-body tensor randomly
        if two_body_tensor_density is not None:
            # Get fraction of "upper half" of the tensor
            num_elements = int(
                (1 - two_body_tensor_density)
                * ((two_body_tensor.size / 2) - two_body_tensor.shape[0])
            )
            # Get tuples of indices of elements to remove
            p_indices = rng.choice(
                np.arange(two_body_tensor.shape[0]), size=num_elements, replace=True
            )
            q_indices = rng.choice(
                np.arange(two_body_tensor.shape[1]), size=num_elements, replace=True
            )
            r_indices = rng.choice(
                np.arange(two_body_tensor.shape[2]), size=num_elements, replace=True
            )
            s_indices = rng.choice(
                np.arange(two_body_tensor.shape[3]), size=num_elements, replace=True
            )
            # Remove elements
            two_body_tensor[p_indices, q_indices, r_indices, s_indices] = 0
            two_body_tensor[s_indices, r_indices, q_indices, p_indices] = 0

        hamiltonian_mpo_with_penalty = (
            make_mpo.make_electronic_hamiltonian_simple_number_enforcement(
                one_body_tensor=one_body_tensor,
                two_body_tensor=two_body_tensor,
                num_sites=num_sites,
                num_physical_dims=num_physical_dims,
                num_electrons=num_particles,
                number_penalty=penalty,
                verbosity=verbosity,
            )
        )

        return hamiltonian_mpo_with_penalty

    def test_hami_prep_1(self):
        # Get Hami
        hamiltonian_mpo_with_penalty = self.hami_prep_1()

        # Convert the Hamiltonian MPO to a dense matrix
        hamiltonian_matrix = mpo_ops.mpo_to_dense_matrix(
            mpo=hamiltonian_mpo_with_penalty
        )

        # Ensure that the Hamiltonian matrix is Hermitian
        npt.assert_allclose(
            hamiltonian_matrix,
            hamiltonian_matrix.conj().T,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
        )

        # Get Hami
        hamiltonian_mpo_with_penalty = self.hami_prep_1(two_body_tensor_density=0.8)

        # Convert the Hamiltonian MPO to a dense matrix
        hamiltonian_matrix = mpo_ops.mpo_to_dense_matrix(
            mpo=hamiltonian_mpo_with_penalty
        )

        # Ensure that the Hamiltonian matrix is Hermitian
        npt.assert_allclose(
            hamiltonian_matrix,
            hamiltonian_matrix.conj().T,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
        )

        # Get Hami
        hamiltonian_mpo_with_penalty = self.hami_prep_1(
            num_sites=6,
            seed=96480,
            num_physical_dims=2,
            num_particles=3,
            two_body_tensor_density=0.2,
        )

        # Convert the Hamiltonian MPO to a dense matrix
        hamiltonian_matrix = mpo_ops.mpo_to_dense_matrix(
            mpo=hamiltonian_mpo_with_penalty
        )

        # Ensure that the Hamiltonian matrix is Hermitian
        npt.assert_allclose(
            hamiltonian_matrix,
            hamiltonian_matrix.conj().T,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
        )

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
            verbosity=0,
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

        #   Check that energies are as expected for statevector and mps
        # Get Hami
        num_particles = np.sum(occupation_numbers)
        hamiltonian_mpo_with_penalty = self.hami_prep_1(
            num_sites=num_sites,
            seed=6387454,
            num_physical_dims=physical_dim,
            num_particles=num_particles,
        )

        # Calculate the energy of the state
        mps_energy = mpo_ops.mpo_general_expectation(
            mps_bra=new_mps,
            mpo=hamiltonian_mpo_with_penalty,
            mps_ket=new_mps,
        )

        # Get dense matrix
        hamiltonian_matrix = mpo_ops.mpo_to_dense_matrix(
            mpo=hamiltonian_mpo_with_penalty
        )

        # Calculate the energy of the state using the dense matrix
        statevector_energy = statevector.conj().T @ hamiltonian_matrix @ statevector

        # Check that the energy of the state is as expected
        npt.assert_allclose(
            mps_energy,
            statevector_energy,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )

        # Get diagonal Hami
        number_penalty = None
        num_electrons = None
        one_body_tensor = np.zeros((num_sites, num_sites))
        one_body_tensor[0, 0] = 1
        one_body_tensor[1, 1] = 2
        one_body_tensor[2, 2] = 3
        one_body_tensor[3, 3] = 7
        expected_energy = (np.diag(one_body_tensor) * occupation_numbers).sum()

        two_body_tensor = None
        diagonal_hami_mpo = (
            make_mpo.make_electronic_hamiltonian_simple_number_enforcement(
                one_body_tensor=one_body_tensor,
                two_body_tensor=two_body_tensor,
                num_sites=num_sites,
                num_physical_dims=physical_dim,
                num_electrons=num_electrons,
                number_penalty=number_penalty,
            )
        )

        # Calculate the energy of the state
        mps_energy = mpo_ops.mpo_general_expectation(
            mps_bra=new_mps,
            mpo=diagonal_hami_mpo,
            mps_ket=new_mps,
        )

        # Check that the energy of the state is as expected
        npt.assert_allclose(
            mps_energy,
            expected_energy,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
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
            verbosity=0,
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

        #   Check that energies are as expected for statevector and mps
        # Get Hami
        num_particles = np.sum(occupation_numbers)
        hamiltonian_mpo_with_penalty = self.hami_prep_1(
            num_sites=num_sites,
            seed=984515,
            num_physical_dims=physical_dim,
            num_particles=num_particles,
            verbosity=1,
            two_body_tensor_density=0.1,
        )

        # Calculate the energy of the state
        mps_energy = mpo_ops.mpo_general_expectation(
            mps_bra=new_mps,
            mpo=hamiltonian_mpo_with_penalty,
            mps_ket=new_mps,
        )

        # Get dense matrix
        hamiltonian_matrix = mpo_ops.mpo_to_dense_matrix(
            mpo=hamiltonian_mpo_with_penalty
        )

        # Calculate the energy of the state using the dense matrix
        statevector_energy = statevector.conj().T @ hamiltonian_matrix @ statevector

        # Check that the energy of the state is as expected
        npt.assert_allclose(
            mps_energy,
            statevector_energy,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )

        # Get diagonal Hami
        number_penalty = None
        num_electrons = None
        one_body_tensor = np.zeros((num_sites, num_sites))
        one_body_tensor[0, 0] = 1
        one_body_tensor[1, 1] = 2
        one_body_tensor[2, 2] = 3
        one_body_tensor[3, 3] = 7
        one_body_tensor[4, 4] = 11
        one_body_tensor[5, 5] = 6
        one_body_tensor[6, 6] = -2

        expected_energy = (np.diag(one_body_tensor) * occupation_numbers).sum()

        two_body_tensor = None
        diagonal_hami_mpo = (
            make_mpo.make_electronic_hamiltonian_simple_number_enforcement(
                one_body_tensor=one_body_tensor,
                two_body_tensor=two_body_tensor,
                num_sites=num_sites,
                num_physical_dims=physical_dim,
                num_electrons=num_electrons,
                number_penalty=number_penalty,
                verbosity=1,
            )
        )

        # Calculate the energy of the state
        mps_energy = mpo_ops.mpo_general_expectation(
            mps_bra=new_mps,
            mpo=diagonal_hami_mpo,
            mps_ket=new_mps,
        )

        # Check that the energy of the state is as expected
        npt.assert_allclose(
            mps_energy,
            expected_energy,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )

    def test_statevector_to_mps_truncate_bond_dim_1(self):
        # Occupation numbers
        num_sites = 7
        physical_dim = 2
        occupation_numbers = np.array([1, 0, 0, 0, 1, 0, 1])
        bond_dim = 2

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
            verbosity=0,
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

        #   Check that energies are as expected for statevector and mps
        # Get Hami
        num_particles = np.sum(occupation_numbers)
        hamiltonian_mpo_with_penalty = self.hami_prep_1(
            num_sites=num_sites,
            seed=984515,
            num_physical_dims=physical_dim,
            num_particles=num_particles,
            verbosity=0,
            two_body_tensor_density=0.1,
        )

        # Calculate the energy of the state
        mps_energy = mpo_ops.mpo_general_expectation(
            mps_bra=new_mps,
            mpo=hamiltonian_mpo_with_penalty,
            mps_ket=new_mps,
        )

        # Get dense matrix
        hamiltonian_matrix = mpo_ops.mpo_to_dense_matrix(
            mpo=hamiltonian_mpo_with_penalty
        )

        # Calculate the energy of the state using the dense matrix
        statevector_energy = statevector.conj().T @ hamiltonian_matrix @ statevector

        # Check that the energy of the state is as expected
        npt.assert_allclose(
            mps_energy,
            statevector_energy,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )

        # Get diagonal Hami
        number_penalty = None
        num_electrons = None
        one_body_tensor = np.zeros((num_sites, num_sites))
        one_body_tensor[0, 0] = 1
        one_body_tensor[1, 1] = 2
        one_body_tensor[2, 2] = 3
        one_body_tensor[3, 3] = 7
        one_body_tensor[4, 4] = 11
        one_body_tensor[5, 5] = 6
        one_body_tensor[6, 6] = -2

        expected_energy = (np.diag(one_body_tensor) * occupation_numbers).sum()

        two_body_tensor = None
        diagonal_hami_mpo = (
            make_mpo.make_electronic_hamiltonian_simple_number_enforcement(
                one_body_tensor=one_body_tensor,
                two_body_tensor=two_body_tensor,
                num_sites=num_sites,
                num_physical_dims=physical_dim,
                num_electrons=num_electrons,
                number_penalty=number_penalty,
                verbosity=0,
            )
        )

        # Calculate the energy of the state
        mps_energy = mpo_ops.mpo_general_expectation(
            mps_bra=new_mps,
            mpo=diagonal_hami_mpo,
            mps_ket=new_mps,
        )

        # Check that the energy of the state is as expected
        npt.assert_allclose(
            mps_energy,
            expected_energy,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )

    def test_statevector_to_mps_truncate_bond_dim_2(self):
        # Occupation numbers
        num_sites = 4
        physical_dim = 2
        occupation_numbers = np.array([1, 0, 1, 0])
        bond_dim = 1

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
            verbosity=0,
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

        #   Check that energies are as expected for statevector and mps
        # Get Hami
        num_particles = np.sum(occupation_numbers)
        hamiltonian_mpo_with_penalty = self.hami_prep_1(
            num_sites=num_sites,
            seed=6387454,
            num_physical_dims=physical_dim,
            num_particles=num_particles,
        )

        # Calculate the energy of the state
        mps_energy = mpo_ops.mpo_general_expectation(
            mps_bra=new_mps,
            mpo=hamiltonian_mpo_with_penalty,
            mps_ket=new_mps,
        )

        # Get dense matrix
        hamiltonian_matrix = mpo_ops.mpo_to_dense_matrix(
            mpo=hamiltonian_mpo_with_penalty
        )

        # Calculate the energy of the state using the dense matrix
        statevector_energy = statevector.conj().T @ hamiltonian_matrix @ statevector

        # Check that the energy of the state is as expected
        npt.assert_allclose(
            mps_energy,
            statevector_energy,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )

        # Get diagonal Hami
        number_penalty = None
        num_electrons = None
        one_body_tensor = np.zeros((num_sites, num_sites))
        one_body_tensor[0, 0] = 1
        one_body_tensor[1, 1] = 2
        one_body_tensor[2, 2] = 3
        one_body_tensor[3, 3] = 7
        expected_energy = (np.diag(one_body_tensor) * occupation_numbers).sum()

        two_body_tensor = None
        diagonal_hami_mpo = (
            make_mpo.make_electronic_hamiltonian_simple_number_enforcement(
                one_body_tensor=one_body_tensor,
                two_body_tensor=two_body_tensor,
                num_sites=num_sites,
                num_physical_dims=physical_dim,
                num_electrons=num_electrons,
                number_penalty=number_penalty,
            )
        )

        # Calculate the energy of the state
        mps_energy = mpo_ops.mpo_general_expectation(
            mps_bra=new_mps,
            mpo=diagonal_hami_mpo,
            mps_ket=new_mps,
        )

        # Check that the energy of the state is as expected
        npt.assert_allclose(
            mps_energy,
            expected_energy,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )

    def test_statevector_to_mps_pad_bond_dim_1(self):
        # Occupation numbers
        num_sites = 7
        physical_dim = 2
        occupation_numbers = np.array([1, 0, 0, 0, 1, 0, 1])
        bond_dim = 20

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
            verbosity=0,
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

        #   Check that energies are as expected for statevector and mps
        # Get Hami
        num_particles = np.sum(occupation_numbers)
        hamiltonian_mpo_with_penalty = self.hami_prep_1(
            num_sites=num_sites,
            seed=984515,
            num_physical_dims=physical_dim,
            num_particles=num_particles,
            verbosity=0,
            two_body_tensor_density=0.1,
        )

        # Calculate the energy of the state
        mps_energy = mpo_ops.mpo_general_expectation(
            mps_bra=new_mps,
            mpo=hamiltonian_mpo_with_penalty,
            mps_ket=new_mps,
        )

        # Get dense matrix
        hamiltonian_matrix = mpo_ops.mpo_to_dense_matrix(
            mpo=hamiltonian_mpo_with_penalty
        )

        # Calculate the energy of the state using the dense matrix
        statevector_energy = statevector.conj().T @ hamiltonian_matrix @ statevector

        # Check that the energy of the state is as expected
        npt.assert_allclose(
            mps_energy,
            statevector_energy,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )

        # Get diagonal Hami
        number_penalty = None
        num_electrons = None
        one_body_tensor = np.zeros((num_sites, num_sites))
        one_body_tensor[0, 0] = 1
        one_body_tensor[1, 1] = 2
        one_body_tensor[2, 2] = 3
        one_body_tensor[3, 3] = 7
        one_body_tensor[4, 4] = 11
        one_body_tensor[5, 5] = 6
        one_body_tensor[6, 6] = -2

        expected_energy = (np.diag(one_body_tensor) * occupation_numbers).sum()

        two_body_tensor = None
        diagonal_hami_mpo = (
            make_mpo.make_electronic_hamiltonian_simple_number_enforcement(
                one_body_tensor=one_body_tensor,
                two_body_tensor=two_body_tensor,
                num_sites=num_sites,
                num_physical_dims=physical_dim,
                num_electrons=num_electrons,
                number_penalty=number_penalty,
                verbosity=0,
            )
        )

        # Calculate the energy of the state
        mps_energy = mpo_ops.mpo_general_expectation(
            mps_bra=new_mps,
            mpo=diagonal_hami_mpo,
            mps_ket=new_mps,
        )

        # Check that the energy of the state is as expected
        npt.assert_allclose(
            mps_energy,
            expected_energy,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )

    def test_statevector_to_mps_pad_bond_dim_2(self):
        # Occupation numbers
        num_sites = 4
        physical_dim = 2
        occupation_numbers = np.array([1, 0, 1, 0])
        bond_dim = 23

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
            verbosity=0,
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

        #   Check that energies are as expected for statevector and mps
        # Get Hami
        num_particles = np.sum(occupation_numbers)
        hamiltonian_mpo_with_penalty = self.hami_prep_1(
            num_sites=num_sites,
            seed=6387454,
            num_physical_dims=physical_dim,
            num_particles=num_particles,
        )

        # Calculate the energy of the state
        mps_energy = mpo_ops.mpo_general_expectation(
            mps_bra=new_mps,
            mpo=hamiltonian_mpo_with_penalty,
            mps_ket=new_mps,
        )

        # Get dense matrix
        hamiltonian_matrix = mpo_ops.mpo_to_dense_matrix(
            mpo=hamiltonian_mpo_with_penalty
        )

        # Calculate the energy of the state using the dense matrix
        statevector_energy = statevector.conj().T @ hamiltonian_matrix @ statevector

        # Check that the energy of the state is as expected
        npt.assert_allclose(
            mps_energy,
            statevector_energy,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )

        # Get diagonal Hami
        number_penalty = None
        num_electrons = None
        one_body_tensor = np.zeros((num_sites, num_sites))
        one_body_tensor[0, 0] = 1
        one_body_tensor[1, 1] = 2
        one_body_tensor[2, 2] = 3
        one_body_tensor[3, 3] = 7
        expected_energy = (np.diag(one_body_tensor) * occupation_numbers).sum()

        two_body_tensor = None
        diagonal_hami_mpo = (
            make_mpo.make_electronic_hamiltonian_simple_number_enforcement(
                one_body_tensor=one_body_tensor,
                two_body_tensor=two_body_tensor,
                num_sites=num_sites,
                num_physical_dims=physical_dim,
                num_electrons=num_electrons,
                number_penalty=number_penalty,
            )
        )

        # Calculate the energy of the state
        mps_energy = mpo_ops.mpo_general_expectation(
            mps_bra=new_mps,
            mpo=diagonal_hami_mpo,
            mps_ket=new_mps,
        )

        # Check that the energy of the state is as expected
        npt.assert_allclose(
            mps_energy,
            expected_energy,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )
