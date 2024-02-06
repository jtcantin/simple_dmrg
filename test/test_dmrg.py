"""Testing for src/simple_dmrg/dmrg_functions.py"""
import unittest  # The test framework

import numpy as np
import numpy.testing as npt

import simple_dmrg.dmrg_functions as dmrg
import simple_dmrg.mpo_construction as make_mpo
import simple_dmrg.mpo_operations as mpo_ops
import simple_dmrg.mps_functions as mps_func

NUMPY_RTOL = 1e-10
NUMPY_ATOL = 1e-13


class TestDMRG(unittest.TestCase):
    """Tests for dmrg.py"""

    def test_integrated_diagonal_one_body_manual(self):
        """Ensure that DMRG gets the correct answer for a simple one-body
        Hamiltonian."""

        # Parameters
        num_sites = 4
        penalty = 1000  # μ
        num_particles = 3  # N_e
        num_physical_dims = 2  # Number of physical dimensions of
        # the Hilbert space at each site;
        # it is 2 for spinless fermions
        # (the site is either occupied or unoccupied)

        bond_dimension = 10

        # Define the one-body tensor
        one_body_tensor = np.zeros(shape=(num_sites, num_sites))
        one_body_tensor[0, 0] = 1
        one_body_tensor[1, 1] = -3
        one_body_tensor[2, 2] = -1
        one_body_tensor[3, 3] = 1

        # # Define the two-body tensor
        # # Just use a simple diagonal tensor for illustrative purposes
        # two_body_tensor = np.zeros(shape=(num_sites, num_sites, num_sites, num_sites))
        # two_body_tensor[0, 0, 0, 0] = 1
        # two_body_tensor[1, 1, 1, 1] = -3
        # two_body_tensor[2, 2, 2, 2] = -1
        # two_body_tensor[3, 3, 3, 3] = 1

        hamiltonian_mpo_with_penalty = (
            make_mpo.make_electronic_hamiltonian_simple_number_enforcement(
                one_body_tensor=one_body_tensor,
                two_body_tensor=None,
                num_sites=num_sites,
                num_physical_dims=num_physical_dims,
                num_electrons=num_particles,
                number_penalty=penalty,
            )
        )

        # Convert the Hamiltonian MPO to a dense matrix
        hamiltonian_matrix = mpo_ops.mpo_to_dense_matrix(
            mpo=hamiltonian_mpo_with_penalty
        )

        # Ensure that the Hamiltonian matrix is Hermitian
        self.assertTrue(np.allclose(hamiltonian_matrix, hamiltonian_matrix.conj().T))

        output_dict = dmrg.drmg_main(
            mpo=hamiltonian_mpo_with_penalty,
            num_sites=num_sites,
            physical_dimension=num_physical_dims,  # 2 for one possible Fermion per site (aka spin-orbital)
            bond_dimension=bond_dimension,
            seed=0,  # Random seed for the initial state
            num_sweeps=3,  # Number of DMRG sweeps. Each sweep includes a left-to-right and right-to-left sweep
            verbosity=0,  # 0: No output, 1: Basic output, 2: Detailed output for debugging
        )

        optimized_mps = output_dict["optimized_mps"]

        # Calculate the energy of the optimized state
        # For 3 particles, the ground state energy is -3.0
        gs_energy = mpo_ops.mpo_general_expectation(
            mps_bra=optimized_mps,
            mpo=hamiltonian_mpo_with_penalty,
            mps_ket=optimized_mps,
        )
        npt.assert_allclose(
            gs_energy, -3.0, rtol=NUMPY_RTOL, atol=NUMPY_ATOL, equal_nan=False
        )

        # Calculate the particle number of the optimized state
        # It should be 3
        # particle_number_op = ∑_i n_i
        particle_number_mpo = make_mpo.make_particle_number_mpo(num_sites=num_sites)

        gs_num_particles = mpo_ops.mpo_general_expectation(
            mps_bra=optimized_mps, mpo=particle_number_mpo, mps_ket=optimized_mps
        )

        npt.assert_allclose(
            gs_num_particles,
            num_particles,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )

        # Calculate the ons-site densities of the optimized state
        # They should be x,1,1,y where x,y>=0 and x+y=1
        densities = mps_func.calculate_on_site_densities(mps=optimized_mps)
        npt.assert_allclose(
            densities[0] + densities[3],
            1,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )
        npt.assert_allclose(
            np.sum(densities),
            num_particles,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )
        npt.assert_allclose(
            densities[1],
            1,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )
        npt.assert_allclose(
            densities[2],
            1,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )

    def test_integrated_diagonal_two_body_manual(self):
        """Ensure that DMRG gets the correct answer for a simple two-body
        Hamiltonian."""
        # Parameters
        num_sites = 4
        penalty = 1000  # μ
        num_particles = 3  # N_e
        num_physical_dims = 2  # Number of physical dimensions of
        # the Hilbert space at each site;
        # it is 2 for spinless fermions
        # (the site is either occupied or unoccupied)

        bond_dimension = 10

        # # Define the one-body tensor
        # one_body_tensor = np.zeros(shape=(num_sites, num_sites))
        # one_body_tensor[0, 0] = 1
        # one_body_tensor[1, 1] = -3
        # one_body_tensor[2, 2] = -1
        # one_body_tensor[3, 3] = 1

        # Define the two-body tensor
        two_body_tensor = np.zeros(shape=(num_sites, num_sites, num_sites, num_sites))
        two_body_tensor[0, 0, 0, 0] = 1
        two_body_tensor[1, 1, 1, 1] = -3
        two_body_tensor[2, 2, 2, 2] = -1
        two_body_tensor[3, 3, 3, 3] = 1

        hamiltonian_mpo_with_penalty = (
            make_mpo.make_electronic_hamiltonian_simple_number_enforcement(
                one_body_tensor=None,
                two_body_tensor=two_body_tensor,
                num_sites=num_sites,
                num_physical_dims=num_physical_dims,
                num_electrons=num_particles,
                number_penalty=penalty,
            )
        )

        # Convert the Hamiltonian MPO to a dense matrix
        hamiltonian_matrix = mpo_ops.mpo_to_dense_matrix(
            mpo=hamiltonian_mpo_with_penalty
        )

        # Ensure that the Hamiltonian matrix is Hermitian
        self.assertTrue(np.allclose(hamiltonian_matrix, hamiltonian_matrix.conj().T))

        output_dict = dmrg.drmg_main(
            mpo=hamiltonian_mpo_with_penalty,
            num_sites=num_sites,
            physical_dimension=num_physical_dims,  # 2 for one possible Fermion per site (aka spin-orbital)
            bond_dimension=bond_dimension,
            seed=0,  # Random seed for the initial state
            num_sweeps=3,  # Number of DMRG sweeps. Each sweep includes a left-to-right and right-to-left sweep
            verbosity=0,  # 0: No output, 1: Basic output, 2: Detailed output for debugging
        )

        optimized_mps = output_dict["optimized_mps"]

        # Calculate the energy of the optimized state
        # For 3 particles, the ground state energy is -3.0
        gs_energy = mpo_ops.mpo_general_expectation(
            mps_bra=optimized_mps,
            mpo=hamiltonian_mpo_with_penalty,
            mps_ket=optimized_mps,
        )
        npt.assert_allclose(
            gs_energy, -3.0, rtol=NUMPY_RTOL, atol=NUMPY_ATOL, equal_nan=False
        )

        # Calculate the particle number of the optimized state
        # It should be 3
        # particle_number_op = ∑_i n_i
        particle_number_mpo = make_mpo.make_particle_number_mpo(num_sites=num_sites)

        gs_num_particles = mpo_ops.mpo_general_expectation(
            mps_bra=optimized_mps, mpo=particle_number_mpo, mps_ket=optimized_mps
        )

        npt.assert_allclose(
            gs_num_particles,
            num_particles,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )

        # Calculate the ons-site densities of the optimized state
        # They should be x,1,1,y where x,y>=0 and x+y=1
        densities = mps_func.calculate_on_site_densities(mps=optimized_mps)
        npt.assert_allclose(
            densities[0] + densities[3],
            1,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )
        npt.assert_allclose(
            np.sum(densities),
            num_particles,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )
        npt.assert_allclose(
            densities[1],
            1,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )
        npt.assert_allclose(
            densities[2],
            1,
            rtol=NUMPY_RTOL,
            atol=NUMPY_ATOL,
            equal_nan=False,
        )
