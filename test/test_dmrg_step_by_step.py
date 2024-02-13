"""Tests that go through the algorithm step-by-step."""
import unittest  # The test framework

import numpy as np
import numpy.testing as npt

import simple_dmrg.dmrg_functions as dmrg
import simple_dmrg.mpo_construction as make_mpo
import simple_dmrg.mpo_operations as mpo_ops
import simple_dmrg.mps_functions as mps_func

NUMPY_RTOL = 1e-10
NUMPY_ATOL = 1e-13


class TestDMRGSteps(unittest.TestCase):
    """Tests that go through the algorithm step-by-step."""

    def make_initial_state_mps_1(self):
        """Make an initial state with 4 sites and 3 electrons."""
        num_sites = 3

        num_physical_dims = 2
        bond_dim = 3
        occupation_numbers = np.array([1, 1, 0])

        # Convert the occupation numbers to a statevector
        statevector = mps_func.occupation_numbers_to_statevector(
            occupation_numbers=occupation_numbers,
            num_sites=num_sites,
            physical_dim=num_physical_dims,
        )

        # Convert the statevector to an MPS
        initial_state_mps = mps_func.statevector_to_mps(
            state_vector=statevector,
            num_sites=num_sites,
            physical_dim=num_physical_dims,
            right_normalize=True,
            orig_bond_dim=bond_dim,
            verbosity=0,
        )

        return (
            initial_state_mps,
            num_sites,
            num_physical_dims,
            bond_dim,
            occupation_numbers,
        )

    def simple_diagonal_hamiltonian_mpo_and_initial_state_1(self):
        (
            initial_state_mps,
            num_sites,
            num_physical_dims,
            bond_dim,
            occupation_numbers,
        ) = self.make_initial_state_mps_1()

        num_particles = np.sum(occupation_numbers)
        energies = np.array([1, 2, 3])

        one_body_tensor = np.zeros(shape=(num_sites, num_sites))
        one_body_tensor[0, 0] = energies[0]
        one_body_tensor[1, 1] = energies[1]
        one_body_tensor[2, 2] = energies[2]

        simple_diagonal_hamiltonian_mpo = make_mpo.make_one_body_mpo(
            one_body_tensor=one_body_tensor, num_sites=num_sites
        )

        return (
            initial_state_mps,
            simple_diagonal_hamiltonian_mpo,
            num_sites,
            num_physical_dims,
            bond_dim,
            occupation_numbers,
            num_particles,
        )

    def test_make_proto_L_R_tensors(self):
        """Test the function that makes the first L and R tensors."""

        L_tensor_list, R_tensor_list = dmrg.make_proto_L_R_tensors()

        # Check that the lists have the right length
        self.assertEqual(len(L_tensor_list), 1)
        self.assertEqual(len(R_tensor_list), 1)

        # Check that the tensors have the right shape
        self.assertEqual(L_tensor_list[0].shape, (1, 1, 1))
        self.assertEqual(R_tensor_list[0].shape, (1, 1, 1))

        # Check that the tensors have the right values
        npt.assert_allclose(L_tensor_list[0], 1)
        npt.assert_allclose(R_tensor_list[0], 1)

    def test_make_remaining_initial_R_tensors(self):
        (
            initial_state_mps,
            simple_diagonal_hamiltonian_mpo,
            num_sites,
            num_physical_dims,
            bond_dim,
            occupation_numbers,
            num_particles,
        ) = self.simple_diagonal_hamiltonian_mpo_and_initial_state_1()
        L_tensor_list, R_tensor_list = dmrg.make_proto_L_R_tensors()

        # print("Initial state MPS")

        # for isite, tensor in enumerate(initial_state_mps):
        #     print(f"Site {isite}")
        #     print(repr(tensor))

        # print("Hamiltonian MPO")
        # print(simple_diagonal_hamiltonian_mpo)

        R_tensor_list = dmrg.make_remaining_initial_R_tensors(
            mpo=simple_diagonal_hamiltonian_mpo,
            num_sites=num_sites,
            mps_ket=initial_state_mps,
            R_tensor_list=R_tensor_list,
            verbosity=0,
        )
        # Stopped here because I determined the source of the issue we had that prompted this DMRG package
