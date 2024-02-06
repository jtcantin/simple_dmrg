from typing import List

import numpy as np
import scipy.sparse as spar

import simple_dmrg.mpo_construction as make_mpo
import simple_dmrg.mpo_operations as mpo_ops


def statevector_to_mps(
    state_vector: np.ndarray,
    num_sites: int,
    physical_dim: int,
    right_normalize: bool = True,
    orig_bond_dim: int = None,
    verbosity: int = 0,
) -> List[np.ndarray]:
    """Converts a state vector to an MPS.
    Assumes state vector index is (n_0,n_1,...,n_N-1,n_N) where n_i runs over the physical dimensions of site i,
    N is the number of sites, n_0 is the leftmost index, n_N runs the fastest (i.e. C's row-major order).
    For the MPO, each tensor (all rank-4) corresponds to a site where the indices abcd are ordered as
    abcd: a (left,bond_dim) b (bottom,physical_dim) c (right,bond_dim).
    Only tested for single slater determinants for now."""

    # Check vector is correct length
    assert (
        state_vector.shape[0] == physical_dim**num_sites
    ), "State vector is not the correct length"

    if verbosity > 0:
        print("state_vector.shape", state_vector.shape)
    # Perform SVD step by step, starting from the right
    # The first tensor and the last tensor have a dummy outer bond dimension
    # If bond_dim is not specified, use the bond dimension determined from SVD
    # If right_normalize is True, the final u and s are discarded. If False, they are incorporated
    # into the last tensor
    max_bond_dim = 0
    mps = []
    # Reshape to matrix for last site
    state_vector_new = np.reshape(state_vector, (-1, physical_dim), order="C")
    if verbosity > 0:
        print("state_vector_new.shape", state_vector_new.shape)
    # SVD
    u, s, vh = np.linalg.svd(
        state_vector_new,
        full_matrices=False,
        compute_uv=True,
        hermitian=False,
    )
    bond_dim = orig_bond_dim
    if verbosity > 0:
        print("bond_dim", bond_dim)
        print("u.shape", u.shape)
        print("s.shape", s.shape)
        print("vh.shape", vh.shape)
        print("state_vector_new.shape", state_vector_new.shape)

    # If orig_bond_dim is specified, either truncate or pad the SVD matrices
    if orig_bond_dim is not None:
        raise NotImplementedError
        if s.shape[0] > bond_dim:
            s = s[:bond_dim]
            u = u[:, :bond_dim]
            vh = vh[:bond_dim, :]
        else:
            u = np.pad(
                u,
                ((0, 0), (0, bond_dim - s.shape[0])),
                mode="constant",
                constant_values=0,
            )
            vh = np.pad(
                vh,
                ((bond_dim - s.shape[0], 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            s = np.pad(
                s, (0, bond_dim - s.shape[0]), mode="constant", constant_values=0
            )
            # Ensure shapes are correct
            assert u.shape == (
                physical_dim ** (num_sites - 1),
                bond_dim,
            ), f"u.shape: {u.shape} expected({physical_dim**(num_sites-1)}, {bond_dim})"
            assert s.shape == (bond_dim,), f"s.shape: {s.shape} bond_dim: {bond_dim}"
            assert vh.shape == (
                bond_dim,
                physical_dim,
            ), f"vh.shape: {vh.shape} bond_dim: {bond_dim} physical_dim: {physical_dim}"

            # Check that original state vector can be reconstructed
            test_vector = np.einsum("ab,b,bc->ac", u, s, vh)
            assert np.allclose(
                test_vector, state_vector_new
            ), f"test_vector: {test_vector} state_vector_new: {state_vector_new}"
    else:
        bond_dim = s.shape[0]

    max_bond_dim = max(max_bond_dim, bond_dim)

    # Add dummy outer bond dimension
    vh = np.expand_dims(vh, axis=-1)
    mps.append(vh)

    # Contract u and s to the left
    state_vector_new = np.einsum("ab,b->ab", u, s)
    if verbosity > 0:
        print("bond_dim", bond_dim)
        print("u.shape", u.shape)
        print("s.shape", s.shape)
        print("vh.shape", vh.shape)
        print("state_vector_new.shape", state_vector_new.shape)
        print("mps", mps)
    old_bond_dim = bond_dim
    for isite in range(num_sites - 2, -1, -1):
        # Reshape to matrix
        if verbosity > 0:
            print("isite", isite)

            print("bond_dim", bond_dim)
            print("old_bond_dim", old_bond_dim)
            print("state_vector_new.shape", state_vector_new.shape)

        state_vector_new = np.reshape(
            state_vector_new, (-1, physical_dim * bond_dim), order="C"
        )
        if verbosity > 0:
            print("bond_dim", bond_dim)
            print("state_vector_new.shape", state_vector_new.shape)

        # SVD
        u, s, vh = np.linalg.svd(
            state_vector_new,
            full_matrices=False,
            compute_uv=True,
            hermitian=False,
        )
        bond_dim = vh.shape[0]
        if verbosity > 0:
            print("bond_dim", bond_dim)
            print("u.shape", u.shape)
            print("s.shape", s.shape)
            print("vh.shape", vh.shape)

        # If orig_bond_dim is specified, either truncate or pad the SVD matrices
        if orig_bond_dim is not None:
            if s.shape[0] > bond_dim:
                s = s[:bond_dim]
                u = u[:, :bond_dim]
                vh = vh[:bond_dim, :]
            else:
                u = np.pad(
                    u,
                    ((0, 0), (0, bond_dim - s.shape[0])),
                    mode="constant",
                    constant_values=0,
                )
                vh = np.pad(
                    vh,
                    ((bond_dim - s.shape[0], 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                s = np.pad(
                    s, (0, bond_dim - s.shape[0]), mode="constant", constant_values=0
                )
                # Check that original state vector can be reconstructed
                test_vector = np.einsum("ab,b,bc->ac", u, s, vh)
                assert np.allclose(test_vector, state_vector_new)
        else:
            bond_dim = s.shape[0]

        max_bond_dim = max(max_bond_dim, bond_dim)
        if verbosity > 0:
            print("bond_dim", bond_dim)
            print("u.shape", u.shape)
            print("s.shape", s.shape)
            print("vh.shape", vh.shape)

        # Reshape to tensor
        vh = np.reshape(vh, (bond_dim, physical_dim, old_bond_dim), order="C")

        if verbosity > 0:
            print("bond_dim", bond_dim)
            print("u.shape", u.shape)
            print("s.shape", s.shape)
            print("vh.shape", vh.shape)

        mps = [vh] + mps
        old_bond_dim = bond_dim

        # Contract u and s to the left for all but the last tensor
        # For the last tensor, u is a complex number of absolute value 1 and s is the norm of the MPS,
        # so if right_normalize is True, we just discard them to normalize the MPS and set the global phase to 1
        if isite > 0:
            state_vector_new = np.einsum("ab,b->ab", u, s)
        else:
            assert np.allclose(
                np.abs(u), 1
            ), "u is not a complex number of absolute value 1"
            assert s.shape[0] == 1, "s is not a scalar"
            assert np.allclose(
                s[0], np.linalg.norm(state_vector)
            ), "s is not the norm of the state vector"
            if verbosity > 0:
                print("MPS norm: ", s[0])
                print("Final u: ", u)

            if right_normalize:
                continue
            else:
                mps[0] = np.einsum("ab,b,bcd->acd", u, s, mps[0])

    if verbosity > 0:
        print("max_bond_dim", max_bond_dim)
        for tensor in mps:
            print(tensor.shape)
    return mps


def mps_to_statevector(mps: List[np.ndarray], verbosity: int = 0):
    num_sites = len(mps)
    # Contract the mpo tensors.
    for isite, tensor in enumerate(mps):
        if verbosity > 0:
            print("isite", isite)
            print("tensor.shape", tensor.shape)

        if isite == 0:
            final_tensor = tensor
        else:
            final_tensor = np.einsum("...a,abc->...bc", final_tensor, tensor)
        if verbosity > 0:
            print("final_tensor.shape", final_tensor.shape)

    if verbosity > 0:
        print("Final tensor formed.")
        print("final_tensor.shape", final_tensor.shape)

    # Remove the dummy dimensions.
    final_tensor = np.squeeze(final_tensor)
    if verbosity > 0:
        print("Dummy dimensions removed.")
        print("final_tensor.shape", final_tensor.shape)

    # Reshape the rank-N tensor to a vector.
    final_vector = np.reshape(
        final_tensor,
        (-1),
        order="C",
    )

    if verbosity > 0:
        print("Vector formed.")
        print("final_vector.shape", final_vector.shape)

    return final_vector


def occupation_numbers_to_statevector(
    occupation_numbers: np.ndarray, num_sites: int, physical_dim: int
) -> np.ndarray:
    """Converts occupation numbers to a state vector.
    occupation_numbers is a 1D array of length num_sites,
    where each element indicates which physical state is occupied.
    The state vector is a 1D array of length physical_dim**num_sites.
    The state vector index is (n_0,n_1,...,n_N-1,n_N) where n_i runs
    over the physical dimensions of site i, N is the number of sites,
    n_0 is the leftmost index, n_N runs the fastest (i.e. C's row-major order).
    """

    assert occupation_numbers.shape == (
        num_sites,
    ), f"occupation_numbers has the wrong shape: {occupation_numbers.shape}"
    state_vector = np.zeros(shape=(physical_dim**num_sites,), dtype=np.complex_)

    occupation_numbers_matrix = np.zeros((num_sites, physical_dim))
    for i, n in enumerate(occupation_numbers):
        occupation_numbers_matrix[i, n] = 1

    # # Add vacuum states
    # if occupation_numbers.shape == (num_sites,):
    #     # Add dummy dimension
    #     occupation_numbers = np.expand_dims(occupation_numbers, axis=1)

    # # occupation_numbers = np.hstack(
    # #     (np.zeros((num_sites, 1), dtype=int), occupation_numbers)
    # # )
    state_vector_index = 0
    for isite in range(num_sites):
        for iphysical in range(physical_dim):
            if occupation_numbers_matrix[isite, iphysical] == 1:
                state_vector_index += iphysical * physical_dim ** (
                    num_sites - isite - 1
                )

    state_vector[state_vector_index] = 1

    return state_vector


# def recursive_indexing(num_sites: int, physical_dim: int,indices:List[List[int]])->List[List[int]]:
#     """Generate all possible occupation numbers for a given number of sites and physical dimensions"""
#     if num_sites==0:
#         return indices
#     else:
#         new_indices=[]
#         state = indices[-1]
#         for
def generate_random_statevector(
    num_qubits: int,
    seed: int = 0,
    dtype: np.dtype = np.complex128,
    normalize: bool = True,
):
    """Generate a random statevector."""
    rng = np.random.default_rng(seed)
    state_vec = rng.random(2**num_qubits, dtype=dtype)
    if normalize:
        state_vec /= np.linalg.norm(state_vec)
    return state_vec


def build_random_mps(
    num_sites: int = 4,
    physical_dimension: int = 2,
    bond_dimension: int = 3,
    seed: int = 0,
) -> List[np.ndarray]:
    """Builds a random MPS with bond dimension bond_dimension"""
    # Initialize MPS with random numbers
    rng = np.random.default_rng(seed)
    mps = []
    for i in range(num_sites):
        # First and last tensors have a dummy outer bond dimension
        if i == 0:
            mps.append(
                rng.uniform(low=-1, high=1, size=(physical_dimension, bond_dimension))
            )
            mps[i] = np.expand_dims(mps[i], axis=0)
        elif i == num_sites - 1:
            mps.append(
                rng.uniform(low=-1, high=1, size=(bond_dimension, physical_dimension))
            )
            mps[i] = np.expand_dims(mps[i], axis=-1)
        else:
            mps.append(
                rng.uniform(
                    low=-1,
                    high=1,
                    size=(bond_dimension, physical_dimension, bond_dimension),
                )
            )

    # for tensor in mps:
    #     print(tensor.shape)
    #     print(tensor)

    # # Normalize MPS
    # mps_norm = mps_inner_product(mps, mps)
    # print("MPS norm: ", mps_norm)
    # mps = [tensor / np.sqrt(mps_norm) for tensor in mps]
    # print(mps_inner_product(mps, mps))

    # assert np.allclose(mps_inner_product(mps, mps), 1.0)
    return mps


def mps_tensor_svd_right_normalized(tensor: np.ndarray) -> List[np.ndarray]:
    """Transforms an MPS tensor to right-canonical form
    Follows Fig 11b of https://doi.org/10.1140/epjb/s10051-023-00575-2
    mps indices are abg"""
    tensor_orig_shape = tensor.shape
    # Fuse bg
    tensor = np.reshape(tensor, (tensor.shape[0], -1), order="C")

    # SVD
    u, s, vh = np.linalg.svd(
        tensor,
        full_matrices=False,  # Means that the the "thin" or "reduced" SVD is computed;
        # see 4.2 of https://doi.org/10.1140/epjb/s10051-023-00575-2 for more details
        compute_uv=True,
        hermitian=False,
    )

    # Split bg
    vh = np.reshape(
        vh, (s.shape[0], tensor_orig_shape[1], tensor_orig_shape[2]), order="C"
    )

    return [u, s, vh]


def right_normalize_mps(mps: List[np.ndarray]) -> List[np.ndarray]:
    """Transforms an MPS to right-canonical form
    Follows Fig 11b of https://doi.org/10.1140/epjb/s10051-023-00575-2
    mps indices are abg"""

    for iiter in range(len(mps) - 1, -1, -1):
        tensor = mps[iiter]
        # print("Tensor ", iiter)
        # print(tensor.shape)
        # Do SVD
        u, s, vh = mps_tensor_svd_right_normalized(tensor)
        # print("u shape: ", u.shape)
        # print("s shape: ", s.shape)
        # print("vh shape: ", vh.shape)

        # Update tensor
        mps[iiter] = vh
        # print("New tensor shape: ", mps[iiter].shape)

        # Contract u and s to the left for all but the last tensor
        # For the last tensor, u is a complex number of absolute value 1 and s is the norm of the MPS,
        # so we just discard them to normalize the MPS and set the global phase to 1
        if iiter > 0:
            mps[iiter - 1] = np.einsum("gda,ae,ez->gdz", mps[iiter - 1], u, np.diag(s))
            # print("New MPS shape: ", mps[iiter - 1].shape)
        # else:
        #     print("MPS norm: ", s[0])
        #     print("Final u: ", u)

    return mps


def mps_inner_product(
    mps_ket: List[np.ndarray], mps_bra: List[np.ndarray]
) -> np.float_:
    """Computes the inner product of two MPSs"""
    # Take conjugate transpose of bra MPS
    mps_bra = [np.conj(tensor) for tensor in mps_bra]
    # Contract tensors using the zipper method
    # See 4.3.1 and Fig. 9 of  https://doi.org/10.1140/epjb/s10051-023-00575-2
    contraction_matrix = np.eye(1)
    for iiter in range(len(mps_ket)):
        ket_tensor = mps_ket[iiter]
        bra_tensor = mps_bra[iiter]

        c_ket_tensor = np.einsum("ab,agd->bgd", contraction_matrix, ket_tensor)
        contraction_matrix = np.einsum("bgd,bge->de", c_ket_tensor, bra_tensor)

    assert contraction_matrix.shape == (1, 1)
    return contraction_matrix[0, 0]


def mps_tensor_svd_left_normalized(tensor: np.ndarray) -> List[np.ndarray]:
    """Transforms an MPS tensor to left-canonical form
    Follows Fig 11b of https://doi.org/10.1140/epjb/s10051-023-00575-2
    mps indices are abg"""
    tensor_orig_shape = tensor.shape
    # Fuse bg
    tensor = np.reshape(
        tensor, (tensor.shape[0] * tensor.shape[1], tensor.shape[2]), order="C"
    )

    # SVD
    u, s, vh = np.linalg.svd(
        tensor,
        full_matrices=False,  # Means that the the "thin" or "reduced" SVD is computed;
        # see 4.2 of https://doi.org/10.1140/epjb/s10051-023-00575-2 for more details
        compute_uv=True,
        hermitian=False,
    )

    # Split ab
    u = np.reshape(
        u, (tensor_orig_shape[0], tensor_orig_shape[1], s.shape[0]), order="C"
    )

    return [u, s, vh]


def calculate_on_site_densities(mps: List[np.ndarray]):
    """Calculate the on-site densities of an MPS using an operator MPO"""
    densities = np.zeros(shape=(len(mps),), dtype=np.complex_)
    for isite in range(len(mps)):
        density_mpo = make_mpo.on_site_number_operator_mpo(
            site=isite, num_sites=len(mps)
        )
        density = mpo_ops.mpo_general_expectation(
            mps_bra=mps, mpo=density_mpo, mps_ket=mps
        )
        densities[isite] = density

    return densities


# def make_density_mpos(num_sites: int) -> List[List[np.ndarray]]:
#     """Return a list of on-site density MPOs.
#     Assume 2 physical dimensions per site."""
#     density_mpo_list = []
#     for isite in range(num_sites):
#         density_mpo = on_site_number_operator_mpo(site=isite, num_sites=num_sites)
#         density_mpo_list.append(density_mpo)
#     return density_mpo_list
