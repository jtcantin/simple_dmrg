from typing import List

import numpy as np
import scipy.sparse as spar

import simple_dmrg.mpo_construction as make_mpo
import simple_dmrg.mpo_operations as mpo_ops


def statevector_to_mps(normalize: bool = True) -> List[np.ndarray]:
    pass


def mps_to_statevector():
    pass


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
