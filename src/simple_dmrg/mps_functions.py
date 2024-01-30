from typing import List

import numpy as np
import scipy.sparse as spar


def add_mpos(mpo1: List[np.ndarray], mpo2: List[np.ndarray]) -> List[np.ndarray]:
    """Add two MPOs. This follows IVB of https://link.aps.org/doi/10.1103/PhysRevB.95.035129"""
    mpo1_physical_dims = mpo1[0].shape[1]
    mpo2_physical_dims = mpo2[0].shape[1]
    assert mpo1_physical_dims == mpo2_physical_dims, "Physical dimensions must match"

    new_mpo = []

    R1_shape = (
        1,
        mpo1_physical_dims,
        mpo1_physical_dims,
        mpo1[0].shape[-1] + mpo2[0].shape[-1],
    )

    R1 = np.zeros(R1_shape)
    R1[:, :, :, : mpo1[0].shape[-1]] = mpo1[0]
    R1[:, :, :, mpo1[0].shape[-1] :] = mpo2[0]
    new_mpo.append(R1)

    for isite in range(1, len(mpo1) - 1):
        R_i_shape = (
            mpo1[isite].shape[0] + mpo2[isite].shape[0],
            mpo1_physical_dims,
            mpo1_physical_dims,
            mpo1[isite].shape[-1] + mpo2[isite].shape[-1],
        )
        R_i = np.zeros(R_i_shape)
        R_i[: mpo1[isite].shape[0], :, :, : mpo1[isite].shape[-1]] = mpo1[isite]
        R_i[mpo1[isite].shape[0] :, :, :, mpo1[isite].shape[-1] :] = mpo2[isite]
        new_mpo.append(R_i)

    R_end_shape = (
        mpo1[-1].shape[0] + mpo2[-1].shape[0],
        mpo1_physical_dims,
        mpo1_physical_dims,
        1,
    )
    R_end = np.zeros(R_end_shape)
    R_end[: mpo1[-1].shape[0], :, :, :] = mpo1[-1]
    R_end[mpo1[-1].shape[0] :, :, :, :] = mpo2[-1]
    new_mpo.append(R_end)

    return new_mpo


def multiply_mpos(mpo1: List[np.ndarray], mpo2: List[np.ndarray]) -> List[np.ndarray]:
    """Multiply two MPOs. This follows IVA and Fig. 5 of https://link.aps.org/doi/10.1103/PhysRevB.95.035129
    Order is mp1 * mpo2, so mpo2 is on the top and mpo1 is on the bottom."""
    mpo1_physical_dims = mpo1[0].shape[1]
    mpo2_physical_dims = mpo2[0].shape[1]
    assert mpo1_physical_dims == mpo2_physical_dims, "Physical dimensions must match"

    new_mpo = []
    for isite in range(len(mpo1)):
        # Contract over the matching physical indices.
        # Ket is on the top, bra is on the bottom, so mpo2 is on the top.
        # mpo2[isite] : abcd: a (left,bond_dim) b (top,physical_dim,ket) c (bottom,physical_dim,bra) d (right,bond_dim)
        # mpo1[isite] : ecfg: e (left,bond_dim) c (top,physical_dim,ket) f (bottom,physical_dim,bra) g (right,bond_dim)
        # R_i : aebfdg: a (left top,bond_dim_2) e(left bottom,bond_dim_1) b (top,physical_dim,ket) f (bottom,physical_dim,bra) d (right top, bond_dim_2) g (right bottom, bond_dim_1)
        R_i = np.einsum(
            "abcd,ecfg->aebfdg",
            mpo2[isite],
            mpo1[isite],
        )

        # Merge d and g indices.
        # aebfdg -> aebfk
        R_i = np.reshape(
            R_i,
            (
                R_i.shape[0],
                R_i.shape[1],
                R_i.shape[2],
                R_i.shape[3],
                R_i.shape[4] * R_i.shape[5],
            ),
            order="C",
        )

        # Rotate out a and e indices.
        # aebfk -> bfkae
        R_i = np.transpose(R_i, (2, 3, 4, 0, 1))

        # Merge a and e indices.
        # bfkae -> bfkl
        R_i = np.reshape(
            R_i,
            (
                R_i.shape[0],
                R_i.shape[1],
                R_i.shape[2],
                R_i.shape[3] * R_i.shape[4],
            ),
            order="C",
        )

        # Rotate back to the original order.
        # bfkl -> lbfk
        R_i = np.transpose(R_i, (3, 0, 1, 2))

        # Now, lbfk : l (left,bond_dim_1*bond_dim_2) b (top,physical_dim,ket) f (bottom,physical_dim,bra) k (right,bond_dim_1*bond_dim_2)

        new_mpo.append(R_i)

    return new_mpo


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


def mpo_mult_by_scalar(
    mpo: List[np.ndarray], scalar: complex, deposit_site: int = None
) -> List[np.ndarray]:
    """Multiply an MPO by a scalar. If deposit_site is None, distribute the scalar evenly across all sites
    by taking the Nth root, where N is the number of sites.
    Otherwise, deposit the scalar on the given site."""
    new_mpo = []
    if deposit_site is None:
        scalar_per_site = scalar ** (1 / len(mpo))
        for isite in range(len(mpo)):
            new_mpo.append(mpo[isite] * scalar_per_site)

    else:
        for isite in range(len(mpo)):
            if isite == deposit_site:
                new_mpo.append(mpo[isite] * scalar)
            else:
                new_mpo.append(mpo[isite])

    return new_mpo


def mpo_to_dense_matrix(mpo: List[np.ndarray]):
    num_sites = len(mpo)
    # Contract the mpo tensors.
    for isite, tensor in enumerate(mpo):
        # print("isite", isite)
        # print("tensor.shape", tensor.shape)

        if isite == 0:
            final_tensor = tensor
        else:
            final_tensor = np.einsum("...a,abcd->...bcd", final_tensor, tensor)
        # print("final_tensor.shape", final_tensor.shape)
    # print("final_tensor.shape", final_tensor.shape)
    # Remove the dummy dimensions.
    final_tensor = np.squeeze(final_tensor)
    # print("final_tensor.shape", final_tensor.shape)

    # Rearrange the indices from ket,bra,ket,bra,... to ket,ket,...,bra,bra,...
    final_tensor = np.transpose(
        final_tensor,
        list(range(0, num_sites * 2, 2)) + list(range(1, num_sites * 2, 2)),
    )
    # print("final_tensor.shape", final_tensor.shape)

    # Reshape the rank-2N tensor to a matrix.
    final_matrix = np.reshape(
        final_tensor,
        (2**num_sites, 2**num_sites),
        order="C",
    )
    # Transpose the matrix to match the convention of the sparse matrix.
    final_matrix = np.transpose(final_matrix)
    # print("final_tensor.shape", final_tensor.shape)

    return final_matrix


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
