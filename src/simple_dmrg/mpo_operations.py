from typing import List

import numpy as np


def mpo_general_expectation(mps_bra, mpo, mps_ket):
    """Computes the expectation value of an MPO with respect to two MPSs
    See 4.4 and Fig.14 of https://doi.org/10.1140/epjb/s10051-023-00575-2
    Memory requirements may be reduced by replacing instead of assigning
    to new variable at each step."""
    assert len(mps_bra) == len(mps_ket)
    assert len(mps_bra) == len(mpo)

    mps_bra = [np.conj(tensor) for tensor in mps_bra]
    # Contract tensors using the extended zipper method
    # See 4.4, pg. 17 of https://doi.org/10.1140/epjb/s10051-023-00575-2
    contraction_tensor = np.eye(1)
    contraction_tensor = np.expand_dims(contraction_tensor, axis=-1)
    # print("Initial contraction_tensor shape: ", contraction_tensor.shape)
    # print(contraction_tensor)

    for iiter in range(len(mps_ket)):
        ket_tensor = mps_ket[iiter]
        bra_tensor = mps_bra[iiter]
        mpo_tensor = mpo[iiter]

        # print("Iteration ", iiter)
        # print("ket tensor shape: ", ket_tensor.shape)
        # print("bra tensor shape: ", bra_tensor.shape)
        # print("mpo tensor shape: ", mpo_tensor.shape)
        # print("contraction matrix shape: ", contraction_tensor.shape)

        # Contract ket tensor with contraction matrix
        # contraction_tensor: a (top) d (right) f (bottom)
        # ket_tensor: a (left) c (bottom) b (right)
        # c_ket_tensor: f (left) d (bottom left) c (bottom right) b (right)
        c_ket_tensor = np.einsum("adf,acb->fdcb", contraction_tensor, ket_tensor)
        # print("c_ket_tensor shape: ", c_ket_tensor.shape)
        # print(c_ket_tensor)

        # Contract with mpo tensor
        # mpo_tensor: d (left) c (top) g (bottom) e(right)
        # c_ket_tensor: f (left) d (bottom left) c (bottom right) b (right)
        # c_ket_o_tensor: b (top) e (right) f (bottom left) g (bottom right)
        c_ket_o_tensor = np.einsum("fdcb,dcge->befg", c_ket_tensor, mpo_tensor)
        # print("c_ket_o_tensor shape: ", c_ket_o_tensor.shape)
        # print(c_ket_o_tensor)

        # Contract with bra tensor
        # c_ket_o_tensor: b (top) e (right) f (bottom left) g (bottom right)
        # bra_tensor: f (left) g (top) h (right)
        # contraction_tensor: b (top) e (right) h (bottom)
        contraction_tensor = np.einsum("befg,fgh->beh", c_ket_o_tensor, bra_tensor)
        # print("contraction_tensor shape: ", contraction_tensor.shape)
        # print(c_ket_tensor)

    assert contraction_tensor.shape == (1, 1, 1)
    return contraction_tensor[0, 0, 0]


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


def mpo_to_dense_matrix(mpo: List[np.ndarray], verbosity: int = 0):
    num_sites = len(mpo)
    # Contract the mpo tensors.
    for isite, tensor in enumerate(mpo):
        if verbosity > 0:
            print("isite", isite)
            print("tensor.shape", tensor.shape)

        if isite == 0:
            final_tensor = tensor
        else:
            final_tensor = np.einsum("...a,abcd->...bcd", final_tensor, tensor)
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

    # Rearrange the indices from ket,bra,ket,bra,... to ket,ket,...,bra,bra,...
    final_tensor = np.transpose(
        final_tensor,
        list(range(0, num_sites * 2, 2)) + list(range(1, num_sites * 2, 2)),
    )
    if verbosity > 0:
        print("Indices rearranged.")
        print("final_tensor.shape", final_tensor.shape)

    # Reshape the rank-2N tensor to a matrix.
    final_matrix = np.reshape(
        final_tensor,
        (2**num_sites, 2**num_sites),
        order="C",
    )
    # Transpose the matrix to match the convention of the sparse matrix.
    final_matrix = np.transpose(final_matrix)
    if verbosity > 0:
        print("Matrix formed.")
        print("final_matrix.shape", final_matrix.shape)

    return final_matrix
