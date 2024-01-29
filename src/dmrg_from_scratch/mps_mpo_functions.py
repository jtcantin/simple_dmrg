from typing import List

import numpy as np
import scipy.sparse as spar


def fermion_on_site_annihilation_operator() -> np.ndarray:
    return np.array(
        [
            [0, 0],  # Bra is on the right, ket is on the left,
            [1, 0],  # corresponds to ket on the top and bra on the bottom for the MPO
        ]
    )


def fermion_on_site_creation_operator() -> np.ndarray:
    return np.array(
        [
            [0, 1],  # Bra is on the right, ket is on the left,
            [0, 0],  # corresponds to ket on the top and bra on the bottom for the MPO
        ]
    )


def on_site_parity_operator() -> np.ndarray:
    return np.array(
        [
            [1, 0],
            [0, -1],
        ]
    )


def fermion_operator_mpo(site: int, num_sites: int, op_type: str) -> np.ndarray:
    """Return the fermion creation operator for a given site.
    On-site, there is a creation operator and a string of
    parity operators before it. All bond dimensions are 1.
    This comes from a combination of Sec. III, Sec. III.A of https://link.aps.org/doi/10.1103/PhysRevB.95.035129
    and Sec. 2.1.1 of http://dx.doi.org/10.1016/j.cpc.2014.08.019
    """
    assert site >= 0
    assert site < num_sites
    assert num_sites > 0

    mpo = []
    # Add the dummy bond indices.
    if op_type == "creation":
        on_site_tensor = np.expand_dims(fermion_on_site_creation_operator(), axis=0)
        on_site_tensor = np.expand_dims(on_site_tensor, axis=-1)
    elif op_type == "annihilation":
        on_site_tensor = np.expand_dims(fermion_on_site_annihilation_operator(), axis=0)
        on_site_tensor = np.expand_dims(on_site_tensor, axis=-1)
    else:
        raise ValueError("op_type must be 'creation' or 'annihilation'")
    off_site_left_tensor = np.expand_dims(on_site_parity_operator(), axis=0)
    off_site_left_tensor = np.expand_dims(off_site_left_tensor, axis=-1)
    off_site_right_tensor = np.expand_dims(np.eye(2), axis=0)
    off_site_right_tensor = np.expand_dims(off_site_right_tensor, axis=-1)

    # print("on_site_tensor.shape", on_site_tensor.shape)
    # print("off_site_left_tensor.shape", off_site_left_tensor.shape)
    # print("off_site_right_tensor.shape", off_site_right_tensor.shape)

    # Put parity operators on the left of the site, and identity operators on the right.
    # In combination with κκ†+κ†κ=1 on the same site, where κ is fermion_on_site_annihilation_operator,
    # this accounts for the anti-commutation relations of the fermion operators.
    for isite in range(num_sites):
        if isite == site:
            mpo.append(on_site_tensor)
        elif isite < site:
            mpo.append(off_site_left_tensor)
        else:
            mpo.append(off_site_right_tensor)

    return mpo


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
    rng = np.random.RandomState(seed)
    state_vec = rng.random(2**num_qubits, dtype=dtype)
    if normalize:
        state_vec /= np.linalg.norm(state_vec)
    return state_vec


def sparse_matrix_to_mpo(
    matrix_elements: List[tuple],
    num_physical_dims: int = 2,
    num_sites: int = 4,
    bond_dim: int = 3,
    verbosity: int = 0,
) -> List[np.ndarray]:
    """Convert a sparse matrix to an MPO. NOT WORKING
    Assumes matrix index is (n_0,n_1,...,n_N-1,n_N) where n_i runs over the physical dimensions of site i,
     N is the number of sites, n_0 is the leftmost index, n_N runs the fastest (i.e. C's row-major order).
     For the MPO, each tensor (all rank-4) corresponds to a site where the indices abcd are ordered as
     abcd: a (left,bond_dim) b (top,physical_dim,ket) c (bottom,physical_dim,bra) d (right,bond_dim).

    matrix_elements should be a list of 3-tuples of the form (value,bra,ket) such that each vector (i.e. ket or bra)
    is of the form (n_0,n_1,...,n_N-1,n_N) and the elements of the vector indicate the corresponding
    physical index of the site, starting at 0. value = <bra|operator|ket>

     The MPO is obtained by adding the matrix_elements into a rank-2N tensor that covers the full fock space,
     then sequentially performing SVD site-by-site with each local matrix fusing the physical indices into
     one index and the remaining indices into another. The SVD is truncated to bond_dim at each site.
     The end sites have dummy outer indices of dimension 1. The s matrix is kept in the original site
     and the vh matrix is absorbed into the remaining tensor.
    """
    raise NotImplementedError("sparse_matrix_to_mpo not working")
    # The first half of the dimensions are for the top indices (ket) and the second half are for the bottom indices (bra).
    high_rank_tensor = np.zeros([num_physical_dims] * num_sites * 2, dtype=complex)
    print(high_rank_tensor.shape)

    # Add the matrix elements to the tensor.
    for value, bra, ket in matrix_elements:
        high_rank_tensor[ket + bra] = value

    # SVD sequentially site-by-site.
    mpo = []
    for isite in range(num_sites):
        if verbosity > 0:
            print(f"Site {isite}")

        if isite == 0:
            ket_index = 0
            bra_index = num_sites
            all_indices = list(range(num_sites * 2))
            all_indices.remove(ket_index)
            all_indices.remove(bra_index)
            # Bring the ket and bra indices to the right.
            high_rank_tensor = np.transpose(
                high_rank_tensor, all_indices + [ket_index, bra_index]
            )

            temp_mat = np.reshape(
                high_rank_tensor,
                (num_physical_dims * num_physical_dims, -1),
                order="C",
            )
            # Perform the SVD.
            # SVD
            u, s, vh = np.linalg.svd(
                temp_mat,
                full_matrices=False,  # Means that the the "thin" or "reduced" SVD is computed;
                # see 4.2 of https://doi.org/10.1140/epjb/s10051-023-00575-2 for more details
                compute_uv=True,
                hermitian=False,
            )
            # Truncate the SVD.
            if u.shape[1] > bond_dim:
                u = u[:, :bond_dim]
                s = s[:bond_dim]
                vh = vh[:bond_dim, :]
            else:
                effective_bond_dim = u.shape[1]

            # Reshape the u matrix to the correct shape.
            u = np.reshape(
                u, (num_physical_dims, num_physical_dims, effective_bond_dim), order="C"
            )

            # Add a dummy dimension to u
            # abcd: a (left,bond_dim) b (top,physical_dim,ket) c (bottom,physical_dim,bra) d (right,bond_dim)
            u = np.expand_dims(u, axis=0)

            # Contract the s matrix with the u matrix.
            u = np.einsum("abcd,d->abcd", u, s)

            # Reshape the vh matrix to the tensor shape.
            vh = np.reshape(
                vh,
                (
                    [effective_bond_dim]
                    + [num_physical_dims] * (num_sites - (isite + 1)) * 2
                ),
                order="C",
            )

            mpo.append(u)
            high_rank_tensor = vh

        elif isite == num_sites - 1:
            num_sites_remaining = num_sites - isite
            bond_index = 0
            ket_index = 1  # Index 0 is now the bond index.
            bra_index = ket_index + num_sites_remaining
            all_indices = [0] + list(range(1, (num_sites_remaining * 2) + 1))
            all_indices.remove(ket_index)
            all_indices.remove(bra_index)
            all_indices.remove(bond_index)

            orig_effective_bond_dim = high_rank_tensor.shape[bond_index]

            # Bring the ket and bra indices to the right.
            high_rank_tensor = np.transpose(
                high_rank_tensor, all_indices + [bond_index, ket_index, bra_index]
            )

            temp_mat = np.reshape(
                high_rank_tensor,
                (num_physical_dims * orig_effective_bond_dim * num_physical_dims, -1),
                order="C",
            )
            print("temp_mat.shape", temp_mat.shape)
            # The last site already has a dummy dimension of size 1.
            # temp_mat = np.expand_dims(temp_mat, axis=-1)
            # print("temp_mat.shape with dummy:", temp_mat.shape)

            # Perform the SVD.
            # SVD
            u, s, vh = np.linalg.svd(
                temp_mat,
                full_matrices=False,  # Means that the the "thin" or "reduced" SVD is computed;
                # see 4.2 of https://doi.org/10.1140/epjb/s10051-023-00575-2 for more details
                compute_uv=True,
                hermitian=False,
            )
            print("u.shape", u.shape)
            print("s.shape", s.shape)
            print("vh.shape", vh.shape)
            # Truncation not needed for the last site.
            # # Truncate the SVD.
            # u = u[:, :bond_dim]
            # s = s[:bond_dim]
            # vh = vh[:bond_dim, :]

            # Reshape the u matrix to the correct shape.
            # abcd: a (left,bond_dim) b (top,physical_dim,ket) c (bottom,physical_dim,bra) d (right,bond_dim)
            print(u.shape)
            print(u)
            u = np.reshape(
                u,
                (orig_effective_bond_dim, num_physical_dims, num_physical_dims),
                order="C",
            )
            # Add a dummy dimension to u
            u = np.expand_dims(u, axis=-1)
            print(u.shape)
            print(s.shape)
            # Combine the s matrix with the u tensor.
            u = np.einsum("abcd,d->abcd", u, s)

            # Discard the vh matrix, which is now a scalar global phase
            assert vh.shape == (1, 1)
            mpo.append(u)

        else:
            num_sites_remaining = num_sites - isite
            bond_index = 0
            ket_index = 1  # Index 0 is now the bond index.
            bra_index = ket_index + num_sites_remaining
            all_indices = [0] + list(range(1, (num_sites_remaining * 2) + 1))
            all_indices.remove(ket_index)
            all_indices.remove(bra_index)
            all_indices.remove(bond_index)

            orig_effective_bond_dim = high_rank_tensor.shape[bond_index]

            # Bring the ket and bra indices to the right.
            high_rank_tensor = np.transpose(
                high_rank_tensor, all_indices + [bond_index, ket_index, bra_index]
            )

            temp_mat = np.reshape(
                high_rank_tensor,
                (num_physical_dims * orig_effective_bond_dim * num_physical_dims, -1),
                order="C",
            )
            # Perform the SVD.
            # SVD
            u, s, vh = np.linalg.svd(
                temp_mat,
                full_matrices=False,  # Means that the the "thin" or "reduced" SVD is computed;
                # see 4.2 of https://doi.org/10.1140/epjb/s10051-023-00575-2 for more details
                compute_uv=True,
                hermitian=False,
            )
            # Truncate the SVD.
            if u.shape[1] > bond_dim:
                u = u[:, :bond_dim]
                s = s[:bond_dim]
                vh = vh[:bond_dim, :]
            else:
                effective_bond_dim = u.shape[1]

            # Reshape the u matrix to the correct shape.
            # abcd: a (left,bond_dim) b (top,physical_dim,ket) c (bottom,physical_dim,bra) d (right,bond_dim)
            u = np.reshape(
                u,
                (
                    orig_effective_bond_dim,
                    num_physical_dims,
                    num_physical_dims,
                    effective_bond_dim,
                ),
                order="C",
            )

            # Combine the s matrix with the u tensor.
            u = np.einsum("abcd,d->abcd", u, s)

            # Reshape the vh matrix to the tensor shape.
            vh = np.reshape(
                vh,
                (
                    [effective_bond_dim]
                    + [num_physical_dims] * (num_sites_remaining - 1) * 2
                ),
                order="C",
            )

            mpo.append(u)
            high_rank_tensor = vh

    return mpo


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
        print("isite", isite)
        print("tensor.shape", tensor.shape)

        if isite == 0:
            final_tensor = tensor
        else:
            final_tensor = np.einsum("...a,abcd->...bcd", final_tensor, tensor)
        print("final_tensor.shape", final_tensor.shape)
    print("final_tensor.shape", final_tensor.shape)
    # Remove the dummy dimensions.
    final_tensor = np.squeeze(final_tensor)
    print("final_tensor.shape", final_tensor.shape)

    # Rearrange the indices from ket,bra,ket,bra,... to ket,ket,...,bra,bra,...
    final_tensor = np.transpose(
        final_tensor,
        list(range(0, num_sites * 2, 2)) + list(range(1, num_sites * 2, 2)),
    )
    print("final_tensor.shape", final_tensor.shape)

    # Reshape the rank-2N tensor to a matrix.
    final_matrix = np.reshape(
        final_tensor,
        (2**num_sites, 2**num_sites),
        order="C",
    )
    # Transpose the matrix to match the convention of the sparse matrix.
    final_matrix = np.transpose(final_matrix)
    print("final_tensor.shape", final_tensor.shape)

    return final_matrix


def on_site_number_operator_mpo(site: int, num_sites: int) -> List[np.ndarray]:
    """Return the MPO for the number operator on a given site."""
    # Get separate creation and annihilation operators for the site.
    creation_op = fermion_operator_mpo(site, num_sites, op_type="creation")
    annihilation_op = fermion_operator_mpo(site, num_sites, op_type="annihilation")
    # Multiply them together to get the number operator.
    number_op = multiply_mpos(creation_op, annihilation_op)
    return number_op


def make_particle_number_mpo(num_sites: int) -> List[np.ndarray]:
    """Return the MPO for the particle number operator,
    which is the sum of the number operators on each site."""
    for isite in range(num_sites):
        if isite == 0:
            number_op = on_site_number_operator_mpo(isite, num_sites)
        else:
            number_op = add_mpos(
                number_op, on_site_number_operator_mpo(isite, num_sites)
            )
    return number_op


def make_one_body_mpo(one_body_tensor: np.ndarray, num_sites: int) -> List[np.ndarray]:
    """Return the MPO for a one-body operator.
    The one-body tensor should be of the form
    <bra|operator|ket> where bra and ket are vectors of the form
    (n_0,n_1,...,n_N-1,n_N) where n_i runs over the physical dimensions of site i,
    N is the number of sites, n_0 is the leftmost index, n_N runs the fastest (i.e. C's row-major order).
    """
    assert one_body_tensor.shape == (num_sites, num_sites)
    for isite in range(num_sites):
        c_dagger_i = fermion_operator_mpo(isite, num_sites, op_type="creation")
        for jsite in range(num_sites):
            c_j = fermion_operator_mpo(jsite, num_sites, op_type="annihilation")
            bare_op = multiply_mpos(c_dagger_i, c_j)
            # Put the one-body tensor element on the creation site.
            scalar_and_op = mpo_mult_by_scalar(
                bare_op, one_body_tensor[isite, jsite], deposit_site=isite
            )

            if isite == 0 and jsite == 0:
                one_body_mpo = scalar_and_op
            else:
                one_body_mpo = add_mpos(one_body_mpo, scalar_and_op)
    return one_body_mpo


def make_identity_mpo(num_sites: int, num_physical_dims) -> List[np.ndarray]:
    """Return the MPO for the identity operator."""
    identity_tensor = np.expand_dims(np.eye(num_physical_dims), axis=0)
    identity_tensor = np.expand_dims(identity_tensor, axis=-1)
    identity_mpo = []
    for isite in range(num_sites):
        identity_mpo.append(identity_tensor)
    return identity_mpo
