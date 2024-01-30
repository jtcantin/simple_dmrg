from typing import List

import numpy as np

from simple_dmrg.mps_functions import add_mpos, mpo_mult_by_scalar, multiply_mpos


def build_random_mpo(
    num_sites: int = 4,
    physical_dimension: int = 2,
    bond_dimension: int = 3,
    seed: int = 0,
    # hermitian: bool = True,
) -> List[np.ndarray]:
    """Builds a random MPO with bond dimension bond_dimension
    Dimensions abcd are left (bond), top (physical), bottom (physical), right (bond)"""

    # Make Hermitian matrix
    matrix_dimension = num_sites**physical_dimension
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(low=-1, high=1, size=(matrix_dimension, matrix_dimension))
    matrix = (matrix + matrix.T.conj()) / 2

    # Convert to MPO

    raise NotImplementedError("This does not yet produce a Hermitian MPO.")
    """
    rng = np.random.default_rng(seed)
    mpo = []
    for i in range(num_sites):
        # First and last tensors have a dummy outer bond dimension
        if i == 0:
            mpo.append(
                rng.uniform(
                    low=-1,
                    high=1,
                    size=(physical_dimension, physical_dimension, bond_dimension),
                )
            )
            mpo[i] = np.expand_dims(mpo[i], axis=0)
        elif i == num_sites - 1:
            mpo.append(
                rng.uniform(
                    low=-1,
                    high=1,
                    size=(bond_dimension, physical_dimension, physical_dimension),
                )
            )
            mpo[i] = np.expand_dims(mpo[i], axis=-1)
        else:
            mpo.append(
                rng.uniform(
                    low=-1,
                    high=1,
                    size=(
                        bond_dimension,
                        physical_dimension,
                        physical_dimension,
                        bond_dimension,
                    ),
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

    # if hermitian:
    #     # Make MPO Hermitian
    #     # Fuse indices to a matrix
    #     # Make Hermitian by adding conjugate transpose
    #     # Split indices
    return mpo
    """


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


def make_fermion_operator_mpo(site: int, num_sites: int, op_type: str) -> np.ndarray:
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


def on_site_number_operator_mpo(site: int, num_sites: int) -> List[np.ndarray]:
    """Return the MPO for the number operator on a given site."""
    # Get separate creation and annihilation operators for the site.
    creation_op = make_fermion_operator_mpo(site, num_sites, op_type="creation")
    annihilation_op = make_fermion_operator_mpo(site, num_sites, op_type="annihilation")
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
        c_dagger_i = make_fermion_operator_mpo(isite, num_sites, op_type="creation")
        for jsite in range(num_sites):
            c_j = make_fermion_operator_mpo(jsite, num_sites, op_type="annihilation")
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
