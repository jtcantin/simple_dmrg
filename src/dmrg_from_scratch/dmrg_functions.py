from typing import List

import numpy as np
import scipy.sparse.linalg

eigenvalue_imaginary_threshold = 1e-13


def drmg_main(
    mpo: List[np.ndarray],
    num_sites: int = 4,
    physical_dimension: int = 2,
    bond_dimension: int = 3,
    seed: int = 0,
    num_sweeps: int = 4,  # Here, one sweep is a forward and backward pass
    verbosity: int = 1,
) -> List[np.ndarray]:
    """Main function of the DMRG algorithm.
    Structure follows the pseudocode in Fig. 19 of DOI:  https://doi.org/10.1140/epjb/s10051-023-00575-2
    verbosity: 0: no output, 1: simple output, 2: more output
    """
    ###############
    # Initialization
    ###############

    # Build random MPS
    ###############
    mps_ket = build_random_mps(
        num_sites=num_sites,
        physical_dimension=physical_dimension,
        bond_dimension=bond_dimension,
        seed=seed,
    )
    if verbosity > 0:
        print("Random MPS generated (non-normalized).")

    # Transform MPS to right-canonical form
    ###############
    mps_ket = right_normalize_mps(mps_ket)

    if verbosity > 0:
        print("MPS transformed to right-canonical form.")

    # Set L[1] = R[1] = 1
    ###############
    if verbosity > 0:
        print("Calculating initial L and R tensors...")
    L_tensor_list = [np.eye(1)]
    L_tensor_list[0] = np.expand_dims(L_tensor_list[0], axis=-1)
    R_tensor_list = [np.eye(1)]
    R_tensor_list[-1] = np.expand_dims(R_tensor_list[-1], axis=-1)

    # print("L[0] shape: ", L_tensor_list[0].shape)
    # print("R[-1] shape: ", R_tensor_list[-1].shape)
    # print("L[0]", L_tensor_list[0])
    # print("R[-1]", R_tensor_list[-1])

    # Calculate remaining R tensors using extended zipper method
    # See Appendix B and 4.4 of https://doi.org/10.1140/epjb/s10051-023-00575-2
    ###############
    for iiter in range(num_sites - 2, -1, -1):
        ket_tensor = mps_ket[iiter + 1]
        mpo_tensor = mpo[iiter + 1]
        bra_tensor = np.conj(mps_ket[iiter + 1])
        R_tensor = R_tensor_list[0].copy()

        if verbosity > 1:
            print("Site ", iiter)
            print("ket tensor shape: ", ket_tensor.shape)
            print("bra tensor shape: ", bra_tensor.shape)
            print("mpo tensor shape: ", mpo_tensor.shape)
            print("R tensor shape: ", R_tensor.shape)

        # Contract ket tensor with R tensor
        # R_tensor: c (top) e (left) h (bottom)
        # ket_tensor: a (left) b (bottom) c (right)
        # R_ket_tensor: a (left) b (bottom left) e (bottom right) h (right)
        R_ket_tensor = np.einsum("ceh,abc->abeh", R_tensor, ket_tensor)
        if verbosity > 1:
            print("R_ket_tensor shape: ", R_ket_tensor.shape)

        # Contract with mpo tensor
        # mpo_tensor: d (left) b (top) f (bottom) e (right)
        # R_ket_tensor: a (left) b (bottom left) e (bottom right) h (right)
        # R_ket_o_tensor: a (top) d (left) f (bottom left) h (bottom right)
        R_ket_o_tensor = np.einsum("abeh,dbfe->adfh", R_ket_tensor, mpo_tensor)
        if verbosity > 1:
            print("R_ket_o_tensor shape: ", R_ket_o_tensor.shape)

        # Contract with bra tensor
        # R_ket_o_tensor: a (top) d (left) f (bottom left) h (bottom right)
        # bra_tensor: g (left) f (top) h (right)
        # R_tensor: a (top) d (left) g (bottom)
        R_tensor = np.einsum("adfh,gfh->adg", R_ket_o_tensor, bra_tensor)

        if verbosity > 1:
            print("New R tensor shape: ", R_tensor.shape)

        # Prepend to R_tensor_list
        R_tensor_list = [R_tensor] + R_tensor_list

    # print("R_tensor_list length: ", len(R_tensor_list))
    if verbosity > 0:
        print("Initial L and R tensors calculated.")

    ###############
    # Sweeps
    ###############
    all_eigenvalues_real = True
    for isweep in range(num_sweeps):
        # Sweep left to right
        ###############
        if verbosity > 0:
            print("---------------------")
        for isite in range(num_sites):
            # Local optimization of mps_ket[isite] by diagonalizing the effective Matrix
            if verbosity > 1:
                print("---------------------")
            if verbosity > 0:
                print("Sweep ", isweep, ", left-to-right", ", site ", isite)
            # Calculate effective matrix
            L_local_tensor = L_tensor_list[isite]
            R_local_tensor = R_tensor_list[isite]
            mpo_local_tensor = mpo[isite]

            if verbosity > 1:
                print("L_local_tensor shape: ", L_local_tensor.shape)
                print("R_local_tensor shape: ", R_local_tensor.shape)
                print("mpo_local_tensor shape: ", mpo_local_tensor.shape)

            mps_new_tensor, all_eigenvalues_real = get_new_ket_tensor(
                L_local_tensor=L_local_tensor,
                mpo_local_tensor=mpo_local_tensor,
                R_local_tensor=R_local_tensor,
                original_ket_tensor=mps_ket[isite],
                verbosity=verbosity,
                all_eigenvalues_real=all_eigenvalues_real,
            )

            # Update mps_ket[isite]
            mps_ket[isite] = mps_new_tensor

            # Prepare for next iteration
            # Put MPS into mixed-canonical form by putting the MPS tensor into left-canonical form
            u, s, vh = mps_tensor_svd_left_normalized(mps_ket[isite])
            if verbosity > 1:
                print("u shape: ", u.shape)
                print("s shape: ", s.shape)
                print("vh shape: ", vh.shape)

            # Update mps_ket[isite]
            mps_ket[isite] = u  # u already has the correct shape
            if verbosity > 1:
                print("mps_ket[isite]: ", mps_ket[isite].shape)

            if isite < num_sites - 1:
                # Only keep s and vh when not at the last site as they are norm and global phase, respectively
                mps_ket[isite + 1] = np.einsum(
                    "a,ab,bdg->adg", s, vh, mps_ket[isite + 1]
                )
                if verbosity > 1:
                    print(
                        "Updated mps_ket[isite + 1], shape: ", mps_ket[isite + 1].shape
                    )

                # Calculate new L tensor by contraction of current L tensor with new ket tensor,
                # current mpo tensor, and new bra tensor
                ket_tensor = mps_ket[isite]
                bra_tensor = np.conj(mps_ket[isite])

                # L_local_tensor:a (top) d (right) g (bottom)
                # ket_tensor: a (left) b (bottom) c (right)
                # L_ket_tensor: g (left) d (bottom left) b (bottom right) c (right)
                L_ket_tensor = np.einsum("adg,abc->gdbc", L_local_tensor, ket_tensor)

                # L_ket_tensor: g (left) d (bottom left) b (bottom right) c (right)
                # mpo_local_tensor: d (left) b (top) e (bottom) f (right)
                # L_ket_o_tensor: f (right) c (top) g (bottom left) e (bottom right)
                L_ket_o_tensor = np.einsum(
                    "gdbc,dbef->fcge", L_ket_tensor, mpo_local_tensor
                )

                # L_ket_o_tensor: f (right) c (top) g (bottom left) e (bottom right)
                # bra_tensor: g (left) e (top) h (right)
                # L_tensor: c (top) f (right) h (bottom)
                L_tensor = np.einsum("fcge,geh->cfh", L_ket_o_tensor, bra_tensor)

                if len(L_tensor_list) == isite + 1:
                    L_tensor_list.append(L_tensor)
                else:
                    L_tensor_list[isite + 1] = L_tensor

                if verbosity > 1:
                    print("Updated L_tensor_list, length: ", len(L_tensor_list))
                    print(
                        "Updated L_tensor_list, shape: ", L_tensor_list[isite + 1].shape
                    )

        if verbosity > 0:
            print("---------------------")

        # Sweep right to left
        ###############
        for isite in range(num_sites - 1, -1, -1):
            # Local optimization of mps_ket[isite] by diagonalizing the effective Matrix
            if verbosity > 1:
                print("---------------------")
            if verbosity > 0:
                print("Sweep ", isweep, ", right-to-left", ", site ", isite)
            # Calculate effective matrix
            L_local_tensor = L_tensor_list[isite]
            R_local_tensor = R_tensor_list[isite]
            mpo_local_tensor = mpo[isite]

            if verbosity > 1:
                print("L_local_tensor shape: ", L_local_tensor.shape)
                print("R_local_tensor shape: ", R_local_tensor.shape)
                print("mpo_local_tensor shape: ", mpo_local_tensor.shape)

            mps_new_tensor, all_eigenvalues_real = get_new_ket_tensor(
                L_local_tensor=L_local_tensor,
                mpo_local_tensor=mpo_local_tensor,
                R_local_tensor=R_local_tensor,
                original_ket_tensor=mps_ket[isite],
                verbosity=verbosity,
                all_eigenvalues_real=all_eigenvalues_real,
            )

            # Update mps_ket[isite]
            mps_ket[isite] = mps_new_tensor

            # Prepare for next iteration
            # Put MPS into mixed-canonical form by putting the MPS tensor into right-canonical form
            u, s, vh = mps_tensor_svd_right_normalized(mps_ket[isite])
            if verbosity > 1:
                print("u shape: ", u.shape)
                print("s shape: ", s.shape)
                print("vh shape: ", vh.shape)

            # Update mps_ket[isite]
            mps_ket[isite] = vh  # vh already has the correct shape

            if isite > 0:
                # Only keep s and u when not at the first site as they are norm and global phase, respectively
                mps_ket[isite - 1] = np.einsum(
                    "bdg,ga,a->bda", mps_ket[isite - 1], u, s
                )

                if verbosity > 1:
                    print(
                        "Updated mps_ket[isite - 1], shape: ", mps_ket[isite - 1].shape
                    )

                # Calculate new R tensor by contraction of current R tensor with new ket tensor,
                # current mpo tensor, and new bra tensor
                ket_tensor = mps_ket[isite]
                bra_tensor = np.conj(mps_ket[isite])

                # R_local_tensor: c (top) e (left) g (bottom)
                # ket_tensor: a (left) b (bottom) c (right)
                # R_ket_tensor: a (left) b (bottom left) e (bottom right) g (right)
                R_ket_tensor = np.einsum("ceg,abc->abeg", R_local_tensor, ket_tensor)

                # R_ket_tensor: a (left) b (bottom left) e (bottom right) g (right)
                # mpo_local_tensor: d (left) b (top) f (bottom) e (right)
                # R_ket_o_tensor: a (top) d (left) f (bottom left) g (bottom right)
                R_ket_o_tensor = np.einsum(
                    "abeg,dbfe->adfg", R_ket_tensor, mpo_local_tensor
                )

                # R_ket_o_tensor: a (top) d (left) f (bottom left) g (bottom right)
                # bra_tensor: h (left) f (top) g (right)
                # R_tensor: a (top) d (left) h (bottom)
                R_tensor = np.einsum("adfg,hfg->adh", R_ket_o_tensor, bra_tensor)

                R_tensor_list[isite - 1] = R_tensor

                if verbosity > 1:
                    print("Updated R_tensor_list, length: ", len(R_tensor_list))
                    print(
                        "Updated R_tensor_list, shape: ", R_tensor_list[isite - 1].shape
                    )

    if verbosity > 0:
        print("---------------------")
        print("Sweeps complete.")
        # print("Final MPS shape:",)
        print("ALL EIGENVALUES REAL: ", all_eigenvalues_real)

    if not all_eigenvalues_real:
        print(
            f"WARNING: Not all eigenvalues of the effective matrix were real, with an absolute threshold of {eigenvalue_imaginary_threshold}. This may indicate a problem."
        )

    return mps_ket


def get_new_ket_tensor(
    L_local_tensor: np.ndarray,
    mpo_local_tensor: np.ndarray,
    R_local_tensor: np.ndarray,
    original_ket_tensor: np.ndarray,
    verbosity: int = 0,
    all_eigenvalues_real: bool = True,
) -> np.ndarray:
    """Calculates the new ket tensor by diagonalizing the effective matrix
    See Appendix B and Fig.15 of https://doi.org/10.1140/epjb/s10051-023-00575-2
    """
    # L_local_tensor: a (top) c (right) b (bottom)
    # mpo_local_tensor: c (left) d (top) e (bottom) f (right)
    # R_local_tensor: g (top) f (left) h (bottom)
    # effective_matrix: a (top left) d (top middle) g (top right) b (bottom left) e (bottom middle) h (bottom right)
    effective_matrix = np.einsum(
        "acb,cdef,gfh->adgbeh", L_local_tensor, mpo_local_tensor, R_local_tensor
    )

    effective_matrix_orig_shape = effective_matrix.shape
    if verbosity > 1:
        print("Effective matrix shape: ", effective_matrix.shape)

    # print(effective_matrix)
    # Fuse indices
    # M_a_d_g_b_e_h = M_adg_beh = M_ij
    effective_matrix = np.reshape(
        effective_matrix,
        (
            effective_matrix.shape[0]
            * effective_matrix.shape[1]
            * effective_matrix.shape[2],
            effective_matrix.shape[3]
            * effective_matrix.shape[4]
            * effective_matrix.shape[5],
        ),
        order="C",
    )

    if verbosity > 1:
        print("Calculated effective matrix.")
        print("Effective matrix shape: ", effective_matrix.shape)
        # Check that the effective matrix is Hermitian
        print(
            "Effective matrix Hermitian: ",
            np.allclose(effective_matrix, effective_matrix.conj().T),
        )
        print(
            "Max abs difference: ",
            np.max(np.abs(effective_matrix - effective_matrix.conj().T)),
        )

    # Get initial guess for eigenvector
    # eigenvector: i <- a (left) d (bottom) g (right)
    eigenvector_guess = np.reshape(
        original_ket_tensor,
        (-1),
        order="C",
    )
    if verbosity > 1:
        print("Eigenvector guess shape: ", eigenvector_guess.shape)
        print("original_ket_tensor shape: ", original_ket_tensor.shape)

    # Diagonalize effective matrix
    eigenvalue, eigenvector = scipy.sparse.linalg.eigs(
        A=effective_matrix,
        k=1,  # Number of eigenvalues and eigenvectors to compute
        M=None,
        sigma=None,
        which="SR",  # Smallest real part
        v0=eigenvector_guess,
        ncv=None,
        maxiter=None,
        tol=0,  # Tolerance for convergence, use machine precision
        return_eigenvectors=True,
        Minv=None,
        OPinv=None,
        OPpart=None,
    )
    eigenvalue = eigenvalue[0]
    eigenvector = eigenvector[:, 0]
    if verbosity > 1:
        print("Obtained the ground state of the effective matrix.")
        print("Eigenvalue: ", eigenvalue)
        print("Eigenvector shape: ", eigenvector.shape)
        print("Eigenvector: ", eigenvector)

    # Check that the eigenvalue is real
    if all_eigenvalues_real:
        all_eigenvalues_real = np.abs(eigenvalue.imag) <= eigenvalue_imaginary_threshold

    # Reshape eigenvector
    # eigenvector: i -> a (left) d (bottom) g (right)
    mps_new_tensor = np.reshape(
        eigenvector,
        (
            effective_matrix_orig_shape[0],
            effective_matrix_orig_shape[1],
            effective_matrix_orig_shape[2],
        ),
        order="C",
    ).copy()

    if verbosity > 1:
        print("New MPS tensor shape: ", mps_new_tensor.shape)
        # print("New MPS tensor: ", mps_new_tensor)

    return mps_new_tensor, all_eigenvalues_real


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

    raise NotImplementedError("This does not produce a Hermitian MPO.")

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
