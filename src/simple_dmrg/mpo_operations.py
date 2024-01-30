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
