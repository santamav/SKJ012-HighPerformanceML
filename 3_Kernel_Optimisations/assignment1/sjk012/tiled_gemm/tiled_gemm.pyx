import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

def matmul_tiled_cython(a, b, c, int block_size):
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    matmul_tiled_cython_inner(a, b, c, m, n, k, block_size)
    return c

@cython.boundscheck(False)
@cython.wraparound(False)
cdef matmul_tiled_cython_inner(np.ndarray[np.float32_t, ndim=2] a,
                               np.ndarray[np.float32_t, ndim=2] b,
                               np.ndarray[np.float32_t, ndim=2] c,
                               int m, int n, int k,
                               int block_size):
    
    cdef int m_, n_, k_, nc_, kc_, i, j, l, m_upper, n_upper, k_upper
    cdef float temp

    ###########################################################################
    # TODO: Implement the tiled matmul operation using Cython code.           #
    # Parallelize the loop via prange.                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
   
    for nn in prange(N, nogil=True):
        for cc in range(C):
            for ii in range(filter_height):
                for jj in range(filter_width):
                    row = cc * filter_height * filter_width + ii * filter_width + jj
                    for yy in range(HH):
                        for xx in range(WW):
                            col = nn * HH * WW + yy * WW + xx
                            x_x = xx * stride - padding + jj
                            x_y = yy * stride - padding + ii
                            if x_x >= 0 and x_x < W and x_y >= 0 and x_y < H:
                                cols[row, col] = x[nn, cc, x_y, x_x]
                            else:
                                cols[row, col] = 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
                
                        