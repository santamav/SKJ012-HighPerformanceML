import ctypes
import inspect
import math
import os
import platform
import numpy as np
from ctypes.util import find_library
from importlib import import_module

MATMUL_METHOD = "CBLAS"

# Matmul operation
def matmul(a, b, c=None):
    if MATMUL_METHOD == "NAIVE":
        return matmul_naive(a, b, c)
    elif MATMUL_METHOD == "NUMPY":
        return matmul_numpy(a, b, c)
    elif MATMUL_METHOD == "CBLAS":
        return matmul_cblas(libopenblas(), a, b, c)

RELU_METHOD = "CYTHON"

# ReLU operation
def relu(x):
    if RELU_METHOD == "NUMPY":
        return relu_numpy(x)
    elif RELU_METHOD == "CYTHON":
        from sjk012.relu.relu import relu_cython
        return relu_cython(x)
        

def matmul_naive(a, b, c=None):
    a_rows, a_cols = a.shape
    b_rows, b_cols = b.shape
    
    if a_cols != b_rows:
        raise ValueError("Number of columns in A must be equal to number of rows in B")

    if c is None:
        c = np.zeros((a_rows, b_cols))
        
    ###########################################################################
    # TODO: Implement the matmul operation using only Python code.            #
    # If c is not None you should accumulate the result onto it.              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****      
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return c


def matmul_numpy(a, b, c=None):
    
    ###########################################################################
    # TODO: Implement the matmul operation using only Numpy.                  #
    # If c is not None you should accumulate the result onto it.              #
    # Check the documentation of the numpy library at https://numpy.org/doc/  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return res + c if c is not None else res


def matmul_cblas(lib, a, b, c=None):

    order = 101  # 101 for row-major, 102 for column major data structures
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    
    alpha = 1.0
    if c is None:
        c = np.zeros((m, n), a.dtype, order="C")
        beta = 0.0
    else:
        beta = 1.0
        
    # trans_{a,b} = 111 for no transpose, 112 for transpose, and 113 for conjugate transpose
    if a.flags["C_CONTIGUOUS"]:
        trans_a = 111
        lda = k
    elif a.flags["F_CONTIGUOUS"]:
        trans_a = 112
        lda = m
    else:
        raise ValueError(f"Matrix a data layout not supported.")
    if b.flags["C_CONTIGUOUS"]:
        trans_b = 111
        ldb = n
    elif b.flags["F_CONTIGUOUS"]:
        trans_b = 112
        ldb = k
    else:
        raise ValueError(f"Matrix a data layout not supported.")
    ldc = n

    ###########################################################################
    # TODO: Call to lib.cblas_sgemm function using the ctypes library         #
    # See its interface here:                                                 #
    # https://netlib.org/lapack/explore-html/de/da0/cblas_8h_a1446cddceb275e7cd299157a5d61d5e4.html 
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    lib.cblas_sgemm(ctypes.c_int(order), ctypes.c_int(trans_a), ctypes.c_int(trans_b),
                    ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k),
                    ctypes.c_float(alpha), a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_int(lda),
                    b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_int(ldb),
                    ctypes.c_float(beta), c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_int(ldc))
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return c


def matmul_tiled(lib, a, b, c=None, block_size=32):
    from tiled_gemm import tiled_gemm_cython

    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    if c is None:
        c = np.zeros((m, n), a.dtype, order="C")
    
    ###########################################################################
    # TODO: Call to lib.tiled_gemm function using the ctypes library          #
    #  
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return c
    

def relu_numpy(x):
    
    ###########################################################################
    # TODO: Implement the ReLU operation using only Numpy.                    #
    # The function should return y with the values x > 0 and mask, a            #
    # Check the documentation of the numpy library at https://numpy.org/doc/  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****     
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################    
    return y, mask


def im2col_numpy(x, filter_height, filter_width, padding, stride):
    
    ###########################################################################
    # TODO: Implement the im2col operation using NumPy.                       #
    # You need to reshape the input tensor into a 2D array of shape           #
    # (C * filter_height * filter_width, N * HH * WW), where N is the number  #
    # of samples, C is the number of channels, HH and WW are the height and   # 
    # width of the output after applying the filter, respectively.            #
    # Use zero-padding as specified by the padding parameter.                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################                                
    return cols


def col2im_numpy(cols, input_shape, filter_height, filter_width, padding, stride):
    
    ###########################################################################
    # TODO: Implement the col2im operation using NumPy.
    # Given the columns matrix, reshape it back to the original input tensor shape.
    # You need to reverse the process done in im2col_numpy.
    # Use the input_shape parameter to determine the original shape of the input tensor.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ########################################################################### 
    return x


def load_library(name):
    """
    Loads an external library using ctypes.CDLL.

    It searches the library using ctypes.util.find_library(). If the library is
    not found, it traverses the LD_LIBRARY_PATH until it finds it. If it is not
    in any of the LD_LIBRARY_PATH paths, an ImportError exception is raised.

    Parameters
    ----------
    name : str
        The library name without any prefix like lib, suffix like .so, .dylib or
        version number (this is the form used for the posix linker option -l).

    Returns
    -------
    The loaded library.
    """
    path = None
    full_name = f"lib{name}.%s" % {"Linux": "so", "Darwin": "dylib"}[platform.system()]
    for current_path in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
        if os.path.exists(os.path.join(current_path, full_name)):
            path = os.path.join(current_path, full_name)
            break
            
    if path is None:
        # Didn't find the library
        raise ImportError(f"Library '{name}' could not be found. Please add its path to LD_LIBRARY_PATH.")
        
    return ctypes.CDLL(path)

def libopenblas():
    if not hasattr(libopenblas, "lib"):
        libopenblas.lib = load_library("openblas")
    return libopenblas.lib

def libblis():
    if not hasattr(libblis, "lib"):
        libblis.lib = load_library("blis")
    return libblis.lib

def libtiledgemm():
    #if not hasattr(tiledgemm, "lib"):
    libtiledgemm.lib = load_library("tiledgemm")
    return libtiledgemm.lib

def librelu():
    #if not hasattr(relu, "lib"):
    librelu.lib = load_library("relu")
    return librelu.lib