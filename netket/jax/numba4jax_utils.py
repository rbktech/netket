# code to find the path of a loaded dll
#  doese not work on windows

from numba import cuda as ncuda

import ctypes
import ctypes.util

libdl = ctypes.CDLL(ctypes.util.find_library("dl"))


class Dl_info(ctypes.Structure):
    _fields_ = (
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    )


libdl.dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(Dl_info))


def find_path_of_symbol_in_library(symbol):
    result = libdl.dladdr(symbol, ctypes.byref(info))

    if result and result.dli_fname:
        return result.dli_fname.decode(sys.getfilesystemencoding())
    else:
        raise ValueError("Cannot determine path of Library.")
        libdl_path = "Not Found"
