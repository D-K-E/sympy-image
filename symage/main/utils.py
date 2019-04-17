# Utility functions 
# Author: Kaan Eraslan
# Licensing: see, LICENSE
# No warranties, see LICENSE

import numpy as np
from PIL import Image


def convertPilImg2Array(image):
    "Convert the pil image to numpy array"
    imarr = np.array(image, dtype=np.uint8)
    if len(imarr.shape) > 2:
        imarr = imarr[:, :, :3]
        imarr = np.transpose(imarr,
                             (1, 0, 2)  # new axis order
                             )
    else:
        imarr = imarr.T
    #
    return imarr


def convertNpImg2BytesPng(image: np.ndarray):
    "Converts the numpy matrix into png bytes"
    img = Image.fromarray(image)
    imio = io.BytesIO()
    img.save(imio, format='PNG')
    return imio


def stripExt(str1: str, ext_delimiter='.') -> [str, str]:
    "Strip extension"
    strsplit = str1.split(ext_delimiter)
    ext = strsplit.pop()
    newstr = ext_delimiter.join(strsplit)
    return (newstr, ext)


def strip2end(str1: str, fromStr: str):
    "Strip str1 from fromStr until its end"
    assert fromStr in str1
    pos = str1.find(fromStr)
    newstr = str1[:pos]
    return newstr


def strip2ends(str1: str, fromStrs: [str]):
    "Strip a list of string from str1 end"
    newstr = ""
    for fromStr in fromStrs:
        if fromStr in str1:
            newstr = strip2end(str1, fromStr)
        else:
            continue
    return newstr


def stripPrefix(str1: str, prefix: str):
    "Strip prefix from str1"
    assert prefix in str1
    assert str1.startswith(prefix)
    preflen = len(prefix)
    newstr = str1[preflen:]
    return newstr


def stripStr(str1: str,
             prefix: str,
             fromStr: str):
    "Strip string"
    newstr = stripPrefix(str1, prefix)
    return strip2end(newstr, fromStr)


def stripStrs(str1: str,
              prefix: str,
              fromStrs: [str]):
    "Strip string applied to different fromstrings"
    newstr = ""
    for fromStr in fromStrs:
        if fromStr in str1:
            newstr = stripStr(str1, prefix, fromStr)
        else:
            continue
    return newstr


def assertCond(var, cond: bool, printType=True):
    "Assert condition print message"
    if printType:
        assert cond, 'variable value: {0}\nits type: {1}'.format(var,
                                                                 type(var))
    else:
        assert cond, 'variable value: {0}'.format(var)


def setParams2Dict(paramDict: dict,
                   keyNameTree: [str],
                   val):
    "Travers the parameter dict and set a value to its last key name"
    # creates the key if it does not exist
    assertCond(keyNameTree, isinstance(keyNameTree, list))
    var = [(isinstance(k, str), k, type(k)) for k in keyNameTree]
    assertCond(var, all([isinstance(k, str) for k in keyNameTree]),
               printType=False)

    lastindex = len(keyNameTree) - 1
    for i, key in enumerate(keyNameTree):
        if i == lastindex:
            paramDict[key] = val
        else:
            if key not in paramDict:
                paramDict[key] = {}
            paramDict = paramDict[key]

