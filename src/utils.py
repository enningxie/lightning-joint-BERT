# -*- coding: utf-8 -*-
# Created by xieenning at 2020/10/27
import unicodedata


def is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False
