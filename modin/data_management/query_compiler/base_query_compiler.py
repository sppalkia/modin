from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class BaseQueryCompiler(object):

    def __init__(self):
        raise NotImplementedError("Must be implemented in child class")

    def __constructor__(self, *init_args):
        """By default, constructor method will invoke an init"""
        return type(self)(*init_args)


