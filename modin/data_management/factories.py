from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from .. import __execution_engine__ as execution_engine


class BaseFactory(object):

    @classmethod
    def _determine_engine(cls):
        return getattr(sys.modules[__name__], execution_engine)

    @classmethod
    def build_manager(cls):
        return cls._determine_engine.build_manager()

    @classmethod
    def from_pandas(cls):
        return cls._determine_engine.from_pandas()


class PandasBackedRayFactory(BaseFactory):

    @classmethod
    def build_manager(cls):

