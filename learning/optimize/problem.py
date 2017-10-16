###############################################################################
# The MIT License (MIT)
#
# Copyright (c) 2017 Justin Lovinger
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################
"""Problem constructors to use with Optimizer."""

import functools
import operator


############################
# Problem
############################
class Problem(object):
    """Problem instance for optimizer.

    Functions are prioritized as follows:
        value(s): first priority, second_priority, (first_value_using_best_func, second_func)

        obj: obj_func, obj_jac_func, obj_hess_func, obj_jac_hess_func
        jac: jac_func, obj_jac_func, jac_hess_func, obj_jac_hess_func
        hess: hess_func, obj_hess_func, jac_hess_func, obj_jac_hess_func

        obj_jac: obj_jac_func, obj_jac_hess_func, (obj, jac)
        obj_hess: obj_hess_func, obj_jac_hess_func, (obj, hess)
        jac_hess: jac_hess_func, obj_jac_hess_func, (jac, hess)

        obj_jac_hess: obj_jac_hess_func, (obj_jac_func, hess), (obj_hess_func, jac),
            (obj, jac_hess_func), (obj, jac, hess)
    """

    def __init__(self,
                 obj_func=None,
                 jac_func=None,
                 hess_func=None,
                 obj_jac_func=None,
                 obj_hess_func=None,
                 jac_hess_func=None,
                 obj_jac_hess_func=None):
        # Get objective function
        if obj_func is not None:
            self.get_obj = obj_func
        elif obj_jac_func is not None:
            self.get_obj = functools.partial(_call_return_index, obj_jac_func,
                                             0)
        elif obj_hess_func is not None:
            self.get_obj = functools.partial(_call_return_index, obj_hess_func,
                                             0)
        elif obj_jac_hess_func is not None:
            self.get_obj = functools.partial(_call_return_index,
                                             obj_jac_hess_func, 0)
        else:
            self.get_obj = _return_none

        # Get jacobian function
        if jac_func is not None:
            self.get_jac = jac_func
        elif obj_jac_func is not None:
            self.get_jac = functools.partial(_call_return_index, obj_jac_func,
                                             1)
        elif jac_hess_func is not None:
            self.get_jac = functools.partial(_call_return_index, jac_hess_func,
                                             0)
        elif obj_jac_hess_func is not None:
            self.get_jac = functools.partial(_call_return_index,
                                             obj_jac_hess_func, 1)
        else:
            self.get_jac = _return_none

        # Get hessian function
        if hess_func is not None:
            self.get_hess = hess_func
        elif obj_hess_func is not None:
            self.get_hess = functools.partial(_call_return_index,
                                              obj_hess_func, 1)
        elif jac_hess_func is not None:
            self.get_hess = functools.partial(_call_return_index,
                                              jac_hess_func, 1)
        elif obj_jac_hess_func is not None:
            self.get_hess = functools.partial(_call_return_index,
                                              obj_jac_hess_func, 2)
        else:
            self.get_hess = _return_none

        # Get objective and jacobian function
        if obj_jac_func is not None:
            self.get_obj_jac = obj_jac_func
        elif obj_jac_hess_func is not None:
            self.get_obj_jac = functools.partial(_call_return_indices,
                                                 obj_jac_hess_func, (0, 1))
        else:
            self.get_obj_jac = functools.partial(_bundle, (self.get_obj,
                                                           self.get_jac))

        # Get objective and hessian function
        if obj_hess_func is not None:
            self.get_obj_hess = obj_hess_func
        elif obj_jac_hess_func is not None:
            self.get_obj_hess = functools.partial(_call_return_indices,
                                                  obj_jac_hess_func, (0, 2))
        else:
            self.get_obj_hess = functools.partial(_bundle, (self.get_obj,
                                                            self.get_hess))

        # Get jacobian and hessian function
        if jac_hess_func is not None:
            self.get_jac_hess = jac_hess_func
        elif obj_jac_hess_func is not None:
            self.get_jac_hess = functools.partial(_call_return_indices,
                                                  obj_jac_hess_func, (1, 2))
        else:
            self.get_jac_hess = functools.partial(_bundle, (self.get_jac,
                                                            self.get_hess))

        # Get objective, jacobian, hessian function
        if obj_jac_hess_func is not None:
            self.get_obj_jac_hess = obj_jac_hess_func
        elif obj_jac_func is not None:
            self.get_obj_jac_hess = functools.partial(
                _bundle_add, (obj_jac_func, functools.partial(
                    _tuple_result, self.get_hess)))
        elif obj_hess_func is not None:
            self.get_obj_jac_hess = functools.partial(
                _bundle_add_split, obj_hess_func, self.get_jac)
        elif jac_hess_func is not None:
            self.get_obj_jac_hess = functools.partial(
                _bundle_add, (functools.partial(_tuple_result, self.get_obj),
                              jac_hess_func))
        else:
            self.get_obj_jac_hess = functools.partial(
                _bundle, (self.get_obj, self.get_jac, self.get_hess))


def _call_return_indices(func, indices, *args, **kwargs):
    """Return indices of func called with *args and **kwargs.

    Use with functools.partial to wrap func.
    """
    values = func(*args, **kwargs)
    return [values[i] for i in indices]


def _call_return_index(func, index, *args, **kwargs):
    """Return index of func called with *args and **kwargs.

    Use with functools.partial to wrap func.
    """
    return func(*args, **kwargs)[index]


def _bundle_add(functions, *args, **kwargs):
    """Return results of functions, concatenated together, called with *args and **kwargs."""
    return reduce(operator.add, _bundle(functions, *args, **kwargs))


def _bundle(functions, *args, **kwargs):
    """Return result of each function, called with *args and **kwargs."""
    return [func(*args, **kwargs) for func in functions]


def _bundle_add_split(f_1, f_2, *args, **kwargs):
    """Return f_1[0], f_2, f_1[1], called with *args, **kwargs."""
    f_1_values = f_1(*args, **kwargs)
    return f_1_values[0], f_2(*args, **kwargs), f_1_values[1]


def _tuple_result(func, *args, **kwargs):
    return func(*args, **kwargs),  # , makes tuple


def _return_none(*args, **kwargs):
    """Return None."""
    return None
