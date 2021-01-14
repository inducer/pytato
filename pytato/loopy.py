import numpy as np
import loopy as lp
from loopy.types import NumpyType
from typing import Dict, FrozenSet, Optional
from pytato.array import (DictOfNamedArrays, Namespace, Array, ShapeType, NamedArray)
from pytools.tag import TagsType


class LoopyFunction(DictOfNamedArrays):
    """
    Call to a :mod:`loopy` program.
    """
    _mapper_method = "map_loopy_function"

    def __init__(self,
            namespace: Namespace,
            program: lp.Program,
            bindings: Dict[str, Array],
            entrypoint: str):
        super().__init__({})

        if any([arg.is_input and arg.is_output
                for arg in program[entrypoint].args]):
            # Pytato DAG cannot have stateful nodes.
            raise ValueError("Cannot have a kernel that writes to inputs.")

        # {{{ infer types of the program

        program = lp.add_and_infer_dtypes(program, {name: ary.dtype
            for name, ary in bindings.items()})

        # }}}

        # {{{ infer shapes of the program

        entry_kernel = program[entrypoint]

        def _add_shapes_to_args(arg):
            if arg.name in bindings:
                return arg.copy(shape=bindings[arg.name].shape, order="C",
                                dim_tags=lp.auto)
            else:
                return arg

        entry_kernel = entry_kernel.copy(args=[_add_shapes_to_args(arg)
                                               for arg in entry_kernel.args])

        program = program.with_kernel(entry_kernel)

        program = lp.infer_arg_descr(program)

        # }}}

        self.program = program
        self.bindings = bindings
        self.entrypoint = entrypoint
        self._namespace = namespace

        self._named_arrays = {name: LoopyFunctionResult(self, name)
                              for name, lp_arg in program[entry_kernel].arg_dict
                              if lp_arg.is_output}

    @property
    def namespace(self) -> Namespace:
        return self._namespace

    @property
    def entry_kernel(self) -> lp.LoopKernel:
        return self.program[self.entrypoint]

    def __hash__(self):
        from loopy.tools import LoopyKeyBuilder as lkb
        return hash((lkb()(self.program), tuple(self.bindings.items()),
            self.results))

    def __eq__(self, other):
        if self is other:
            return True

        if not isinstance(other, LoopyFunction):
            return False

        if ((self.entrypoint == other.entrypoint)
             and (self.bindings == other.bindings)
             and (self.program == other.program)):
            return True
        return False


class LoopyFunctionResult(NamedArray):
    """
    """
    _mapper_method = "map_loopy_function_result"

    def expr(self) -> Array:
        raise ValueError("Expressions for results of loopy functions aren't defined")

    @property
    def shape(self) -> ShapeType:
        loopy_arg = self.loopy_function.entry_kernel.arg_dict[self.name]
        return loopy_arg.shape

    @property
    def dtype(self) -> np.dtype:
        loopy_arg = self.loopy_function.entry_kernel.arg_dict[self.name]
        dtype = loopy_arg.dtype

        if isinstance(dtype, np.dtype):
            return dtype
        elif isinstance(dtype, NumpyType):
            return dtype.numpy_dtype
        else:
            raise NotImplementedError(f"Unknown dtype type '{dtype}'")


def call_loopy(namespace: Namespace, program: lp.Program, bindings: dict,
        entrypoint: Optional[str] = None):
    """
    Operates a general :class:`loopy.Program` on the array inputs as specified
    by *bindings*.

    Restrictions on the structure of ``program[entrypoint]``:

    * array arguments of ``program[entrypoint]`` should either be either
      input-only or output-only.
    * all input-only arguments of ``program[entrypoint]`` must appear in
      *bindings*.
    * all output-only arguments of ``program[entrypoint]`` must appear in
      *bindings*.
    * if *program* has been declared with multiple entrypoints, *entrypoint*
      can not be *None*.

    :arg bindings: mapping from argument names of ``program[entrypoint]`` to
        :class:`pytato.array.Array`.
    :arg results: names of ``program[entrypoint]`` argument names that have to
        be returned from the call.
    """
    # FIXME: Sanity checks
    if entrypoint is None:
        if len(program.entrypoints) != 1:
            raise ValueError("cannot infer entrypoint")

        entrypoint, = program.entrypoints

    return LoopyFunction(namespace, program, bindings, entrypoint)


# vim: fdm=marker
