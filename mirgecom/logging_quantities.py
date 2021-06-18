"""Support for time series logging."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

__doc__ = """
.. autoclass:: StateConsumer
.. autoclass:: DiscretizationBasedQuantity
.. autoclass:: LogCFL
.. autoclass:: KernelProfile
.. autoclass:: PythonMemoryUsage
.. autoclass:: DeviceMemoryUsage
.. autofunction:: initialize_logmgr
.. autofunction:: logmgr_add_cl_device_info
.. autofunction:: logmgr_add_device_memory_usage
.. autofunction:: logmgr_add_many_discretization_quantities
.. autofunction:: add_package_versions
.. autofunction:: set_sim_state
.. autofunction:: logmgr_set_time
"""

from typing import Any
from logpyle import (DtConsumer, LogQuantity, LogManager, MultiLogQuantity,
    add_run_info, add_general_quantities, add_simulation_quantities)
from meshmode.array_context import PyOpenCLArrayContext
from meshmode.discretization import Discretization
import pyopencl as cl

from typing import Optional, Callable
import numpy as np


def initialize_logmgr(enable_logmgr: bool,
                      filename: str = None, mode: str = "wu",
                      mpi_comm=None) -> LogManager:
    """Create and initialize a mirgecom-specific :class:`logpyle.LogManager`."""
    if not enable_logmgr:
        return None

    logmgr = LogManager(filename=filename, mode=mode, mpi_comm=mpi_comm)

    add_run_info(logmgr)
    add_package_versions(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)

    try:
        logmgr.add_quantity(PythonMemoryUsage())
    except ImportError:
        from warnings import warn
        warn("psutil module not found, not tracking memory consumption."
             "Install it with 'pip install psutil'")

    return logmgr


def logmgr_add_cl_device_info(logmgr: LogManager, queue: cl.CommandQueue):
    """Add information about the OpenCL device to the log."""
    dev = queue.device
    logmgr.set_constant("cl_device_name", str(dev))
    logmgr.set_constant("cl_device_version", dev.version)
    logmgr.set_constant("cl_platform_version", dev.platform.version)


def logmgr_add_device_name(logmgr: LogManager, queue: cl.CommandQueue):  # noqa: D401
    """Deprecated. Do not use in new code."""
    from warnings import warn
    warn("logmgr_add_device_name is deprecated and will disappear in Q3 2021. "
         "Use logmgr_add_cl_device_info instead.", DeprecationWarning,
         stacklevel=2)
    logmgr_add_cl_device_info(logmgr, queue)


def logmgr_add_device_memory_usage(logmgr: LogManager, queue: cl.CommandQueue):
    """Add the OpenCL device memory usage to the log."""
    if not (queue.device.type & cl.device_type.GPU):
        return
    logmgr.add_quantity(DeviceMemoryUsage())


def logmgr_add_many_discretization_quantities(logmgr: LogManager, discr, dim,
      extract_vars_for_logging, units_for_logging):
    """Add default discretization quantities to the logmgr."""
    for op in ["min", "max", "L2_norm"]:
        for quantity in ["pressure", "temperature"]:
            logmgr.add_quantity(DiscretizationBasedQuantity(
                discr, quantity, op, extract_vars_for_logging, units_for_logging))

        for quantity in ["mass", "energy"]:
            logmgr.add_quantity(DiscretizationBasedQuantity(
                discr, quantity, op, extract_vars_for_logging, units_for_logging))

        for d in range(dim):
            logmgr.add_quantity(DiscretizationBasedQuantity(
                discr, "momentum", op, extract_vars_for_logging, units_for_logging,
                axis=d))


# {{{ Package versions

def add_package_versions(mgr: LogManager, path_to_version_sh: str = None) -> None:
    """Add the output of the emirge version.sh script to the log.

    Parameters
    ----------
    mgr
        The :class:`logpyle.LogManager` to add the versions to.

    path_to_version_sh
        Path to emirge's version.sh script. The function will attempt to find this
        script automatically if this argument is not specified.

    """
    import subprocess
    from warnings import warn

    output = None

    # Find emirge's version.sh in any parent directory
    if path_to_version_sh is None:
        import pathlib
        import mirgecom

        p = pathlib.Path(mirgecom.__file__).resolve()

        for d in p.parents:
            candidate = pathlib.Path(d).joinpath("version.sh")
            if candidate.is_file():
                with open(candidate) as f:
                    if "emirge" in f.read():
                        path_to_version_sh = str(candidate)
                        break

    if path_to_version_sh is None:
        warn("Could not find emirge's version.sh.")

    else:
        try:
            output = subprocess.check_output(path_to_version_sh)
        except OSError as e:
            warn("Could not record emirge's package versions: " + str(e))

    mgr.set_constant("emirge_package_versions", output)

# }}}


# {{{ State handling

def set_sim_state(mgr: LogManager, state, eos) -> None:
    """Update the simulation state of all :class:`StateConsumer` of the log manager.

    Parameters
    ----------
    mgr
        The :class:`logpyle.LogManager` whose :class:`StateConsumer` quantities
        will receive *state*.
    """
    state_vars = {}

    for gd_lst in [mgr.before_gather_descriptors,
            mgr.after_gather_descriptors]:
        for gd in gd_lst:
            if isinstance(gd.quantity, StateConsumer):
                esv_func = gd.quantity.extract_state_vars
                if esv_func not in state_vars:
                    if esv_func is not None:
                        state_vars[esv_func] = esv_func(state, eos)
                    else:
                        state_vars[esv_func] = None

                gd.quantity.set_state(state, state_vars[esv_func])


def logmgr_set_time(mgr: LogManager, steps: int, time: float) -> None:
    """Set the (current/initial) time/step count explicitly (e.g., for restart)."""
    from logpyle import TimestepCounter, SimulationTime

    for gd_lst in [mgr.before_gather_descriptors,
            mgr.after_gather_descriptors]:
        for gd in gd_lst:
            if isinstance(gd.quantity, TimestepCounter):
                gd.quantity.steps = steps
            if isinstance(gd.quantity, SimulationTime):
                gd.quantity.t = time


class StateConsumer:
    """Base class for quantities that require a state for logging.

    .. automethod:: __init__
    .. automethod:: set_state
    """

    def __init__(self, extract_vars_for_logging: Optional[Callable]) -> None:
        """Store the function to extract state variables.

        Parameters
        ----------
        extract_vars_for_logging(dim, state, eos)
            Returns a dict(quantity_name: values) of the state vars for a particular
            state.
        """
        self.extract_state_vars = extract_vars_for_logging
        self.state = None
        self.state_vars = None

    def set_state(self, state: np.ndarray, state_vars) -> None:
        """Update the state vector of the object."""
        self.state = state
        self.state_vars = state_vars

# }}}

# {{{ Discretization-based quantities


class DiscretizationBasedQuantity(LogQuantity, StateConsumer):
    """Logging support for physical quantities.

    Possible rank aggregation operations (``op``) are: min, max, L2_norm.
    """

    def __init__(self, discr: Discretization, quantity: str, op: str,
                 extract_vars_for_logging, units_logging, name: str = None,
                 axis: Optional[int] = None):
        unit = units_logging(quantity)

        if name is None:
            name = f"{op}_{quantity}" + (str(axis) if axis is not None else "")

        LogQuantity.__init__(self, name, unit)
        StateConsumer.__init__(self, extract_vars_for_logging)

        self.discr = discr

        self.quantity = quantity
        self.axis = axis

        from functools import partial

        if op == "min":
            self._discr_reduction = partial(self.discr.nodal_min, "vol")
            self.rank_aggr = min
        elif op == "max":
            self._discr_reduction = partial(self.discr.nodal_max, "vol")
            self.rank_aggr = max
        elif op == "L2_norm":
            self._discr_reduction = partial(self.discr.norm, p=2)
            self.rank_aggr = max
        else:
            raise ValueError(f"unknown operation {op}")

    @property
    def default_aggregator(self):
        """Rank aggregator to use."""
        return self.rank_aggr

    def __call__(self):
        """Return the requested quantity."""
        if self.state_vars is None:
            return None

        quantity = self.state_vars[self.quantity]

        if self.axis is not None:  # e.g. momentum
            quantity = quantity[self.axis]

        return self._discr_reduction(quantity)

# }}}


class LogCFL(LogQuantity, StateConsumer, DtConsumer):
    """Logging support for CFL."""

    def __init__(self, discr, eos, name="cfl", dt: float = 0) -> None:
        LogQuantity.__init__(self, name, "1")
        StateConsumer.__init__(self, None)
        DtConsumer.__init__(self, dt)
        self.discr = discr
        self.eos = eos

    def __call__(self) -> float:
        """Return the value of cfl."""
        if self.state is None:
            return None

        from mirgecom.inviscid import get_inviscid_cfl
        return get_inviscid_cfl(self.discr, eos=self.eos, dt=self.dt, cv=self.state)


class LogUserQuantity(LogQuantity):
    """Logging support for arbitrary user quantities."""

    def __init__(self, name="user_quantity", value=None, user_function=None) -> None:
        LogQuantity.__init__(self, name, "1")
        self._quantity_value = value
        self._uf = user_function

    def __call__(self) -> float:
        """Return the actual logged quantity."""
        if self._uf:
            return self._uf(self._quantity_value)
        return self._quantity_value

    def set_quantity_value(self, value: Any) -> None:
        """Set the value of the logged quantity."""
        self._quantity_value = value


# {{{ Kernel profile quantities

class KernelProfile(MultiLogQuantity):
    """Logging support for statistics of the OpenCL kernel profiling (num_calls, \
    time, flops, bytes_accessed, footprint).

    All statistics except num_calls are averages.

    Parameters
    ----------
    actx
        The array context from which to collect statistics. Must have profiling
        enabled in the OpenCL command queue.

    kernel_name
        Name of the kernel to profile.
    """

    def __init__(self, actx: PyOpenCLArrayContext,
                 kernel_name: str) -> None:
        from mirgecom.profiling import PyOpenCLProfilingArrayContext
        assert isinstance(actx, PyOpenCLProfilingArrayContext)

        from dataclasses import fields
        from mirgecom.profiling import MultiCallKernelProfile

        units_default = {"num_calls": "1",  "time": "s", "flops": "GFlops",
                         "bytes_accessed": "GByte", "footprint_bytes": "GByte"}

        names = [f"{kernel_name}_{f.name}" for f in fields(MultiCallKernelProfile)]
        units = [units_default[f.name] for f in fields(MultiCallKernelProfile)]

        super().__init__(names, units)

        self.kernel_name = kernel_name
        self.actx = actx

    def __call__(self) -> list:
        """Return the requested kernel profile quantity."""
        r = self.actx.get_profiling_data_for_kernel(self.kernel_name)
        self.actx.reset_profiling_data_for_kernel(self.kernel_name)

        return [r.num_calls, r.time.mean(), r.flops.mean(), r.bytes_accessed.mean(),
                r.footprint_bytes.mean()]

# }}}


# {{{ Memory profiling

class PythonMemoryUsage(LogQuantity):
    """Logging support for Python memory usage (RSS, host).

    Uses :mod:`psutil` to track memory usage. Virtually no overhead.
    """

    def __init__(self, name: str = None):

        if name is None:
            name = "memory_usage_python"

        super().__init__(name, "MByte", description="Memory usage (RSS, host)")

        import psutil  # pylint: disable=import-error
        self.process = psutil.Process()

    def __call__(self) -> float:
        """Return the memory usage in MByte."""
        return self.process.memory_info()[0] / 1024 / 1024


class DeviceMemoryUsage(LogQuantity):
    """Logging support for GPU memory usage (Nvidia only currently)."""

    def __init__(self, name: str = None) -> None:

        if name is None:
            name = "memory_usage_gpu"

        super().__init__(name, "MByte", description="Memory usage (GPU)")

        import ctypes
        self.free = ctypes.c_size_t()
        self.total = ctypes.c_size_t()

        try:
            # See https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549#gistcomment-3654335  # noqa
            # on why this calls cuMemGetInfo_v2 and not cuMemGetInfo
            libcuda = ctypes.cdll.LoadLibrary("libcuda.so")
            self.mem_func = libcuda.cuMemGetInfo_v2
        except OSError:
            self.mem_func = None

    def __call__(self) -> float:
        """Return the memory usage in MByte."""
        if self.mem_func is None:
            return None

        import ctypes
        ret = self.mem_func(ctypes.byref(self.free), ctypes.byref(self.total))

        if ret != 0:
            from warnings import warn
            warn(f"cudaMemGetInfo failed with error {ret}.")
            return None
        else:
            return (self.total.value - self.free.value) / 1024 / 1024

# }}}
