# AdaptiveMap is used by Integrator
# TODO: under construction!
""" 
Adaptive map ``y->x(y)`` for multidimensional ``y`` and ``x``.

An :class:`AdaptiveMap` defines a multidimensional map ``y -> x(y)``
from the unit hypercube, with ``0 <= y[d] <= 1``, to an arbitrary
hypercube in ``x`` space. Each direction is mapped independently
with a Jacobian that is tunable (i.e., "adaptive").

The map is specified by a grid in ``x``-space that, by definition,
maps into a uniformly spaced grid in ``y``-space. The nodes of
the grid are specified by ``grid[d, i]`` where d is the
direction (``d=0,1...dim-1``) and ``i`` labels the grid point
(``i=0,1...N``). The mapping for a specific point ``y`` into
``x`` space is::

    y[d] -> x[d] = grid[d, i(y[d])] + inc[d, i(y[d])] * delta(y[d])

where ``i(y)=floor(y*N``), ``delta(y)=y*N - i(y)``, and
``inc[d, i] = grid[d, i+1] - grid[d, i]``. The Jacobian for this map, ::

    dx[d]/dy[d] = inc[d, i(y[d])] * N,

is piece-wise constant and proportional to the ``x``-space grid
spacing. Each increment in the ``x``-space grid maps into an increment of
size ``1/N`` in the corresponding ``y`` space. So regions in
``x`` space where ``inc[d, i]`` is small are stretched out
in ``y`` space, while larger increments are compressed.

The ``x`` grid for an :class:`AdaptiveMap` can be specified explicitly
when the map is created: for example, ::

    m = AdaptiveMap([[0, 0.1, 1], [-1, 0, 1]])

creates a two-dimensional map where the ``x[0]`` interval ``(0,0.1)``
and ``(0.1,1)`` map into the ``y[0]`` intervals ``(0,0.5)`` and
``(0.5,1)`` respectively, while ``x[1]`` intervals ``(-1,0)``
and ``(0,1)`` map into ``y[1]`` intervals ``(0,0.5)`` and  ``(0.5,1)``.

More typically, an uniform map with ``ninc`` increments 
is first created: for example, ::

    m = AdaptiveMap([[0, 1], [-1, 1]], ninc=1000)

creates a two-dimensional grid, with 1000 increments in each direction, 
that spans the volume ``0<=x[0]<=1``, ``-1<=x[1]<=1``. This map is then 
trained with data ``f[j]`` corresponding to ``ny`` points ``y[j, d]``,
with ``j=0...ny-1``, (usually) uniformly distributed in |y| space:
for example, ::

    m.add_training_data(y, f)
    m.adapt(alpha=1.5)

``m.adapt(alpha=1.5)`` shrinks grid increments where ``f[j]``
is large, and expands them where ``f[j]`` is small. Usually 
one has to iterate over several sets of ``y`` and ``f``
before the grid has fully adapted.

The speed with which the grid adapts is determined by parameter ``alpha``.
Large (positive) values imply rapid adaptation, while small values (much
less than one) imply slow adaptation. As in any iterative process that  
involves random numbers, it is  usually a good idea to slow adaptation 
down in order to avoid instabilities caused by random fluctuations.

Args:
    grid (list of arrays): Initial ``x`` grid, where ``grid[d][i]``
        is the ``i``-th node in direction ``d``. Different directions
        can have different numbers of nodes.
    ninc (int or array or ``None``): ``ninc[d]`` (or ``ninc``, if it 
        is a number) is the number of increments along direction ``d`` 
        in the new  ``x`` grid. The new grid is designed to give the same
        Jacobian ``dx(y)/dy`` as the original grid. The default value,
        ``ninc=None``, leaves the grid unchanged.
"""
struct Vegas <: AdaptiveMap
    dim::Int
    ninc::Vector{Int}
    grid::Matrix{Float64}
    inc::Matrix{Float64}

    function Vegas(_grid::AbstractVector, ninc=nothing)
        dim = length(_grid)
        len_g = [length(x) for x in _grid]
        @assert minimum(len_g) >= 2 "grid[d] must have at least 2 elements, not $(mininum(len_g))"
        ninc = len_g .- 1
        inc = zeros(dim, maximum(len_g))
        grid = zeros(dim, maximum(len_g))
        for d in 1:dim
            for (i, griddi) in enumerate(sort(_grid[d]))
                grid[d, i] = griddi
            end
            for i in 1:len_g[d]-1
                inc[d, i] = grid[d, i+1] - grid[d, i]
            end
        end
        # clear()
        # if (isnothing(ninc)==false) && not numpy.all(ninc == self.ninc):
        #     if numpy.all(numpy.asarray(self.ninc) == 1):
        #         self.make_uniform(ninc=ninc)
        #     else:
        #         self.adapt(ninc=ninc)
        #     end
        # end
        return new(dim, ninc, grid, inc)
    end
end

""" 

    function region(vegas::Vegas, d::Int=-1)

x-space region. ``region(d)`` returns a tuple ``(xl,xu)`` specifying the ``x``-space
interval covered by the map in direction ``d``. A list containing
the intervals for each direction is returned if ``d`` is omitted.
"""
function region(vegas::Vegas, d::Int=-1)
    if d < 1
        return [region(vegas, d) for d in 1:vegas.dim]
    else
        return (vegas.grid[d, 1], vegas.grid[d, vegas.ninc[d]+1])
    end
end

""" 
    function extract_grid(vegas::Vegas)

Return a vector of vectors specifying the map's grid. 
"""
extract_grid(vegas::Vegas) = [vegas.grid[d, 1:vegas.ninc[d]+1] for d in 1:vegas.dim]

""" Create string with information about grid nodes.

Creates a string containing the locations of the nodes
in the map grid for each direction. Parameter
``ngrid`` specifies the maximum number of nodes to print
(spread evenly over the grid).
"""
function settings(vegas::Vegas, ngrid=5)
    ans = ""
    if ngrid > 0
        for d in 1:vegas.dim
            grid_d = vegas.grid[d, 1:vegas.ninc[d]+1]
            nskip = vegas.ninc[d] ÷ ngrid
            if nskip < 1
                nskip = 1
            end
            start = nskip ÷ 2 + 1
            ans *= " grid[$d] = $(grid_d[start:nskip:end])\n"
        end
    end
    return ans
end

"""
    function randx(vegas::Vegas, n=nothing)
Create ``n`` random points in |x| space. 
"""
function randx(vegas::Vegas, n=nothing)
    if isnothing(n)
        y = rand(vegas.dim)
    else
        y = rand(n, vegas.dim)
    end
    # return self(y)
end

# def make_uniform(self, ninc=None):
#     """ Replace the grid with a uniform grid.

#     The new grid has ``ninc[d]``  (or ``ninc``, if it is a number) 
#     increments along each direction if ``ninc`` is specified.
#     If ``ninc=None`` (default), the new grid has the same number 
#     of increments in each direction as the old grid.
#     """
#     cdef numpy.npy_intp i, d
#     cdef numpy.npy_intp dim = self.grid.shape[0]
#     cdef double[:] tmp
#     cdef double[:, ::1] new_grid
#     if ninc is None:
#         ninc = numpy.asarray(self.ninc)
#     elif numpy.shape(ninc) == ():
#         ninc = numpy.full(self.dim, int(ninc), dtype=numpy.intp)
#     elif numpy.shape(ninc) == (self.dim,):
#         ninc = numpy.asarray(ninc)
#     else:
#         raise ValueError('ninc has wrong shape -- {}'.format(numpy.shape(ninc)))
#     if min(ninc) < 1:
#         raise ValueError(
#             "no of increments < 1 in AdaptiveMap -- %s"
#             % str(ninc)
#             )
#     new_inc = numpy.empty((dim, max(ninc)), numpy.float_)
#     new_grid = numpy.empty((dim, new_inc.shape[1] + 1), numpy.float_)
#     for d in range(dim):
#         tmp = numpy.linspace(self.grid[d, 0], self.grid[d, self.ninc[d]], ninc[d] + 1)
#         for i in range(ninc[d] + 1):
#             new_grid[d, i] = tmp[i]
#         for i in range(ninc[d]):
#             new_inc[d, i] = new_grid[d, i + 1] - new_grid[d, i]
#     self.ninc = ninc
#     self.grid = new_grid 
#     self.inc = new_inc 
#     self.clear()

"""
function y2x(vegas::Vegas, y=nothing)

Return ``x`` values corresponding to ``y``.
``y`` can be a single ``dim``-dimensional point, or it
can be an array ``y[i,j, ..., d]`` of such points (``d=1..dim``).

If ``y=nothing`` (default), ``y`` is set equal to a (uniform) random point
in the volume.
"""
function y2x(vegas::Vegas, y::Union{Nothing,AbstractArray{Float64}}=nothing)
    if isnothing(y)
        y = rand(vegas.dim)
    end
    y_shape = size(y)
    # y.shape = -1, y.shape[-1]
    # x = 0 * y
    # jac = numpy.empty(y.shape[0], numpy.float_)
    # self.map(y, x, jac)
    # x.shape = y_shape
    # return x
end

"""
    function jac1d(vegas::Vegas, y)

Return the map's Jacobian at ``y`` for each direction.

``y`` can be a single ``dim``-dimensional point, or it
can be an array ``y[i,j,...,d]`` of such points (``d=1..dim``).
Returns an array ``jac`` where ``jac[i,j,...,d]`` is the 
(one-dimensional) Jacobian (``dx[d]/dy[d]``) corresponding 
to ``y[i,j,...,d]``.
"""
function jac1d(vegas::Vegas, y::AbstractArray)
    dim = vegas.dim
    y_size = size(y)
    ny = length(y_size) == 1 ? 1 : reduce(*, y_size[1:end-1])
    y = reshape(y, ny, dim)
    jac = similar(y)
    for i in 1:ny
        for d in 1:dim
            ninc = vegas.ninc[d]
            y_ninc = y[i, d] * ninc
            iy = Int(floor(y_ninc)) + 1
            # dy_ninc = y_ninc - iy
            if iy < ninc
                jac[i, d] = vegas.inc[d, iy] * ninc
            else
                jac[i, d] = vegas.inc[d, ninc-1] * ninc
            end
        end
    end
    jac = reshape(jac, y_size...)
    return jac
end

# def jac(self, y):
#     """ Return the map's Jacobian at ``y``.

#     ``y`` can be a single ``dim``-dimensional point, or it
#     can be an array ``y[i,j,...,d]`` of such points (``d=0..dim-1``).
#     Returns an array ``jac`` where ``jac[i,j,...]`` is the 
#     (multidimensional) Jacobian (``dx/dy``) corresponding 
#     to ``y[i,j,...]``.
#     """
#     return numpy.prod(self.jac1d(y), axis=-1)

# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# cpdef map(
#     self,
#     double[:, ::1] y,
#     double[:, ::1] x,
#     double[::1] jac,
#     numpy.npy_intp ny=-1
#     ):
#     """ Map y to x, where jac is the Jacobian  (``dx/dy``).

#     ``y[j, d]`` is an array of ``ny`` ``y``-values for direction ``d``.
#     ``x[j, d]`` is filled with the corresponding ``x`` values,
#     and ``jac[j]`` is filled with the corresponding Jacobian
#     values. ``x`` and ``jac`` must be preallocated: for example, ::

#         x = numpy.empty(y.shape, float)
#         jac = numpy.empty(y.shape[0], float)

#     Args:
#         y (array): ``y`` values to be mapped. ``y`` is a contiguous
#             2-d array, where ``y[j, d]`` contains values for points
#             along direction ``d``.
#         x (array): Container for ``x[j, d]`` values corresponding
#             to ``y[j, d]``. Must be a contiguous 2-d array.
#         jac (array): Container for Jacobian values ``jac[j]`` (``= dx/dy``)
#             corresponding to ``y[j, d]``. Must be a contiguous 1-d array.
#         ny (int): Number of ``y`` points: ``y[j, d]`` for ``d=0...dim-1``
#             and ``j=0...ny-1``. ``ny`` is set to ``y.shape[0]`` if it is
#             omitted (or negative).
#     """
#     cdef numpy.npy_intp ninc 
#     cdef numpy.npy_intp dim = self.inc.shape[0]
#     cdef numpy.npy_intp i, iy, d
#     cdef double y_ninc, dy_ninc, tmp_jac
#     if ny < 0:
#         ny = y.shape[0]
#     elif ny > y.shape[0]:
#         raise ValueError('ny > y.shape[0]: %d > %d' % (ny, y.shape[0]))
#     for i in range(ny):
#         jac[i] = 1.
#         for d in range(dim):
#             ninc = self.ninc[d]
#             y_ninc = y[i, d] * ninc
#             iy = <int>floor(y_ninc)
#             dy_ninc = y_ninc  -  iy
#             if iy < ninc:
#                 x[i, d] = self.grid[d, iy] + self.inc[d, iy] * dy_ninc
#                 jac[i] *= self.inc[d, iy] * ninc
#             else:
#                 x[i, d] = self.grid[d, ninc]
#                 jac[i] *= self.inc[d, ninc - 1] * ninc
#     return

# cpdef invmap(
#     self,
#     double[:, ::1] x,
#     double[:, ::1] y,
#     double[::1] jac,
#     numpy.npy_intp nx=-1
#     ):
#     """ Map x to y, where jac is the Jacobian (``dx/dy``).

#     ``y[j, d]`` is an array of ``ny`` ``y``-values for direction ``d``.
#     ``x[j, d]`` is filled with the corresponding ``x`` values,
#     and ``jac[j]`` is filled with the corresponding Jacobian
#     values. ``x`` and ``jac`` must be preallocated: for example, ::

#         x = numpy.empty(y.shape, float)
#         jac = numpy.empty(y.shape[0], float)

#     Args:
#         x (array): ``x`` values to be mapped to ``y``-space. ``x`` 
#             is a contiguous 2-d array, where ``x[j, d]`` contains 
#             values for points along direction ``d``.
#         y (array): Container for ``y[j, d]`` values corresponding
#             to ``x[j, d]``. Must be a contiguous 2-d array
#         jac (array): Container for Jacobian values ``jac[j]`` (``= dx/dy``)
#             corresponding to ``y[j, d]``. Must be a contiguous 1-d array
#         nx (int): Number of ``x`` points: ``x[j, d]`` for ``d=0...dim-1``
#             and ``j=0...nx-1``. ``nx`` is set to ``x.shape[0]`` if it is
#             omitted (or negative).
#     """
#     cdef numpy.npy_intp ninc 
#     cdef numpy.npy_intp dim = self.inc.shape[0]
#     cdef numpy.npy_intp[:] iy
#     cdef numpy.npy_intp i, iyi, d
#     cdef double y_ninc, dy_ninc, tmp_jac
#     if nx < 0:
#         nx = x.shape[0]
#     elif nx > x.shape[0]:
#         raise ValueError('nx > x.shape[0]: %d > %d' % (nx, x.shape[0]))
#     for i in range(nx):
#         jac[i] = 1. 
#     for d in range(dim):
#         ninc = self.ninc[d]
#         iy = numpy.searchsorted(self.grid[d, :], x[:, d], side='right')
#         for i in range(nx):
#             if iy[i] > 0 and iy[i] <= ninc:
#                 iyi = iy[i] - 1
#                 y[i, d] = (iyi + (x[i, d] - self.grid[d, iyi]) / self.inc[d, iyi]) / ninc
#                 jac[i] *= self.inc[d, iyi] * ninc
#             elif iy[i] <= 0:
#                 y[i, d] = 0. 
#                 jac[i] *= self.inc[d, 0] * ninc 
#             elif iy[i] > ninc:
#                 y[i, d] = 1.0 
#                 jac[i] *= self.inc[d, ninc - 1] * ninc 
#     return               


# # @cython.boundscheck(False)
# # @cython.wraparound(False)
# cpdef add_training_data(
#     self,
#     double[:, ::1] y,
#     double[::1] f,
#     numpy.npy_intp ny=-1,
#     ):
#     """ Add training data ``f`` for ``y``-space points ``y``.

#     Accumulates training data for later use by ``self.adapt()``.
#     Grid increments will be made smaller in regions where
#     ``f`` is larger than average, and larger where ``f``
#     is smaller than average. The grid is unchanged (converged?)
#     when ``f`` is constant across the grid.

#     Args:
#         y (array): ``y`` values corresponding to the training data.
#             ``y`` is a contiguous 2-d array, where ``y[j, d]``
#             is for points along direction ``d``.
#         f (array): Training function values. ``f[j]`` corresponds to
#             point ``y[j, d]`` in ``y``-space.
#         ny (int): Number of ``y`` points: ``y[j, d]`` for ``d=0...dim-1``
#             and ``j=0...ny-1``. ``ny`` is set to ``y.shape[0]`` if it is
#             omitted (or negative).
#     """
#     cdef numpy.npy_intp ninc 
#     cdef numpy.npy_intp dim = self.inc.shape[0]
#     cdef numpy.npy_intp iy
#     cdef numpy.npy_intp i, d
#     if self.sum_f is None:
#         shape = (self.inc.shape[0], self.inc.shape[1])
#         self.sum_f = numpy.zeros(shape, numpy.float_)
#         self.n_f = numpy.zeros(shape, numpy.float_) + TINY
#     if ny < 0:
#         ny = y.shape[0]
#     elif ny > y.shape[0]:
#         raise ValueError('ny > y.shape[0]: %d > %d' % (ny, y.shape[0]))
#     for d in range(dim):
#         ninc = self.ninc[d]
#         for i in range(ny):
#             if y[i, d] > 0 and y[i, d] < 1:
#                 iy = <int> floor(y[i, d] * ninc)
#                 self.sum_f[d, iy] += abs(f[i])
#                 self.n_f[d, iy] += 1
#     return

# # @cython.boundscheck(False)
# def adapt(self, double alpha=0.0, ninc=None):
#     """ Adapt grid to accumulated training data.

#     ``self.adapt(...)`` projects the training data onto
#     each axis independently and maps it into ``x`` space.
#     It shrinks ``x``-grid increments in regions where the
#     projected training data is large, and grows increments
#     where the projected data is small. The grid along
#     any direction is unchanged if the training data
#     is constant along that direction.

#     The number of increments along a direction can be
#     changed by setting parameter ``ninc`` (array or number).

#     The grid does not change if no training data has
#     been accumulated, unless ``ninc`` is specified, in
#     which case the number of increments is adjusted
#     while preserving the relative density of increments
#     at different values of ``x``.

#     Args:
#         alpha (double): Determines the speed with which the grid
#             adapts to training data. Large (postive) values imply
#             rapid evolution; small values (much less than one) imply
#             slow evolution. Typical values are of order one. Choosing
#             ``alpha<0`` causes adaptation to the unmodified training
#             data (usually not a good idea).
#         ninc (int or array or None): The number of increments in the new 
#             grid is ``ninc[d]`` (or ``ninc``, if it is a number)
#             in direction ``d``. The number is unchanged from the 
#             old grid if ``ninc`` is omitted (or equals ``None``, 
#             which is the default).
#     """
#     cdef double[:, ::1] new_grid
#     cdef double[::1] avg_f, tmp_f
#     cdef double sum_f, acc_f, f_ninc
#     cdef numpy.npy_intp old_ninc
#     cdef numpy.npy_intp dim = self.grid.shape[0]
#     cdef numpy.npy_intp i, j
#     cdef numpy.npy_intp[:] new_ninc

#     # initialization
#     if ninc is None:
#         new_ninc = numpy.array(self.ninc)
#     elif numpy.shape(ninc) == ():
#         new_ninc = numpy.full(dim, int(ninc), numpy.intp)
#     elif len(ninc) == dim:
#         new_ninc = numpy.array(ninc, numpy.intp)
#     else:
#         raise ValueError('badly formed ninc = ' + str(ninc))
#     if min(new_ninc) < 1:
#         raise ValueError('ninc < 1: ' + str(list(new_ninc)))
#     if max(new_ninc) == 1:
#         new_grid = numpy.empty((dim, 2), numpy.float_)
#         for d in range(dim):
#             new_grid[d, 0] = self.grid[d, 0]
#             new_grid[d, 1] = self.grid[d, self.ninc[d]]
#         self.grid = numpy.asarray(new_grid)
#         self.inc = numpy.empty((dim, 1), numpy.float_)
#         self.ninc = numpy.array(dim * [1], dtype=numpy.intp)
#         for d in range(dim):
#             self.inc[d, 0] = self.grid[d, 1] - self.grid[d, 0]
#         self.clear()
#         return

#     # smooth and regrid
#     new_grid = numpy.empty((dim, max(new_ninc) + 1), numpy.float_)
#     avg_f = numpy.ones(self.inc.shape[1], numpy.float_) # default = uniform
#     if alpha > 0 and max(self.ninc) > 1:
#         tmp_f = numpy.empty(self.inc.shape[1], numpy.float_)
#     for d in range(dim):
#         old_ninc = self.ninc[d]
#         if alpha != 0 and old_ninc > 1:
#             if self.sum_f is not None:
#                 for i in range(old_ninc):
#                     if self.n_f[d, i] > 0:
#                         avg_f[i] = self.sum_f[d, i] / self.n_f[d, i]
#                     else:
#                         avg_f[i] = 0.
#             if alpha > 0:
#                 # smooth
#                 tmp_f[0] = abs(7. * avg_f[0] + avg_f[1]) / 8.
#                 tmp_f[old_ninc - 1] = abs(7. * avg_f[old_ninc - 1] + avg_f[old_ninc - 2]) / 8.
#                 sum_f = tmp_f[0] + tmp_f[old_ninc - 1]
#                 for i in range(1, old_ninc - 1):
#                     tmp_f[i] = abs(6. * avg_f[i] + avg_f[i-1] + avg_f[i+1]) / 8.
#                     sum_f += tmp_f[i]
#                 if sum_f > 0:
#                     for i in range(old_ninc):
#                         avg_f[i] = tmp_f[i] / sum_f + TINY
#                 else:
#                     for i in range(old_ninc):
#                         avg_f[i] = TINY
#                 for i in range(old_ninc):
#                     if avg_f[i] > 0 and avg_f[i] <= 0.99999999:
#                         avg_f[i] = (-(1 - avg_f[i]) / log(avg_f[i])) ** alpha
#         # regrid
#         new_grid[d, 0] = self.grid[d, 0]
#         new_grid[d, new_ninc[d]] = self.grid[d, old_ninc]
#         i = 0        # new_x index
#         j = -1         # self_x index
#         acc_f = 0   # sum(avg_f) accumulated
#         f_ninc = 0.
#         for i in range(old_ninc):
#             f_ninc += avg_f[i]
#         f_ninc /= new_ninc[d]     # amount of acc_f per new increment
#         for i in range(1, new_ninc[d]):
#             while acc_f < f_ninc:
#                 j += 1
#                 if j < old_ninc:
#                     acc_f += avg_f[j]
#                 else:
#                     break
#             else:
#                 acc_f -= f_ninc
#                 new_grid[d, i] = (
#                     self.grid[d, j+1]
#                     - (acc_f / avg_f[j]) * self.inc[d, j]
#                     )
#                 continue
#             break
#     self.grid = numpy.asarray(new_grid)
#     self.inc = numpy.empty((dim, self.grid.shape[1] - 1), float)
#     for d in range(dim):
#         for i in range(new_ninc[d]):
#             self.inc[d, i] = self.grid[d, i + 1] - self.grid[d, i]
#     self.ninc = numpy.asarray(new_ninc)
#     self.clear()

# def clear(self):
#     " Clear information accumulated by :meth:`AdaptiveMap.add_training_data`. "
#     self.sum_f = None 
#     self.n_f = None

# def show_grid(self, ngrid=40, axes=None, shrink=False, plotter=None):
#     """ Display plots showing the current grid.

#     Args:
#         ngrid (int): The number of grid nodes in each
#             direction to include in the plot. The default is 40.
#         axes: List of pairs of directions to use in
#             different views of the grid. Using ``None`` in
#             place of a direction plots the grid for only one
#             direction. Omitting ``axes`` causes a default
#             set of pairings to be used.
#         shrink: Display entire range of each axis
#             if ``False``; otherwise shrink range to include
#             just the nodes being displayed. The default is
#             ``False``.
#         plotter: :mod:`matplotlib` plotter to use for plots; plots
#             are not displayed if set. Ignored if ``None``, and 
#             plots are displayed using ``matplotlib.pyplot``.
#     """
#     if plotter is not None:
#         plt = plotter
#     else:
#         try:
#             import matplotlib.pyplot as plt
#         except ImportError:
#             warnings.warn('matplotlib not installed; cannot show_grid')
#             return
#     dim = self.dim
#     if axes is None:
#         axes = []
#         if dim == 1:
#             axes = [(0, None)]
#         for d in range(dim):
#             axes.append((d, (d + 1) % dim))
#     else:
#         if len(axes) <= 0:
#             return
#         for dx,dy in axes:
#             if dx is not None and (dx < 0 or dx >= dim):
#                 raise ValueError('bad directions: %s' % str((dx, dy)))
#             if dy is not None and (dy < 0 or dy >= dim):
#                 raise ValueError('bad directions: %s' % str((dx, dy)))
#     fig = plt.figure()
#     def plotdata(idx, grid=numpy.asarray(self.grid)):
#         dx, dy = axes[idx[0]]
#         if dx is not None:
#             nskip = int(self.ninc[dx] // ngrid)
#             if nskip < 1:
#                 nskip = 1
#             start = nskip // 2
#             xrange = [self.grid[dx, 0], self.grid[dx, self.ninc[dx]]]
#             xgrid = grid[dx, start::nskip]
#             xlabel = 'x[%d]' % dx
#         else:
#             xrange = [0., 1.]
#             xgrid = None
#             xlabel = ''
#         if dy is not None:
#             nskip = int(self.ninc[dy] // ngrid)
#             if nskip < 1:
#                 nskip = 1
#             start = nskip // 2
#             yrange = [self.grid[dy, 0], self.grid[dy, self.ninc[dy]]]
#             ygrid = grid[dy, start::nskip]
#             ylabel = 'x[%d]' % dy
#         else:
#             yrange = [0., 1.]
#             ygrid = None
#             ylabel = ''
#         if shrink:
#             if xgrid is not None:
#                 xrange = [min(xgrid), max(xgrid)]
#             if ygrid is not None:
#                 yrange = [min(ygrid), max(ygrid)]
#         if None not in [dx, dy]:
#             fig_caption = 'axes %d, %d' % (dx, dy)
#         elif dx is None and dy is not None:
#             fig_caption = 'axis %d' % dy
#         elif dx is not None and dy is None:
#             fig_caption = 'axis %d' % dx
#         else:
#             return
#         fig.clear()
#         plt.title(
#             "%s   (press 'n', 'p', 'q' or a digit)"
#             % fig_caption
#             )
#         plt.xlabel(xlabel)
#         plt.ylabel(ylabel)
#         if xgrid is not None:
#             for i in range(len(xgrid)):
#                 plt.plot([xgrid[i], xgrid[i]], yrange, 'k-')
#         if ygrid is not None:
#             for i in range(len(ygrid)):
#                 plt.plot(xrange, [ygrid[i], ygrid[i]], 'k-')
#         plt.xlim(*xrange)
#         plt.ylim(*yrange)

#         plt.draw()

#     idx = [0]
#     def onpress(event, idx=idx):
#         try:    # digit?
#             idx[0] = int(event.key)
#         except ValueError:
#             if event.key == 'n':
#                 idx[0] += 1
#                 if idx[0] >= len(axes):
#                     idx[0] = len(axes) - 1
#             elif event.key == 'p':
#                 idx[0] -= 1
#                 if idx[0] < 0:
#                     idx[0] = 0
#             elif event.key == 'q':
#                 plt.close()
#                 return
#             else:
#                 return
#         plotdata(idx)

#     fig.canvas.mpl_connect('key_press_event', onpress)
#     plotdata(idx)
#     if plotter is None:
#         plt.show()
#     else:
#         return plt

# def adapt_to_samples(self, x, f, nitn=5, alpha=1.0): # , ninc=None):
#     """ Adapt map to data ``{x, f(x)}``.

#     Replace grid with one that is optimized for integrating 
#     function ``f(x)``. New grid is found iteratively

#     Args:
#         x (array): ``x[:, d]`` are the components of the sample points 
#             in direction ``d=0,1...self.dim-1``.
#         f (callable or array): Function ``f(x)`` to be adapted to. If 
#             ``f`` is an array, it is assumes to contain values ``f[i]``
#             corresponding to the function evaluated at points ``x[i]``.
#         nitn (int): Number of iterations to use in adaptation. Default
#             is ``nitn=5``.
#         alpha (float): Damping parameter for adaptation. Default 
#             is ``alpha=1.0``. Smaller values slow the iterative 
#             adaptation, to improve stability of convergence.
#     """
#     cdef numpy.npy_intp i, tmp_ninc, old_ninc
#     x = numpy.ascontiguousarray(x)
#     if len(x.shape) != 2 or x.shape[1] != self.dim:
#         raise ValueError('incompatible shape of x: {}'.format(x.shape))
#     if callable(f):
#         fx = numpy.ascontiguousarray(f(x))
#     else:
#         fx = numpy.ascontiguousarray(f)
#     if fx.shape[0] != x.shape[0]:
#         raise ValueError('shape of x and f(x) mismatch: {} vs {}'.format(x.shape, fx.shape))
#     old_ninc = max(max(self.ninc), Integrator.defaults['maxinc_axis'])
#     tmp_ninc = min(old_ninc, x.shape[0] / 10.) 
#     if tmp_ninc < 2:
#         raise ValueError('not enough samples: {}'.format(x.shape[0]))
#     y = numpy.empty(x.shape, float)
#     jac = numpy.empty(x.shape[0], float)
#     for i in range(nitn):
#         self.invmap(x, y, jac)
#         self.add_training_data(y, (jac * fx) ** 2)
#         self.adapt(alpha=alpha, ninc=tmp_ninc)
#     if numpy.any(tmp_ninc != old_ninc):
#         self.adapt(ninc=old_ninc)