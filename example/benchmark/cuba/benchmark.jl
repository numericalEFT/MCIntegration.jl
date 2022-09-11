### benchmark.jl --- Benchmark Cuba.jl and Cuba C Library

# Copyright (C) 2016-2019  Mosè Giordano

# Maintainer: Mosè Giordano <mose AT gnu DOT org>
# Keywords: numeric integration

# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.

# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

### Commentary:

# Load this file in order to compare performance of Cuba.jl and original Cuba
# Library in C.

### Code:

using Cuba, Printf, MCIntegration

const ndim = 3
const ncomp = 11
const atol = 1e-8
const rtol = 1e-8

rsq(x, y, z) = abs2(x) + abs2(y) + abs2(z)
t1(x, y, z) = sin(x) * cos(y) * exp(z)
t2(x, y, z) = 1.0 / ((x + y) * (x + y) + 0.003) * cos(y) * exp(z)
t3(x, y, z) = 1.0 / (3.75 - cos(pi * x) - cos(pi * y) - cos(pi * z))
t4(x, y, z) = abs(rsq(x, y, z) - 0.125)
t5(x, y, z) = exp(-rsq(x, y, z))
t6(x, y, z) = 1.0 / (1.0 - x * y * z + 1e-10)
t7(x, y, z) = sqrt(abs(x - y - z))
t8(x, y, z) = exp(-x * y * z)
t9(x, y, z) = abs2(x) / (cos(x + y + z + 1.0) + 5.0)
t10(x, y, z) = (x > 0.5) ? 1.0 / sqrt(x * y * z + 1e-5) : sqrt(x * y * z)
t11(x, y, z) = (rsq(x, y, z) < 1.0) ? 1.0 : 0.0
function test(x::Vector{Float64}, f::Vector{Float64})
    @inbounds f[1] = t1(x[1], x[2], x[3])
    @inbounds f[2] = t2(x[1], x[2], x[3])
    @inbounds f[3] = t3(x[1], x[2], x[3])
    @inbounds f[4] = t4(x[1], x[2], x[3])
    @inbounds f[5] = t5(x[1], x[2], x[3])
    @inbounds f[6] = t6(x[1], x[2], x[3])
    @inbounds f[7] = t7(x[1], x[2], x[3])
    @inbounds f[8] = t8(x[1], x[2], x[3])
    @inbounds f[9] = t9(x[1], x[2], x[3])
    @inbounds f[10] = t10(x[1], x[2], x[3])
    @inbounds f[11] = t11(x[1], x[2], x[3])
end

function test2(x, c)
    @inbounds return t1(x[1], x[2], x[3]),
    t2(x[1], x[2], x[3]),
    t3(x[1], x[2], x[3]),
    t4(x[1], x[2], x[3]),
    t5(x[1], x[2], x[3]),
    t6(x[1], x[2], x[3]),
    t7(x[1], x[2], x[3]),
    t8(x[1], x[2], x[3]),
    t9(x[1], x[2], x[3]),
    t10(x[1], x[2], x[3]),
    t11(x[1], x[2], x[3])
end

function test2(x, f::Vector{Float64}, c::Configuration)
    @inbounds f[1] = t1(x[1], x[2], x[3])
    @inbounds f[2] = t2(x[1], x[2], x[3])
    @inbounds f[3] = t3(x[1], x[2], x[3])
    @inbounds f[4] = t4(x[1], x[2], x[3])
    @inbounds f[5] = t5(x[1], x[2], x[3])
    @inbounds f[6] = t6(x[1], x[2], x[3])
    @inbounds f[7] = t7(x[1], x[2], x[3])
    @inbounds f[8] = t8(x[1], x[2], x[3])
    @inbounds f[9] = t9(x[1], x[2], x[3])
    @inbounds f[10] = t10(x[1], x[2], x[3])
    @inbounds f[11] = t11(x[1], x[2], x[3])
end

@info "Performance of Cuba.jl:"
for alg in (vegas, suave, divonne, cuhre)
    # Run the integrator a first time to compile the function.
    alg(test, ndim, ncomp, atol=atol,
        rtol=rtol)
    start_time = time_ns()
    alg(test, ndim, ncomp, atol=atol,
        rtol=rtol)
    end_time = time_ns()
    println(@sprintf("%10.6f", Int(end_time - start_time) / 1e9),
        " seconds (", uppercasefirst(string(nameof(alg))), ")")
    # Vegas result
    # julia > vegas(test, ndim, ncomp, atol=atol, rtol=rtol)
    # Components:
    #   1: 0.6646695631037952 ± 0.0005192264010635556 (prob.: 0.0)
    #   2: 5.268641443643625 ± 0.00937817147924112 (prob.: 0.0)
    #   3: 0.30780794363958314 ± 0.00016915694878023873 (prob.: 0.0)
    #   4: 0.8773127802819499 ± 0.0007272020591230047 (prob.: 0.0)
    #   5: 0.4165412720874263 ± 0.00026445560918383573 (prob.: 0.0)
    #   6: 1.2020166963125194 ± 0.0007653098425722113 (prob.: 0.0)
    #   7: 0.7096235523946715 ± 0.000489330660608438 (prob.: 0.0)
    #   8: 0.8912187300170151 ± 0.00040985374702141484 (prob.: 0.0)
    #   9: 0.08018533940010736 ± 8.303642721037487e-5 (prob.: 0.0)
    #  10: 2.3963590312091303 ± 0.0032631645439722893 (prob.: 7.787446180547297e-17)
    #  11: 0.5236360284653482 ± 0.0005842099358276612 (prob.: 0.0)
    # Integrand evaluations: 1007500
    # Number of subregions:  0
    # Note: The accuracy was not met within the maximum number of evaluations

    # Vegasmc result
    # julia> @time integrate(test2; dof=[[3,] for i in 1:11], neval=1e5, solver=:vegasmc, print=-1)
    # 0.495360 seconds (57.95 k allocations: 4.639 MiB, 4.18% compilation time)
    # Integral 1 = 0.6636210504918344 ± 0.0020729178999968614   (chi2/dof = 0.899)
    # Integral 2 = 5.242919616854019 ± 0.02260128007562855   (chi2/dof = 1.42)
    # Integral 3 = 0.30681239353464046 ± 0.0007850307152916456   (chi2/dof = 0.628)
    # Integral 4 = 0.8780127497538999 ± 0.0027696019511458045   (chi2/dof = 1.19)
    # Integral 5 = 0.4144658711129951 ± 0.0011614492930177123   (chi2/dof = 0.71)
    # Integral 6 = 1.2002718763309468 ± 0.003387144818775182   (chi2/dof = 1.02)
    # Integral 7 = 0.7095634473688555 ± 0.002250824818469535   (chi2/dof = 0.665)
    # Integral 8 = 0.8880588065576488 ± 0.0022868326414642133   (chi2/dof = 0.716)
    # Integral 9 = 0.08033351807552817 ± 0.0002957920163406668   (chi2/dof = 0.937)
    # Integral 10 = 2.417025425999495 ± 0.011011866855686713   (chi2/dof = 0.268)
    # Integral 11 = 0.5200202669127822 ± 0.0016844693189224066   (chi2/dof = 0.806)
end

@info "Performance of MCIntegration:"
for alg in (:vegas, :vegasmc)
    # for alg in (:vegasmc,)
    # Run the integrator a first time to compile the function.
    integrate(test2; dof=[[3,] for i in 1:11], neval=1e4, solver=alg, print=-1)
    start_time = time_ns()
    integrate(test2; dof=[[3,] for i in 1:11], neval=1e5, solver=alg, print=-2)
    # Cuba will run for 1e6 steps
    end_time = time_ns()
    println(@sprintf("%10.6f", Int(end_time - start_time) / 1e9),
        " seconds (", uppercasefirst(string(alg)), ")")
    # result
    # julia> @time integrate(test2; dof=[[3,] for i in 1:11], neval=1e5, solver=:vegas, print=-1)
    # 0.246125 seconds (57.28 k allocations: 4.542 MiB, 9.19% compilation time)
    # Integral 1 = 0.6662389036758806 ± 0.0009370223870430191   (chi2/dof = 0.518)
    # Integral 2 = 5.256465932385138 ± 0.007319608002109763   (chi2/dof = 1.48)
    # Integral 3 = 0.3075345398612458 ± 0.00024117145696156536   (chi2/dof = 1.59)
    # Integral 4 = 0.8788561548719585 ± 0.001217419754053441   (chi2/dof = 0.792)
    # Integral 5 = 0.4161189703927425 ± 0.00038591698396624104   (chi2/dof = 1.59)
    # Integral 6 = 1.2031224696691054 ± 0.0014223459029237325   (chi2/dof = 0.79)
    # Integral 7 = 0.7100206908822683 ± 0.0007923510194941002   (chi2/dof = 1.52)
    # Integral 8 = 0.8912870247582633 ± 0.0007501319218471907   (chi2/dof = 1.25)
    # Integral 9 = 0.08030537560622107 ± 0.00013380714402722626   (chi2/dof = 0.304)
    # Integral 10 = 2.396395164639421 ± 0.0035658664551599913   (chi2/dof = 0.938)
    # Integral 11 = 0.5228992089675735 ± 0.0008064389576126581   (chi2/dof = 1.83)

end

cd(@__DIR__) do
    if mtime("benchmark.c") > mtime("benchmark-c")
        run(`gcc -O3 -I $(Cuba.Cuba_jll.artifact_dir)/include -o benchmark-c benchmark.c $(Cuba.Cuba_jll.libcuba_path) -lm`)
    end
    @info "Performance of Cuba Library in C:"
    withenv(Cuba.Cuba_jll.JLLWrappers.LIBPATH_env => Cuba.Cuba_jll.LIBPATH[]) do
        run(`./benchmark-c`)
    end

    if success(`which gfortran`)
        if mtime("benchmark.f") > mtime("benchmark-fortran")
            run(`gfortran -O3 -fcheck=no-bounds -cpp -o benchmark-fortran benchmark.f $(Cuba.Cuba_jll.libcuba_path) -lm`)
        end
        @info "Performance of Cuba Library in Fortran:"
        withenv(Cuba.Cuba_jll.JLLWrappers.LIBPATH_env => Cuba.Cuba_jll.LIBPATH[]) do
            run(`./benchmark-fortran`)
        end
    end
end
