### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 51f340bc-29cc-11ec-0a93-2b2b6dad10c3
using Pkg; Pkg.activate(".")

# ╔═╡ 160848f5-5fc4-437e-b6bd-9f8b77222818
begin
	using PyCall # Python
	using RCall # R
end

# ╔═╡ 7eb4a41c-b25d-40e7-b931-7079325f0ee8
md"""
# Julia Academy Data Sciences

## 11. Other Languages

This is a Pluto write-up of the Julia Academy Data Sciences course. This time we are looking at how to run code from other languages in Julia. We focus here on Python, R, and C.

Start by loading the packages. I'm using my local package environment as opposed to the Pluto.jl environment.
"""

# ╔═╡ 94df7e61-d82c-44dc-9a37-89d279a83e6d
md"""
### Python

We can import any python package.
"""

# ╔═╡ bef76359-2fc7-4126-b082-1852b1abbd45
maths = pyimport("math")

# ╔═╡ b718b9fe-dc81-4594-9110-c587c523bf95
maths.sin(maths.pi / 4)

# ╔═╡ 38388d3e-3225-4123-979f-7b76683fc0b2
netx = pyimport("networkx")

# ╔═╡ 07dc16bc-803c-4670-bdf3-3e3495e7274b
md"""
We can write python code in a similar set up to how we are writing markdown in Pluto.

We need to have everything in the same cell in Pluto - this is not required for Jupyter.
"""

# ╔═╡ c63fe7e4-8626-4fd5-ad91-5b0fd49d601e
begin
	py"""
	import numpy
	def find_best_fit_python(xvals,yvals):
		meanx = numpy.mean(xvals)
		meany = numpy.mean(yvals)
		stdx = numpy.std(xvals)
		stdy = numpy.std(yvals)
		r = numpy.corrcoef(xvals,yvals)[0][1]
		a = r*stdy/stdx
		b = meany - a*meanx
		return a,b
	"""
	xvals = repeat(1:0.5:10, inner=2)
	yvals = 3 .+ xvals .+ 2 .* rand(length(xvals)) .-1
	find_best_fit_python = py"find_best_fit_python"
	a,b = find_best_fit_python(xvals,yvals)
end

# ╔═╡ cae9ff4e-4394-47c3-a104-a59479a26802
md"""
We can load python files using `pyimport`. For example if the above function was in a file called `fit_linear.py`:

```julia
python_fit = pyimport("fit_linear")
python_fit.find_best_fit_python(xvals, yvals)
```
"""

# ╔═╡ 9fbdd475-7da3-4e59-bb96-e7b59f8699ea
md"""
# R Code using `RCall`

In the Julia REPL we can switch to an `R` REPL using `$`. This doesn't work in a notebook environment.
"""

# ╔═╡ dc1e7043-f2aa-4092-865a-21e7a9389b5a
r = rcall(:sum, Float64[1.0, 4.0, 6.0])

# ╔═╡ d281c921-465f-4667-838e-9bf95ad83f43
typeof(r[1])

# ╔═╡ d6c2edb7-6f5d-44ac-ae9c-3daf56e77941
typeof(r)

# ╔═╡ 72e8951b-513b-4a87-a0c0-e74c8428fa8c
md"""
We can put a variable into `R` context using the `@rput` macro.
"""

# ╔═╡ 84908591-491b-48d2-9044-84d183a014f1
z = 1; @rput z

# ╔═╡ a777c8bf-2ae9-423f-bb2a-a95a762d63c3
r2 = R"z+z"

# ╔═╡ 3438433e-8930-40be-805a-ffc120d2e392
md"""
We can also apply `R` functions to Julia variables using `@rimport`.
"""

# ╔═╡ 10de9f75-efef-40b1-bdad-d52e1844a9a3
begin
	xrd = randn(10)
	
	@rimport base as rbase
	rbase.sum(xrd)
end

# ╔═╡ 8b533498-d2c1-4417-8a20-3cf4eae5e7dd
using HypothesisTests; OneSampleTTest(xrd)

# ╔═╡ c45603dc-7f70-458e-9325-736e800920b2
sum(xrd)

# ╔═╡ 7437bc44-af7f-41a2-8dcf-06f05dbb3616
md"""
We can also load libraries - if they are already installed in `R`.
"""

# ╔═╡ b25dd584-6313-4139-8af8-d89ce811d121
@rlibrary boot

# ╔═╡ f5a435e9-0b85-4284-800a-87f66177c8e4
R"t.test($xrd)"

# ╔═╡ 8810beb7-6809-4dce-b028-f32b5062084e
md"""
### C Code

We can call `C` libraries directly without needing to load an additional package.
"""

# ╔═╡ f71d223b-8d3a-4d30-b2cb-bb02e482f85b
t = ccall(:clock, Int32, ())

# ╔═╡ 1d17bdd2-4901-4677-8ff2-e734928fa67e
md"""
### Summary

We can easily make use of different languages in Julia. Often this is easier in the REPL, for example switching to `R` REPL using `$` - just as we can switch to `shell` using `;`. 

It is important to note that if we want to load an `R` or `python` (for example) library they must be installed on our system first - for `python` I need to use `pip`, but for `R` we can do it in the Julia REPL with `$` and `install.packages`.
"""

# ╔═╡ Cell order:
# ╠═51f340bc-29cc-11ec-0a93-2b2b6dad10c3
# ╟─7eb4a41c-b25d-40e7-b931-7079325f0ee8
# ╠═160848f5-5fc4-437e-b6bd-9f8b77222818
# ╟─94df7e61-d82c-44dc-9a37-89d279a83e6d
# ╠═bef76359-2fc7-4126-b082-1852b1abbd45
# ╠═b718b9fe-dc81-4594-9110-c587c523bf95
# ╠═38388d3e-3225-4123-979f-7b76683fc0b2
# ╟─07dc16bc-803c-4670-bdf3-3e3495e7274b
# ╠═c63fe7e4-8626-4fd5-ad91-5b0fd49d601e
# ╟─cae9ff4e-4394-47c3-a104-a59479a26802
# ╟─9fbdd475-7da3-4e59-bb96-e7b59f8699ea
# ╠═dc1e7043-f2aa-4092-865a-21e7a9389b5a
# ╠═d281c921-465f-4667-838e-9bf95ad83f43
# ╠═d6c2edb7-6f5d-44ac-ae9c-3daf56e77941
# ╟─72e8951b-513b-4a87-a0c0-e74c8428fa8c
# ╠═84908591-491b-48d2-9044-84d183a014f1
# ╠═a777c8bf-2ae9-423f-bb2a-a95a762d63c3
# ╟─3438433e-8930-40be-805a-ffc120d2e392
# ╠═10de9f75-efef-40b1-bdad-d52e1844a9a3
# ╠═c45603dc-7f70-458e-9325-736e800920b2
# ╟─7437bc44-af7f-41a2-8dcf-06f05dbb3616
# ╠═b25dd584-6313-4139-8af8-d89ce811d121
# ╠═f5a435e9-0b85-4284-800a-87f66177c8e4
# ╠═8b533498-d2c1-4417-8a20-3cf4eae5e7dd
# ╟─8810beb7-6809-4dce-b028-f32b5062084e
# ╠═f71d223b-8d3a-4d30-b2cb-bb02e482f85b
# ╟─1d17bdd2-4901-4677-8ff2-e734928fa67e
