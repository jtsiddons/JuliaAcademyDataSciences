### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 1581f292-1d72-40ab-8126-d48a482f11d1
using CairoMakie, Statistics, StatsBase, DataFrames, GLM, Tables, XLSX, MLBase, LsqFit, LinearAlgebra, RDatasets

# ╔═╡ e5115c3a-25cd-11ec-1d64-67ddeae7e0f7
md"""
# Julia Academy Data Sciences Course

## 07. Regression

Another `Pluto.jl` write-up of the Julia Academy Data Sciences Course. Class 7 is on Regression.

Regression is the method by which we find a set of coefficients for predictors to predict the outcome of some data. Typically the funciton is linear and we want to find a linear combination of our predictors to find our values.

The course version of this notebook imports `PyCall` to demonstrate that the `numpy.polyfit` function gives the same results.

As ever, start by loading required libraries.
"""

# ╔═╡ ebba71b6-497f-4e7b-b68e-2f90dcdbbd72
md"""
### A quick demonstration

First we generate some random data that we can fit a straight line to.
"""

# ╔═╡ 0682072e-a9f6-453c-8e2a-c8793d6c7579
begin
	xvals = repeat(0:0.5:10, inner=3)
	yvals = 4 .- xvals .+ 2 .* rand(length(xvals)) .+2
	F = Figure()
	ax = Axis(F[1,1])
	scatter!(ax,xvals, yvals, color=:black)
	F
end

# ╔═╡ e2fcc903-cd21-4fbf-9da1-4ffa4747080a
df = DataFrame(X=xvals, Y=yvals);

# ╔═╡ 67ce0bab-6fd6-44fb-9a10-4aa4e14f63cd
md"""
#### Linear Regression

Let's write a quick function to perform linear regression on the above data.

For 

$$y = ax + b$$

we have

$$r= cor(x,y)$$

$$a = r \frac{σ_y}{σ_x}$$

$$b = \bar{y} - a\bar{x}$$
"""

# ╔═╡ 5c3b5507-4520-484d-93e4-5d3e87a6a03d
function find_best_fit(x, y)
	x̄ = mean(x); ȳ = mean(y);
	σ_x = std(x); σ_y = std(y);
	r = cor(x, y);
	
	a = r * σ_y/σ_x;
	b = ȳ - a*x̄;
	return a, b
end

# ╔═╡ a253c672-93be-494f-9677-f8b0e799b653
a, b = find_best_fit(xvals, yvals)

# ╔═╡ df7dbef2-2a1b-4c4c-8b5a-75c8ca3a9c85
y_new_hm = a .* xvals .+ b;

# ╔═╡ f9dd21a1-560a-4903-b8dd-869ad3ebc8bf
begin
	lines!(ax, xvals, y_new_hm, label="find_best_fit", linewidth=2)
	axislegend(ax)
	F
end

# ╔═╡ 753113d8-56a0-42ca-805c-d30d899087c0
md"""
As we can see it performs reasonably well at predicting the original data. 

#### `GLM.jl`

Now let's use the `GLM` package and see how it compares. We use the DataFrame form of the data, stored in `df`. We can then use `GLM` in much the same way as `R`. We define our function `Y~X` in an `@formula`.
"""

# ╔═╡ 6f52bfa8-fc97-442c-b0ad-860c271f686f
ols = lm(@formula(Y ~ X), df)

# ╔═╡ c055cc56-778f-42df-b11a-8e56d88cba1b
begin
	lines!(ax, xvals, predict(ols), linewidth=2, linestyle=:dash, label="GLM")
	axislegend(ax)
	F
end

# ╔═╡ b1e4efe8-ae39-4033-9e6a-3c7f05fc4e0b
norm(y_new_hm-predict(ols))

# ╔═╡ cee7352b-c55f-4220-a8f9-19bb45bf39bf
md"""
### Sales Data

We now look at the Zillow Sales Data that we looked at in a previous notebook. We look to build a regression between number of houses listed against the number sold in different states.
"""

# ╔═╡ 02fd8825-9178-412c-98be-938d71c599aa
file = "./data/zillow_data_download_april2020.xlsx"

# ╔═╡ 32928d1b-d6f9-44d6-b68d-8a2c9c239c07
md"""
We want the `Sale_counts_city` worksheet and the `MonthlyListings_City` worksheet.
"""

# ╔═╡ 98531e6c-b365-4562-b108-3218b35af9b1
begin
	sale_counts = "Sale_counts_city"
	df_sale_counts = DataFrame(XLSX.readtable(file,sale_counts)...)
end

# ╔═╡ 1d31819a-21ca-4d54-a0ac-062de0036da0
begin
	monthly_listings = "MonthlyListings_City"
	df_monthly_listings = DataFrame(XLSX.readtable(file,monthly_listings)...)
end

# ╔═╡ f8885a1f-90c1-44eb-ae3e-04b5d1997d5d
md"""
In this exercise we are only interested in the last month of data - so from "2020-02".
"""

# ╔═╡ 116335a3-892c-4296-90bb-f044589bc0b3
begin
	select!(df_sale_counts, ["RegionID", "2020-02"])
	rename!(df_sale_counts, "2020-02" => :sales_2020_02)
	
	select!(df_monthly_listings, ["SizeRank", "RegionID", "RegionName", "RegionType", "StateName", "2020-02"])
	rename!(df_monthly_listings, "2020-02" => :listings_2020_02)
end

# ╔═╡ 9d04b5d0-ea66-4adb-920b-1f59b58328fa
df_2020_02 = innerjoin(df_monthly_listings, df_sale_counts, on=:RegionID)

# ╔═╡ 38f65135-cc7c-4165-9a5f-c6d1f1fe92ff
dropmissing!(df_2020_02)

# ╔═╡ 0be742f8-c4d8-4e5b-8383-06993dade831
md"""
Let's look at the sold/listings for each state. I select the 10 states with the most sales.
"""

# ╔═╡ 9b7495b2-a3f3-47d9-81a3-1a2814a9c5ab
begin
	df_state_sales = combine(groupby(df_2020_02, :StateName), :sales_2020_02 => sum => :state_sales)
	sort!(df_state_sales, :state_sales, rev=true)
	states = df_state_sales.StateName[1:10]
end

# ╔═╡ 6037bf5b-53ab-482f-97ae-f8ca1441b97d
md"""
#### With Intercept
"""

# ╔═╡ c994a0e0-16dc-4c9d-b3aa-8432681b8d08
begin
	# Convert columns to Float.
	df_2020_02.sales_2020_02 = float.(df_2020_02.sales_2020_02)
	df_2020_02.listings_2020_02 = float.(df_2020_02.listings_2020_02)
	
	# Figure set-up
	F_states1 = Figure(resolution=(1500,600))
	cols = [:lightblue, :lightcoral, :lightgoldenrod, :darkseagreen1, :paleturquoise1, :cadetblue1, :lightskyblue1, :plum1, :bisque2, :lightpink]
	modls = []
	for (i,st) in enumerate(states)
		tmpdf = filter(row -> row.StateName == st, df_2020_02)
		ols = lm(@formula(sales_2020_02 ~ listings_2020_02), tmpdf)
		push!(modls, ols)
		
		ii,jj = (Int(floor((i-1)/5)+1), ((i-1)%5)+1);
		
		curAx = Axis(F_states1[ii,jj])
		scatter!(curAx, tmpdf.listings_2020_02, tmpdf.sales_2020_02, color=cols[i], markersize=5)
		lines!(tmpdf.listings_2020_02, predict(ols), color=:black, linewidth=2)
		curAx.title = st
		if ii == 2 && jj == 3
			curAx.xlabel = "Listings"
		end
		if jj == 1
			curAx.ylabel = "Sales"
		end
		curAx.limits = (0, 500, 0, 500)
		curAx.aspect=1
	end
	F_states1
end		

# ╔═╡ b39ded5d-c480-4e34-907a-95eba07cd43f
md"""
#### No Intercept
"""

# ╔═╡ 0c2b2e28-7854-400e-b85d-0ae950daa84a
begin
	# Convert columns to Float.
	df_2020_02.sales_2020_02 = float.(df_2020_02.sales_2020_02)
	df_2020_02.listings_2020_02 = float.(df_2020_02.listings_2020_02)
	
	# Figure set-up
	F_states0 = Figure(resolution=(1500,600))
	modls0 = []
	for (i,st) in enumerate(states)
		tmpdf = filter(row -> row.StateName == st, df_2020_02)
		ols = lm(@formula(sales_2020_02 ~ 0 + listings_2020_02), tmpdf)
		push!(modls0, ols)
		
		ii,jj = (Int(floor((i-1)/5)+1), ((i-1)%5)+1);
		
		curAx = Axis(F_states0[ii,jj])
		scatter!(curAx, tmpdf.listings_2020_02, tmpdf.sales_2020_02, color=cols[i], markersize=5)
		lines!(tmpdf.listings_2020_02, predict(ols), color=:black, linewidth=2)
		curAx.title = st
		if ii == 2 && jj == 3
			curAx.xlabel = "Listings"
		end
		if jj == 1
			curAx.ylabel = "Sales"
		end
		curAx.limits = (0, 500, 0, 500)
		curAx.aspect=1
	end
	F_states0
end		

# ╔═╡ 1493dd7c-667f-42eb-a326-5e159f958262
md"""
#### All on one plot - No Intercept

One issue with `Makie` is the lack of `alpha`. This makes the plot look ugly compared to with `Plots`. Oh well. To counter this, I have added the lines after adding all of the scatter points. This puts the lines on top of the scatter points.
"""

# ╔═╡ ced60d91-ad1e-43d1-8187-6034140e9e6b
begin
	# Convert columns to Float.
	df_2020_02.sales_2020_02 = float.(df_2020_02.sales_2020_02)
	df_2020_02.listings_2020_02 = float.(df_2020_02.listings_2020_02)
	
	# Figure set-up
	F_states = Figure(resolution=(600,600))
	Ax_states = Axis(F_states[1,1])
	for (i,st) in enumerate(states)
		tmpdf = filter(row -> row.StateName == st, df_2020_02)		
		scatter!(Ax_states, tmpdf.listings_2020_02, tmpdf.sales_2020_02, color=cols[i], markersize=5)
	end
	for (i,st) in enumerate(states)
		tmpdf = filter(row -> row.StateName == st, df_2020_02)	
		lines!(Ax_states,tmpdf.listings_2020_02, predict(modls0[i]), color=cols[i], linewidth=4, label=st)
		
	end
	Ax_states.xlabel = "Listings"
	Ax_states.ylabel = "Sales"
	Ax_states.limits = (0, 500, 0, 500)
	Ax_states.aspect=1
	axislegend(Ax_states, position=:lt)
	F_states
end	

# ╔═╡ b52c79ef-8fa4-4fc7-a292-b0d1914f8782
md"""
### Logistic Regression

Often times we need to predict whether a variable is either 1 or 0. An example of this is predicting if we will observe a phytoplankton taxon in a given sample. 

Let's see a quick example of what can go wrong with regular regression.
"""

# ╔═╡ d121d7f1-384f-485e-b4de-5ee0fdcbd778
begin
	bin_df = DataFrame(X=collect(1:7), Y=[1,0,1,1,1,1,1]);
	linear_bin_reg = lm(@formula(Y~X), bin_df)
	F_bin = Figure();
	Ax_bin = Axis(F_bin[1,1]);
	scatter!(Ax_bin, bin_df.X, bin_df.Y, markersize=10)
	lines!(1:7, predict(linear_bin_reg),color=:red)
	F_bin
end

# ╔═╡ 4e20326d-c30d-47ec-85bd-caed67a28005
md"""
As we can see, this performs badly. In particular it may predict values outside of `[0,1]`. We need to use **Logistic Regression**. 

The `GLM` package allows us to take advantage of tools like this, which are classified as **Generalised Linear Models**. We specify a `distribution family` and `link` function. 

For logistic regression these are `Binomial()` and `LogitLink()` respectively.
"""

# ╔═╡ 39c8db92-67c2-4ba0-88b8-cdac74af8c89
catsdf = dataset("MASS", "cats")

# ╔═╡ 6473efb0-8e52-47a4-bd1d-e05f03b97a43
md"""
Let's map the gender of each cat to a binary 0/1 value. We do this in the usual way by using `labelencoce` and `labelmap`.
"""

# ╔═╡ f38052d6-9580-4250-b592-da3e97e5f5da
begin
	lmap = labelmap(catsdf.Sex)
	catsdf.SexBinary = labelencode(lmap, catsdf.Sex).-1
	F_bincat = Figure()
	Ax_bincat = Axis(F_bincat[1,1])
	scatter!(Ax_bincat,catsdf.BWt, catsdf.HWt, color=catsdf.SexBinary)
	F_bincat
end

# ╔═╡ 88e7be82-79d1-4eca-bedd-77277df17c07
md"""
I am not sure why I cannot apply a legend properly in the above plot. Using `Plots.jl` I can filter by colour and add a label for each color factor. In `Makie` it appears I cannot do this, I can only set `label` to be a `String` and not a `Vector{String}`.

I can alternatively loop through the options in `catsdf.SexBinary` to add to the plot and add a label individually but I feel this should be something that is automatic. I was liking `Makie` up-to this point but I feel that this is a failing point. Combined with the lack of `alpha` parameter in plots I may now return to `Plots.jl`. Ho hum. Admittedly, I do enjoy that I can sequentially add plots to a figure in different Pluto (and I assume Jupyter) cells, but I am not sure that this makes up for these limitations. 

Female cats (blue) are more dominant in the lower left corner.
"""

# ╔═╡ 9fd67717-2d81-462f-882b-f813e4fd3601
probit = glm(@formula(SexBinary ~ HWt), catsdf, Binomial(), LogitLink())

# ╔═╡ 24197300-6968-4482-bd35-2bf65c9725e0
begin
	F_cat_pred = Figure()
	A_cat_pred = Axis(F_cat_pred[1,1])
	scatter!(A_cat_pred,catsdf.HWt,catsdf.SexBinary, color=:blue,label="ground truth")
	scatter!(A_cat_pred,catsdf.HWt,predict(probit), color=:red,label="prediction")
	axislegend()
	F_cat_pred
end

# ╔═╡ 1e3489bc-58bb-4968-bc62-ae667c41fce5
md"""
We see that our results do not go outside of `[0,1]`.

### Non-Linear Regression

We can also fit non-linear data. We use a function to fi our observables. This could be quadratic, or sinosoidal function. We minimise the least-square-error between fitted function and our observables.

We use the `LsqFit` package for this task.

Let's generate some data.
"""

# ╔═╡ d693ba8f-dd05-4824-ba6b-2629ed3a4ac0
begin
	F_nl = Figure()
	Ax_nl = Axis(F_nl[1,1])
	x_vals = 0:0.1:10
	y_vals = 1*exp.(-x_vals*2) + 2*sin.(0.8*pi*x_vals) + 0.15 * randn(length(x_vals));
	scatter!(Ax_nl, x_vals, y_vals)
	F_nl
end

# ╔═╡ 82bae0e3-0936-4b9b-8977-3e4e7fcaf9f9
md"""
We set up our model. Lets start with

$$p_1 \text{e}^{-x p_2} + p_3 \sin(0.8\pi x)$$
"""

# ╔═╡ 69551038-272f-498d-866f-0c77e5226de6
@. model(x,p) = p[1]*exp(-x*p[2]) + p[3]*sin(p[4]*pi*x)

# ╔═╡ 289a877c-9397-442f-96a5-53db2ebb4952
begin
	p0 = [0.5,0.5,0.5,0.7];
	fit1 = curve_fit(model, x_vals, y_vals, p0)
end

# ╔═╡ 6b6fc5ce-696c-4be3-858d-ee874bd9ec37


# ╔═╡ 5859cecd-fa99-41cc-b358-4e40169f62fb
md"""
⚠️ A note about `curve_fit`: this function can take multiple other inputs, for instance the Jacobian of what you are trying to fit. We don't dive into these details here, but be sure to check out the `LsqFit` package to see what other things you can can pass to create a better fit.

Also note that julia has multiple packages that allow you to create Jacobians so you don't have to write them yourself. Two such packages are `FiniteDifferences` or `ForwardDiff`.

↪️ Back to our example. We are ready now to plot the curve we have generated
"""

# ╔═╡ a0b2c058-dfe9-4885-aaf1-3745d0614230
begin
	p1 = fit1.param
	y_vals_fit1 = p1[1]*exp.(-x_vals*p1[2]) + p1[3]*sin.(p1[4]*pi*x_vals);
	lines!(Ax_nl, x_vals, y_vals_fit1, label="Fit 1",color=:red)
	axislegend(Ax_nl)
	F_nl
end

# ╔═╡ d8f0fcee-d997-410c-805f-2f19d68fcd13
md"""
Let's fit a linear function for fun!
"""

# ╔═╡ d5bed83d-4785-42ad-9754-321c4650b176
@. model2(x,p) = p[1]*x + p[2]

# ╔═╡ 57be51c7-4951-49e0-b585-f81e4750f3fb
fit2 = curve_fit(model2, x_vals, y_vals, [0.5, 0.5])

# ╔═╡ e269d0d3-c201-4ec3-86e7-719600012abe
begin
	p2 = fit2.param
	y_vals_fit2 = p2[1]*x_vals .+ p2[2];
	lines!(Ax_nl, x_vals, y_vals_fit2, label="Fit 2",color=:green)
	axislegend(Ax_nl)
	F_nl
end

# ╔═╡ dfcc41c5-2b6d-4e8f-81bb-318e3aece6b4
md"""
### Summary

We can use `GLM` and `LsqFit` to perform various different types of regression. 

`GLM` allows us to perform linear and generalised linear regressions, we can assign distributions and links to these models in order to capture binary values.

`LsqFit` can allow us to fit non-linear funtions to our data. We need to prescribe a `model` using `@.`. 

For the data we have played with today, we can see that the number of listings can be used to predict the number of sales in a given region. Different states have different sold/listed ratios, and in the results I looked at we can see that Colorado has the best ratio.

We only looked at Feb 2020 for the housing data. We could simply stack our data and try this again for the full data.

I will start a second notebook to test this out!
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LsqFit = "2fda8390-95c7-5789-9bda-21331edee243"
MLBase = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
RDatasets = "ce6b1742-4840-55fa-b093-852dadbb1d8b"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[compat]
CairoMakie = "~0.6.5"
DataFrames = "~1.2.2"
GLM = "~1.5.1"
LsqFit = "~0.12.1"
MLBase = "~0.9.0"
RDatasets = "~0.7.5"
StatsBase = "~0.33.10"
Tables = "~1.6.0"
XLSX = "~0.7.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "b8d49c34c3da35f220e7295659cd0bab8e739fed"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.33"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Automa]]
deps = ["Printf", "ScanByte", "TranscodingStreams"]
git-tree-sha1 = "d50976f217489ce799e366d9561d56a98a30d7fe"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "0.8.2"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CSV]]
deps = ["Dates", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode"]
git-tree-sha1 = "b83aa3f513be680454437a0eee21001607e5d983"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.8.5"

[[Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[CairoMakie]]
deps = ["Base64", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "SHA", "StaticArrays"]
git-tree-sha1 = "8664989955daccc90002629aa80193e44893bb45"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.6.5"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "fbc5c413a005abdeeb50ad0e54d85d000a1ca667"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "a325370b9dd0e6bf5656a6f1a7ae80755f8ccc46"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.7.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "45efb332df2e86f2cb2e992239b6267d97c9e0b6"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.7"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "31d0151f5716b655421d9d75b7fa74cc4e744df2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.39.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "7220bc21c33e990c14f4a9a319b1d242ebc5b269"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.3.1"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "ff7890c74e2eaffbc0b3741811e3816e64b6343d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.18"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "8041575f021cba5a099a456b4163c9a08b566a02"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[EzXML]]
deps = ["Printf", "XML2_jll"]
git-tree-sha1 = "0fa3b52a04a4e210aeb1626def9c90df3ae65268"
uuid = "8f5d6c58-4d21-5cfd-889c-e3ad7ee6a615"
version = "1.1.0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "3c041d2ac0a52a12a27af2782b34900d9c3ee68c"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.1"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "29890dfbc427afa59598b8cfcc10034719bd7744"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.6"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "8b3c09b56acaf3c0e581c66638b85c8650ee9dca"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.8.1"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "c4203b60d37059462af370c4f3108fb5d155ff13"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.20"

[[FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "StaticArrays"]
git-tree-sha1 = "19d0f1e234c13bbfd75258e55c52aa1d876115f5"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.2"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "f564ce4af5e79bb88ff1f4488e64363487674278"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.5.1"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Match", "Observables"]
git-tree-sha1 = "e2f606c87d09d5187bb6069dab8cee0af7c77bdb"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.6.1"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "9a5c62f231e5bba35695a20988fc7cd6de7eeb5a"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.3"

[[ImageIO]]
deps = ["FileIO", "Netpbm", "OpenEXR", "PNGFiles", "TiffImages", "UUIDs"]
git-tree-sha1 = "13c826abd23931d909e4c5538643d9691f62a617"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.5.8"

[[Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "34dc30f868e368f8a17b728a1238f3fcda43931a"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.3"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LsqFit]]
deps = ["Distributions", "ForwardDiff", "LinearAlgebra", "NLSolversBase", "OptimBase", "Random", "StatsBase"]
git-tree-sha1 = "91aa1442e63a77f101aff01dec5a821a17f43922"
uuid = "2fda8390-95c7-5789-9bda-21331edee243"
version = "0.12.1"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MLBase]]
deps = ["IterTools", "Random", "Reexport", "StatsBase"]
git-tree-sha1 = "3bd9fd4baf19dfc1edf344bc578da7f565da2e18"
uuid = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
version = "0.9.0"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[Makie]]
deps = ["Animations", "Base64", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Distributions", "DocStringExtensions", "FFMPEG", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MakieCore", "Markdown", "Match", "MathTeXEngine", "Observables", "Packing", "PlotUtils", "PolygonOps", "Printf", "Random", "RelocatableFolders", "Serialization", "Showoff", "SignedDistanceFields", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "UnicodeFun"]
git-tree-sha1 = "7e49f989e7c7f50fe55bd92d45329c9cf3f2583d"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.15.2"

[[MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "7bcc8323fb37523a6a51ade2234eee27a11114c8"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.1.3"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[Match]]
git-tree-sha1 = "5cf525d97caf86d29307150fcba763a64eaa9cbe"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "1.1.0"

[[MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "Test"]
git-tree-sha1 = "70e733037bbf02d691e78f95171a1fa08cdc6332"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.2.1"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Mocking]]
deps = ["Compat", "ExprTools"]
git-tree-sha1 = "29714d0a7a8083bba8427a4fbfb00a540c681ce7"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.3"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "144bab5b1443545bc4e791536c9f1eacb4eed06a"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.1"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0e9e582987d36d5a61e650e6e543b9e44d9914b"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.7"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OptimBase]]
deps = ["NLSolversBase", "Printf", "Reexport"]
git-tree-sha1 = "9cb1fee807b599b5f803809e85c81b582d2009d6"
uuid = "87e2bd06-a317-5318-96d9-3ecbac512eee"
version = "2.0.2"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "e14c485f6beee0c7a8dcf6128bf70b85f1fe201e"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.9"

[[Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "1155f6f937fa2b94104162f01fa400e192e4272f"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.4.2"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "646eed6f6a5d8df6708f15ea7e02a7a2c4fe4800"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.10"

[[Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9bc1871464b12ed19297fbc56c4fb4ba84988b0d"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.47.0+0"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "bfd7d8c7fd87f04543810d9cbd3995972236ba1b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "69fd065725ee69950f3f58eceb6d144ce32d627d"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[RData]]
deps = ["CategoricalArrays", "CodecZlib", "DataFrames", "Dates", "FileIO", "Requires", "TimeZones", "Unicode"]
git-tree-sha1 = "19e47a495dfb7240eb44dc6971d660f7e4244a72"
uuid = "df47a6cb-8c03-5eed-afd8-b6050d6c41da"
version = "0.8.3"

[[RDatasets]]
deps = ["CSV", "CodecZlib", "DataFrames", "FileIO", "Printf", "RData", "Reexport"]
git-tree-sha1 = "06d4da8e540edb0314e88235b2e8f0429404fdb7"
uuid = "ce6b1742-4840-55fa-b093-852dadbb1d8b"
version = "0.7.5"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "9a4b7698b59b24003e8475df70c1b83b958b1f62"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.1"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SIMD]]
git-tree-sha1 = "9ba33637b24341aba594a2783a502760aa0bff04"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.3.1"

[[ScanByte]]
deps = ["Libdl", "SIMD"]
git-tree-sha1 = "9cc2955f2a254b18be655a4ee70bc4031b2b189e"
uuid = "7b38b023-a4d7-4c5e-8d43-3f3097f304eb"
version = "0.3.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "54f37736d8934a12a200edea2f9206b03bdf3159"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.7"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "793793f1df98e3d7d554b65a107e9c9a6399a6ed"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.7.0"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "a8f30abc7c64a39d389680b74e749cf33f872a70"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.3.3"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "95072ef1a22b057b1e80f73c2a89ad238ae4cfff"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.12"

[[StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "1bc8cc83e458c8a5036ec7206a04d749b9729fe8"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.26"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TiffImages]]
deps = ["ColorTypes", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "OrderedCollections", "PkgVersion", "ProgressMeter"]
git-tree-sha1 = "945b8d87c5e8d5e34e6207ee15edb9d11ae44716"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.4.3"

[[TimeZones]]
deps = ["Dates", "Future", "LazyArtifacts", "Mocking", "Pkg", "Printf", "RecipesBase", "Serialization", "Unicode"]
git-tree-sha1 = "6c9040665b2da00d30143261aea22c7427aada1c"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.5.7"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "59e2ad8fd1591ea019a5259bd012d7aee15f995c"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.3"

[[XLSX]]
deps = ["Dates", "EzXML", "Printf", "Tables", "ZipFile"]
git-tree-sha1 = "96d05d01d6657583a22410e3ba416c75c72d6e1d"
uuid = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"
version = "0.7.8"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "3593e69e469d2111389a9bd06bac1f3d730ac6de"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.4"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[isoband_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "a1ac99674715995a536bbce674b068ec1b7d893d"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.2+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─e5115c3a-25cd-11ec-1d64-67ddeae7e0f7
# ╠═1581f292-1d72-40ab-8126-d48a482f11d1
# ╟─ebba71b6-497f-4e7b-b68e-2f90dcdbbd72
# ╠═0682072e-a9f6-453c-8e2a-c8793d6c7579
# ╠═e2fcc903-cd21-4fbf-9da1-4ffa4747080a
# ╟─67ce0bab-6fd6-44fb-9a10-4aa4e14f63cd
# ╠═5c3b5507-4520-484d-93e4-5d3e87a6a03d
# ╠═a253c672-93be-494f-9677-f8b0e799b653
# ╠═df7dbef2-2a1b-4c4c-8b5a-75c8ca3a9c85
# ╠═f9dd21a1-560a-4903-b8dd-869ad3ebc8bf
# ╟─753113d8-56a0-42ca-805c-d30d899087c0
# ╠═6f52bfa8-fc97-442c-b0ad-860c271f686f
# ╠═c055cc56-778f-42df-b11a-8e56d88cba1b
# ╠═b1e4efe8-ae39-4033-9e6a-3c7f05fc4e0b
# ╟─cee7352b-c55f-4220-a8f9-19bb45bf39bf
# ╠═02fd8825-9178-412c-98be-938d71c599aa
# ╟─32928d1b-d6f9-44d6-b68d-8a2c9c239c07
# ╠═98531e6c-b365-4562-b108-3218b35af9b1
# ╠═1d31819a-21ca-4d54-a0ac-062de0036da0
# ╟─f8885a1f-90c1-44eb-ae3e-04b5d1997d5d
# ╠═116335a3-892c-4296-90bb-f044589bc0b3
# ╠═9d04b5d0-ea66-4adb-920b-1f59b58328fa
# ╠═38f65135-cc7c-4165-9a5f-c6d1f1fe92ff
# ╟─0be742f8-c4d8-4e5b-8383-06993dade831
# ╠═9b7495b2-a3f3-47d9-81a3-1a2814a9c5ab
# ╟─6037bf5b-53ab-482f-97ae-f8ca1441b97d
# ╠═c994a0e0-16dc-4c9d-b3aa-8432681b8d08
# ╠═b39ded5d-c480-4e34-907a-95eba07cd43f
# ╠═0c2b2e28-7854-400e-b85d-0ae950daa84a
# ╠═1493dd7c-667f-42eb-a326-5e159f958262
# ╠═ced60d91-ad1e-43d1-8187-6034140e9e6b
# ╟─b52c79ef-8fa4-4fc7-a292-b0d1914f8782
# ╠═d121d7f1-384f-485e-b4de-5ee0fdcbd778
# ╟─4e20326d-c30d-47ec-85bd-caed67a28005
# ╠═39c8db92-67c2-4ba0-88b8-cdac74af8c89
# ╟─6473efb0-8e52-47a4-bd1d-e05f03b97a43
# ╠═f38052d6-9580-4250-b592-da3e97e5f5da
# ╟─88e7be82-79d1-4eca-bedd-77277df17c07
# ╠═9fd67717-2d81-462f-882b-f813e4fd3601
# ╠═24197300-6968-4482-bd35-2bf65c9725e0
# ╟─1e3489bc-58bb-4968-bc62-ae667c41fce5
# ╠═d693ba8f-dd05-4824-ba6b-2629ed3a4ac0
# ╟─82bae0e3-0936-4b9b-8977-3e4e7fcaf9f9
# ╠═69551038-272f-498d-866f-0c77e5226de6
# ╠═289a877c-9397-442f-96a5-53db2ebb4952
# ╠═6b6fc5ce-696c-4be3-858d-ee874bd9ec37
# ╟─5859cecd-fa99-41cc-b358-4e40169f62fb
# ╠═a0b2c058-dfe9-4885-aaf1-3745d0614230
# ╟─d8f0fcee-d997-410c-805f-2f19d68fcd13
# ╠═d5bed83d-4785-42ad-9754-321c4650b176
# ╠═57be51c7-4951-49e0-b585-f81e4750f3fb
# ╠═e269d0d3-c201-4ec3-86e7-719600012abe
# ╠═dfcc41c5-2b6d-4e8f-81bb-318e3aece6b4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
