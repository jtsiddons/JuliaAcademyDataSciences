### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 56ee7d38-1fae-11ec-18e1-598efc676c2a
using DataFrames, DelimitedFiles,CSV, XLSX, HTTP, BenchmarkTools

# ╔═╡ fdf86734-31de-4c80-a410-3663a01b0b92
begin
	using JLD
	jld_data = JLD.load("data/mytempdata.jld") # Load data - get a dictionary
	save("data/mywrite.jld", jld_data) # Save data
end

# ╔═╡ 5f36a2eb-c1d0-4c0c-95a2-2fb9b6d8b205
begin
	using RData
	R_data = RData.load("data/mytempdata.rda")
	# Save
	using RCall
	@rput R_data
	R"save(R_data, file=\"data/mydata.rda\")"
end	

# ╔═╡ 3bb3a1e1-32eb-4e58-846e-b8931e9d9fb5
md"""
# Julia Academy Data Sciences Course

## 01: Data

This is a rewrite of 01 Data.ipynb into Pluto using more up-to-date approaches - for example, CSV.read is no longer recommended in favour of CSV.File & then piping into (or directly enclosing in) DataFrame.

First, load the required packages 😄
"""

# ╔═╡ 0f1e3f32-cdee-4678-9e7d-0b8effd3bf8d
md"""
### Downloading Data

One can use the `download` function, but I prefer `HTTP.download`. This stores it in a temporary file

This is getting data containing programming languages.

I can also use `curl` or `wget` in the terminal - I can run a shell command by leading with `;` in Julia command.
"""

# ╔═╡ 4a3dbf72-3662-45b8-9f0c-7d9b7fa92ebf
begin
	url = "https://raw.githubusercontent.com/nassarhuda/easy_data/master/programming_languages.csv";
	r = HTTP.download(url, "./data/programming_languages.csv")
end

# ╔═╡ c57cad3c-4010-492f-8ad2-4daa49fd8792
md"""
### Reading Data

There are a number of ways to make use of the downloaded file. Here we have a `.csv` file called $r. 

I like to use `CSV` and `XLSX` packages - depending upon the file type.

#### Reading Delimited Files

The first approach in the course uses `DelimitedFiles`. We use the `readdlm` function to read it in, this returns 2 matrices, the data and a header. Note that the delimiter needs to be in single quotes! - An `AbstractChar`.
"""

# ╔═╡ 2dd1e88f-6b74-4330-b9fd-60231f3be976
P,H = readdlm("programming_languages.csv",',',header=true);

# ╔═╡ 883fee6d-03f5-49d4-aa19-9e85e3ab2480
H

# ╔═╡ 9c91446f-199f-4ee3-adc2-00cd8ebe1177
P

# ╔═╡ 6676ac39-3873-48a1-8485-b4fa351de8dd
md"""
This data contains programming languages and the year in which they were first introduced/developed.

#### Writing data

We can use `writedlm` to write delimited data. Note that a header is not included.
"""

# ╔═╡ e345c260-1053-487c-a286-ca5416255ac5
writedlm("./data/programming_languages_dlm.txt", P, '|')

# ╔═╡ 86271992-7c69-4b7a-9b97-236b94de9a8e
readdlm("./data/programming_languages_dlm.txt",'|')

# ╔═╡ 6f7b8cfc-9534-4ba2-8b18-4d2dd0af8a80
md"""

Delimited files is a powerful tool if you have more complicated delimited files. 

#### Using CSV

The course suggests `CSV.read`. However, this now requires a `sink`. In the example that the course runs it automatically gets piped into a `DataFrame`. Now I need to specify output. Personally I prefer `DataFrame(CSV.File(source))`, this is the currently recommended approach.
"""

# ╔═╡ 4810edfb-6c3a-4f01-98ef-1bdb14b82f46
C = CSV.read(r,DataFrame)

# ╔═╡ a0874d30-16ca-46e3-aef0-0b0d89974dc4
Ctype = typeof(C)

# ╔═╡ 1ca1b028-0396-446a-b0a1-394878d94ea3
md"""
We can see that `C` is a $Ctype.
"""

# ╔═╡ 7459a262-7463-4b97-a709-8ef1ccbe2738
md"""
We can access parts of the DataFrame by using C[`rows`, `columns`]. We can specify indices or columnames.
"""

# ╔═╡ b9d6be82-d0ec-419b-b626-eea82c525b00
C[1:10,:]

# ╔═╡ f4e090ce-6e86-4d4f-b3ed-71128d3af3f5
C[:,:year]

# ╔═╡ ed0d283b-ad30-40e3-9e6e-de577872a92a
C.language

# ╔═╡ 5a20250b-dfc8-481a-9d5f-59ee41fdf550
md"""
We can see how quick various commands are - let us see if it is quicker to use `readdlm` or `CSV.read`. As you can see there is very little difference. If anything, doing `CSV.read()` or `DataFrame(CSV.File())` is quicker as they required conversion to a DataFrame.
"""

# ╔═╡ 07a72053-5fb8-41e4-9e3b-e0cd50d87401
@btime P,H = readdlm(r, ','; header=true);

# ╔═╡ 462387d1-0283-4aed-aa18-a533671329fd
@btime C = CSV.read(r, DataFrame);

# ╔═╡ 0a2d9e5c-bc4a-4bd9-8b11-1712d7084acd
@btime df = DataFrame(CSV.File(r));

# ╔═╡ 4d74a5d9-5de7-4163-bd6e-56cedea97bf9
md"""
#### Writing CSV data
"""

# ╔═╡ edc9882b-3ece-40c0-97f4-001f289cbb47
CSV.write("./data/programming_languages_CSV.csv", C)

# ╔═╡ 3b126c88-cbc1-4a6e-9cc8-89b768ba45d3
md"""
### Looking at Data

We now have our DataFrame `C`. We can show some summary information quite easily.

Below, I get the columnnames, a summary of the data, and the size of the DataFrame.
"""

# ╔═╡ a4bf4723-a772-46f9-bb1b-d2fa0eb98d65
[names(C), describe(C), size(C)]

# ╔═╡ 0b304a41-55f0-42e5-96db-6c75a544fbb1
md"""
### XLSX Data

We often have to deal with `xls` or `xlsx` files. We can use the `XLSX` package and the `readdata` function to open a file, select a specific sheet and cell range
"""

# ╔═╡ e59fdb69-51a7-469f-97b6-732e8a6d9d56
T = XLSX.readdata(
	"./data/zillow_data_download_april2020.xlsx", # file name
	"Sale_counts_city", # sheet name
	"A1:F9" # cell range
	)

# ╔═╡ 27509dfd-af6c-4221-af9d-8492d543bdf2
md"""
If we don't know the cell range, we can use `readtable`. This will automatically detect what it thinks is the end of the table, which can miss data. It returns a `tuple` containing the data and headers. In contrast `readdata` returns a matrix - which includes the headers.

If you know cell ranges it is likely to be more effective.

`readtable` takes longer as it is a large file & `readtable` is looking for e.o.f.
"""

# ╔═╡ 9a1be4d7-0f7b-4e8e-8e18-dc7b919f8d40
G  = XLSX.readtable("./data/zillow_data_download_april2020.xlsx", "Sale_counts_city")

# ╔═╡ d023e914-41a6-422c-8522-afc1f16539bc
md"""
This can be parsed into a `DataFrame` by appending `...` to the variable name - in our case `G`. So we would call `DataFrame(G...)`.

The `...` expands out the tuple, it is equivalent to `DataFrame(G[1], G[2])`. 

The `...` operator is also known as the `splat` operator. This cannot be called outside of a function such as `DataFrame`.
"""

# ╔═╡ f27523e1-e39d-49f1-86d2-5ca184c69363
md"""
### DataFrames & DataFrame manipulations

Note that we have missing data in row 1. I will drop missing data when I transform to long format data
"""

# ╔═╡ 2c86b54e-91bd-461f-9fdd-744480b6d488
zilDF = DataFrame(G...)

# ╔═╡ 5594bb66-52f9-47d8-989a-4625be782b0d
[names(zilDF), describe(zilDF), size(zilDF)]

# ╔═╡ fd1f40ad-f78c-4bcb-8203-ba7a7723fa01
md"""
As you can see it is a very large dataset. There are 144 columns containing sales data over 28759 rows!

Looking at `G` which is the `readtable` output. We get a tuple which contains

* `G[1]` is the columns of the data - stored as an vector of vectors.
* `G[2]` is the column names - stored as a vector.
"""

# ╔═╡ c44dbb59-c60b-4c4d-91b3-a550c23e02e0
G[1][1]

# ╔═╡ cca73d57-c621-4fcd-9066-9b0ad06ad384
[typeof(G), typeof(G[1]), typeof(G[1][1]), typeof(G[2])]

# ╔═╡ 88cea1ea-6962-405e-987d-d9c116db37fd
md"""
We can use `combine` and `groupby` to get useful summaries of the data. For example, we can get the number of stores in each state.
"""

# ╔═╡ 3e241b8d-2dbd-4efd-ae88-f159b7b98cbd
combine(
	groupby(zilDF, :StateName),
	nrow
	)

# ╔═╡ a391a041-d165-4fc8-a371-afaded75129b
md"""
We can turn this into long format using `stack`
"""

# ╔═╡ 62a58f26-8f84-4795-8655-536acbad7b5b
# I don't know why this didn't load...
import DataFrames.stack

# ╔═╡ 33eabc62-52ec-42ae-af34-76ebdd4aed14
zilDFlong = stack(zilDF, 5:ncol(zilDF), variable_name=:Month, value_name=:Sales)

# ╔═╡ dbd74a99-ea2a-45bc-ac17-40e689b865d0
dropmissing!(zilDFlong)

# ╔═╡ 4b5423eb-280f-401d-b7e5-8f904422e1a9
zilDFlong_gp = combine(
	groupby(zilDFlong, [:StateName, :Month]),
	:Sales => sum => :SumSales)

# ╔═╡ 7d35f3b6-bd10-4789-83f0-dfa7facf9f86
zilSalesByStateMonth = unstack(zilDFlong_gp, :StateName, :Month, :SumSales)

# ╔═╡ 9bf02ce3-202e-4052-8bec-e06d199e66c9
md"""
#### Manually build DataFrame

We can manually build DataFrames.
"""

# ╔═╡ 2aead321-84e0-472f-98f7-ba90880ed90c
begin
	foods = ["apple", "cucumber", "tomato", "banana"]
	calories = [105, 47, 22, 105]
	df_calories = DataFrame(item=foods, calories=calories)
end

# ╔═╡ 9ac9eb62-c346-4b5c-b25b-01663af4817e
md"""
#### Joining DataFrames

In the past we could just use the `join` function. Now it is required that we use the necessary function for the type of join. In the example here, I use a `leftjoin` which will only select from the `right` DataFrame values which exist in the `left`. We can join on the right, inner or outer. Inner is the intersect & outer the union. Other joins can be found [here](https://dataframes.juliadata.org/stable/man/joins/)
"""

# ╔═╡ 8c883ad0-00a7-4d84-a349-2bb882e7a339
begin
	prices = [0.85, 1.60, 0.80, 0.60]
	df_prices = DataFrame(item=foods, prices=prices)
end

# ╔═╡ dcd3d619-c7d0-4c6c-a593-d331fe443b73
DF = leftjoin(df_calories, df_prices, on=:item)

# ╔═╡ 55b04783-75ad-43c3-b8ed-78faee1eb965
md"""
#### Writing DataFrame to XLSX

We can write DataFrames to XLSX too.
"""

# ╔═╡ 8549dd60-3dc0-4ca2-b32e-f18be1fb1849
XLSX.writetable("./data/joined_food_df.xlsx",DF)

# ╔═╡ a304a33c-bfab-4698-a602-3636041f33d1
XLSX.readdata("./data/joined_food_df.xlsx","Sheet1","A1:C5")

# ╔═╡ 6f4864ec-3a42-44ac-97e4-050336207ed6
md"""
We can also do it with matrices
"""

# ╔═╡ 111ad359-98ac-4c63-b1b6-05a97e5061ea
begin
	A = collect(eachcol(DF)) # Columns
	B = names(DF) # names
	[typeof(A), typeof(B)]
end

# ╔═╡ 9aad19ed-c164-4278-9c6e-fe9170f448b7
begin
	XLSX.writetable("./data/joined_food_df_2.xlsx",A, B) # Note columns, then names
	XLSX.readdata("./data/joined_food_df_2.xlsx","Sheet1","A1:C5")
end

# ╔═╡ 092a6bec-5327-42b9-8b9b-f4fa9206e0a6
md"""
### Importing and Exporting Data from other languages
"""

# ╔═╡ 96aaab25-e1e3-49f3-82cd-a136aa95bec7
md"""
Loading and saving Julia data 
"""

# ╔═╡ dc505f42-32b4-4dc5-8c67-f3a22f747c2d
jld_data

# ╔═╡ 002664ca-d0a6-4f54-b4d3-555d7cd3d5fe
JLD.load("./data/mywrite.jld")

# ╔═╡ f9972d4e-7180-474b-879a-d7d770302379
md"""
We can do the same for NPZ - which is numpy data. We need to load the `NPZ` package and run `npzread`. Saving is done with `npzwrite` in the same form as `save` for JLD.

Matlab requires `MAT`, `matread` and `matwrite`.

For R data, we need to use `RData` and `RData.load` to load data. Saving is more complicated, we need to load `RCall` and save using R language. We run R code in an `@rput`. The R command is in an `R`-string, quotes need to be escaped.
"""

# ╔═╡ e1326898-2a17-4eca-9eec-22ef1691f4f4
[typeof(jld_data), typeof(R_data)]

# ╔═╡ 28af9ba8-5a5b-480c-9bd3-709ed044abeb
md"""
### Processing Data in Julia

Mainly will focus on `Matrix`, `DataFrame`, and `Dict`. We return to the programming languages data, which we stored in `P` & `H` as matrices, as well as `C` as a DataFrame.

#### Some questions that we can answer.

* Which year was a given language invented?
* How many languages were created in a given year?

Something to notice. If we look at `C.languages` we notice that some of the strings have leading or following spaces. Let's remove those now.
"""

# ╔═╡ 2197c98b-e984-4e9d-a3bb-c9e419966fed
C.language = String.(lstrip.(rstrip.(C.language)))

# ╔═╡ 28e183f1-473b-4103-9228-d2bc1dab864d
# Q1
function year_created(lang::String)
	# Convert to lower case to have best chance of finding
	lang2 = lowercase(lang);
	loc = findfirst(lowercase.(C.language) .== lang2)
	# Let's handle the possible error of loc being nothing
	isnothing(loc) ? error("Error: Language $lang not found") : return C.year[loc]
end

# ╔═╡ 931ea6cc-cb78-4d0a-91a9-11a53fa4aa53
year_created("c++")

# ╔═╡ 81c75a8b-95a0-4c49-886e-cdcb5f9f6596
md"""
Q2, let's create a summary dataframe
"""

# ╔═╡ 536762dd-a04e-4366-8c1e-a936f3124c86
C_yearcounts = combine(
	groupby(C,:year), # group by year
	nrow => :nlangs # summarise by getting nrow for each element of group (year)
	)

# ╔═╡ df51a0ab-dbb8-4c3d-9776-dce45b6f4bf0
# Q2 - alternatively we could get a length of findall
function langs_in_year(yr::Int64)
	loc = findfirst(C_yearcounts.year .== yr);
	# Handle possible error of loc being nothing
	isnothing(loc) ? error("Error: No languages created in $yr") : return C_yearcounts.nlangs[loc]
end

# ╔═╡ 24b36ef0-63db-4f97-9daa-7b011dc9a97b
langs_in_year(2012)

# ╔═╡ bef93289-7a71-4b74-9eda-0d6e2fb1d070
md"""
### Dictionaries

First a quick example:
"""

# ╔═╡ 93105023-14dd-4b5f-ac71-420d4bd33223
Dict([("A",1), ("B", 2), ("j", [1, 4])])

# ╔═╡ 46d3e0a5-ebe9-4440-be8b-fe88f8cb85cf
P_dict = Dict()

# ╔═╡ e48f6eda-8758-49d1-aed1-a89274020421
P_dict[67] = ["julia", "programming"]

# ╔═╡ bc728671-0093-4093-a4c3-401c3192c8bc
P_dict["julia"] = 7

# ╔═╡ 16cbd5b7-dcbd-4a1d-b6a9-fa5bb110bbd0
P_dict

# ╔═╡ 8a40dc9d-6884-4cdd-a187-0949df21014a
md"""
#### Transforming `P` into a Dictionary

Let's see how we can transform `P` into a dictionary. I would like the key to be year and the value to be an array containing all languages in that year
"""

# ╔═╡ b3d70edd-beb9-43ed-b098-d0c7dcefac7e
dict = Dict{Integer,Vector{String}}()

# ╔═╡ 9c3efaab-e311-4754-8108-bf1ef2ba83d7
uyears = unique(P[:,1])

# ╔═╡ 70b12c1a-5901-4635-898a-22939bb48741
# Of course we can do this with DataFrame C too 😄
for uy in uyears
	dict[uy] = P[findall(P[:,1] .== uy),2]
end

# ╔═╡ 43832022-483a-4937-ade2-73749267449b
dict

# ╔═╡ 3b7e5788-6324-496a-aca0-e70b47075be8
dict[2012]

# ╔═╡ 853ed5b4-eb2a-4ef8-ab7f-9a473f1314ae
md"""
### Missing Data

We can quite often handle missing data with `dropmissing` or `dropmissing!` to do it inplace.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
JLD = "4138dd39-2aa7-5051-a626-17a0bb65d9c8"
RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
RData = "df47a6cb-8c03-5eed-afd8-b6050d6c41da"
XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[compat]
BenchmarkTools = "~1.2.0"
CSV = "~0.9.4"
DataFrames = "~1.2.2"
HTTP = "~0.9.15"
JLD = "~0.12.3"
RCall = "~0.13.12"
RData = "~0.8.3"
XLSX = "~0.7.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "61adeb0823084487000600ef8b1c00cc2474cd47"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.2.0"

[[Blosc]]
deps = ["Blosc_jll"]
git-tree-sha1 = "84cf7d0f8fd46ca6f1b3e0305b4b4a37afe50fd6"
uuid = "a74b3585-a348-5f62-a45c-50e91977d574"
version = "0.7.0"

[[Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "e747dac84f39c62aff6956651ec359686490134e"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.0+0"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "3a877c2fc5c9b88ed7259fd0bdb7691aad6b50dc"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.9.4"

[[CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "fbc5c413a005abdeeb50ad0e54d85d000a1ca667"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bd4afa1fdeec0c8b89dad3c6e92bc6e3b0fec9ce"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.6.0"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "31d0151f5716b655421d9d75b7fa74cc4e744df2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.39.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

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

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[EzXML]]
deps = ["Printf", "XML2_jll"]
git-tree-sha1 = "0fa3b52a04a4e210aeb1626def9c90df3ae65268"
uuid = "8f5d6c58-4d21-5cfd-889c-e3ad7ee6a615"
version = "1.1.0"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "3c041d2ac0a52a12a27af2782b34900d9c3ee68c"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.1"

[[FilePathsBase]]
deps = ["Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "6d4b609786127030d09e6b1ee0e2044ec20eb403"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.11"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[HDF5]]
deps = ["Blosc", "Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires"]
git-tree-sha1 = "83173193dc242ce4b037f0263a7cc45afb5a0b85"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.15.6"

[[HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "fd83fa0bde42e01952757f01149dd968c06c4dba"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.0+1"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "24675428ca27678f003414a98c9e473e45fe6a21"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.15"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLD]]
deps = ["FileIO", "HDF5", "Printf"]
git-tree-sha1 = "1d291ba1730de859903b480e6f85a0dc40c19dcb"
uuid = "4138dd39-2aa7-5051-a626-17a0bb65d9c8"
version = "0.12.3"

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

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

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

[[Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

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
deps = ["ExprTools"]
git-tree-sha1 = "748f6e1e4de814b101911e64cc12d83a6af66782"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.2"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

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

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "9d8c00ef7a8d110787ff6f170579846f776133a9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.4"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

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
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[RCall]]
deps = ["CategoricalArrays", "Conda", "DataFrames", "DataStructures", "Dates", "Libdl", "Missings", "REPL", "Random", "Requires", "StatsModels", "WinReg"]
git-tree-sha1 = "80a056277142a340e646beea0e213f9aecb99caa"
uuid = "6f49c342-dc21-5d91-9882-a32aef131414"
version = "0.13.12"

[[RData]]
deps = ["CategoricalArrays", "CodecZlib", "DataFrames", "Dates", "FileIO", "Requires", "TimeZones", "Unicode"]
git-tree-sha1 = "19e47a495dfb7240eb44dc6971d660f7e4244a72"
uuid = "df47a6cb-8c03-5eed-afd8-b6050d6c41da"
version = "0.8.3"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

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
deps = ["ChainRulesCore", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ad42c30a6204c74d264692e633133dcea0e8b14e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.2"

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
git-tree-sha1 = "46d7ccc7104860c38b11966dd1f72ff042f382e4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.10"

[[StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "1bc8cc83e458c8a5036ec7206a04d749b9729fe8"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.26"

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
git-tree-sha1 = "1162ce4a6c4b7e31e0e6b14486a6986951c73be9"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.2"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

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

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "80229be1f670524750d905f8fc8148e5a8c4537f"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.0"

[[WeakRefStrings]]
deps = ["DataAPI", "Parsers"]
git-tree-sha1 = "4a4cfb1ae5f26202db4f0320ac9344b3372136b0"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.3.0"

[[WinReg]]
deps = ["Test"]
git-tree-sha1 = "808380e0a0483e134081cc54150be4177959b5f4"
uuid = "1b915085-20d7-51cf-bf83-8f477d6f5128"
version = "0.3.1"

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

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "3593e69e469d2111389a9bd06bac1f3d730ac6de"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.4"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═3bb3a1e1-32eb-4e58-846e-b8931e9d9fb5
# ╠═56ee7d38-1fae-11ec-18e1-598efc676c2a
# ╠═0f1e3f32-cdee-4678-9e7d-0b8effd3bf8d
# ╠═4a3dbf72-3662-45b8-9f0c-7d9b7fa92ebf
# ╠═c57cad3c-4010-492f-8ad2-4daa49fd8792
# ╠═2dd1e88f-6b74-4330-b9fd-60231f3be976
# ╠═883fee6d-03f5-49d4-aa19-9e85e3ab2480
# ╠═9c91446f-199f-4ee3-adc2-00cd8ebe1177
# ╠═6676ac39-3873-48a1-8485-b4fa351de8dd
# ╠═e345c260-1053-487c-a286-ca5416255ac5
# ╠═86271992-7c69-4b7a-9b97-236b94de9a8e
# ╠═6f7b8cfc-9534-4ba2-8b18-4d2dd0af8a80
# ╠═4810edfb-6c3a-4f01-98ef-1bdb14b82f46
# ╠═1ca1b028-0396-446a-b0a1-394878d94ea3
# ╠═a0874d30-16ca-46e3-aef0-0b0d89974dc4
# ╠═7459a262-7463-4b97-a709-8ef1ccbe2738
# ╠═b9d6be82-d0ec-419b-b626-eea82c525b00
# ╠═f4e090ce-6e86-4d4f-b3ed-71128d3af3f5
# ╠═ed0d283b-ad30-40e3-9e6e-de577872a92a
# ╠═5a20250b-dfc8-481a-9d5f-59ee41fdf550
# ╠═07a72053-5fb8-41e4-9e3b-e0cd50d87401
# ╠═462387d1-0283-4aed-aa18-a533671329fd
# ╠═0a2d9e5c-bc4a-4bd9-8b11-1712d7084acd
# ╠═4d74a5d9-5de7-4163-bd6e-56cedea97bf9
# ╠═edc9882b-3ece-40c0-97f4-001f289cbb47
# ╠═3b126c88-cbc1-4a6e-9cc8-89b768ba45d3
# ╠═a4bf4723-a772-46f9-bb1b-d2fa0eb98d65
# ╠═0b304a41-55f0-42e5-96db-6c75a544fbb1
# ╠═e59fdb69-51a7-469f-97b6-732e8a6d9d56
# ╠═27509dfd-af6c-4221-af9d-8492d543bdf2
# ╠═9a1be4d7-0f7b-4e8e-8e18-dc7b919f8d40
# ╠═d023e914-41a6-422c-8522-afc1f16539bc
# ╠═f27523e1-e39d-49f1-86d2-5ca184c69363
# ╠═2c86b54e-91bd-461f-9fdd-744480b6d488
# ╠═5594bb66-52f9-47d8-989a-4625be782b0d
# ╠═fd1f40ad-f78c-4bcb-8203-ba7a7723fa01
# ╠═c44dbb59-c60b-4c4d-91b3-a550c23e02e0
# ╠═cca73d57-c621-4fcd-9066-9b0ad06ad384
# ╠═88cea1ea-6962-405e-987d-d9c116db37fd
# ╠═3e241b8d-2dbd-4efd-ae88-f159b7b98cbd
# ╠═a391a041-d165-4fc8-a371-afaded75129b
# ╠═62a58f26-8f84-4795-8655-536acbad7b5b
# ╠═33eabc62-52ec-42ae-af34-76ebdd4aed14
# ╠═dbd74a99-ea2a-45bc-ac17-40e689b865d0
# ╠═4b5423eb-280f-401d-b7e5-8f904422e1a9
# ╠═7d35f3b6-bd10-4789-83f0-dfa7facf9f86
# ╠═9bf02ce3-202e-4052-8bec-e06d199e66c9
# ╠═2aead321-84e0-472f-98f7-ba90880ed90c
# ╠═9ac9eb62-c346-4b5c-b25b-01663af4817e
# ╠═8c883ad0-00a7-4d84-a349-2bb882e7a339
# ╠═dcd3d619-c7d0-4c6c-a593-d331fe443b73
# ╠═55b04783-75ad-43c3-b8ed-78faee1eb965
# ╠═8549dd60-3dc0-4ca2-b32e-f18be1fb1849
# ╠═a304a33c-bfab-4698-a602-3636041f33d1
# ╠═6f4864ec-3a42-44ac-97e4-050336207ed6
# ╠═111ad359-98ac-4c63-b1b6-05a97e5061ea
# ╠═9aad19ed-c164-4278-9c6e-fe9170f448b7
# ╠═092a6bec-5327-42b9-8b9b-f4fa9206e0a6
# ╠═96aaab25-e1e3-49f3-82cd-a136aa95bec7
# ╠═fdf86734-31de-4c80-a410-3663a01b0b92
# ╠═dc505f42-32b4-4dc5-8c67-f3a22f747c2d
# ╠═002664ca-d0a6-4f54-b4d3-555d7cd3d5fe
# ╠═f9972d4e-7180-474b-879a-d7d770302379
# ╠═5f36a2eb-c1d0-4c0c-95a2-2fb9b6d8b205
# ╠═e1326898-2a17-4eca-9eec-22ef1691f4f4
# ╠═28af9ba8-5a5b-480c-9bd3-709ed044abeb
# ╠═2197c98b-e984-4e9d-a3bb-c9e419966fed
# ╠═28e183f1-473b-4103-9228-d2bc1dab864d
# ╠═931ea6cc-cb78-4d0a-91a9-11a53fa4aa53
# ╠═81c75a8b-95a0-4c49-886e-cdcb5f9f6596
# ╠═536762dd-a04e-4366-8c1e-a936f3124c86
# ╠═df51a0ab-dbb8-4c3d-9776-dce45b6f4bf0
# ╠═24b36ef0-63db-4f97-9daa-7b011dc9a97b
# ╠═bef93289-7a71-4b74-9eda-0d6e2fb1d070
# ╠═93105023-14dd-4b5f-ac71-420d4bd33223
# ╠═46d3e0a5-ebe9-4440-be8b-fe88f8cb85cf
# ╠═e48f6eda-8758-49d1-aed1-a89274020421
# ╠═bc728671-0093-4093-a4c3-401c3192c8bc
# ╠═16cbd5b7-dcbd-4a1d-b6a9-fa5bb110bbd0
# ╠═8a40dc9d-6884-4cdd-a187-0949df21014a
# ╠═b3d70edd-beb9-43ed-b098-d0c7dcefac7e
# ╠═9c3efaab-e311-4754-8108-bf1ef2ba83d7
# ╠═70b12c1a-5901-4635-898a-22939bb48741
# ╠═43832022-483a-4937-ade2-73749267449b
# ╠═3b7e5788-6324-496a-aca0-e70b47075be8
# ╠═853ed5b4-eb2a-4ef8-ab7f-9a473f1314ae
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
