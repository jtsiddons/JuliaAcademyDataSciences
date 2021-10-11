### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 41a65105-f9c3-4e12-a70d-10d885ec2696
using DataFrames, CSV, JSON, Clustering, VegaLite, VegaDatasets, Statistics, Distances, HTTP

# ╔═╡ 4cd10f86-2450-11ec-3a70-0bb30b3fb525
md"""
# Julia Academy Data Sciences Course

## 05. Clustering

This is a write-up of the 5th lecture of the Julia Academy Data Sciences - Clustering.

This is the task of grouping samples/observations which are similar in some respect. In the last class we saw that the `mtcars` dataset resulted in seperation of the observations into 3 groups after application of three dimension reduction algorithms. 

Let's start by loading in the necessary packages.
"""

# ╔═╡ 367f19de-2587-4b6b-ae51-014bc60c59fc
md"""
### Housing Data in California

We load in housing data from California, we will then look to see if housing prices correlate with map location within California.
"""

# ╔═╡ d3cfdd07-6803-412a-bb50-bb70613ac9fa
begin
	loc = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv";
	housedf = CSV.File(HTTP.download(loc)) |> DataFrame
end

# ╔═╡ 2d4af172-a17f-42da-9572-b51ee72a1f1b
describe(housedf)

# ╔═╡ 140ec774-f1bd-479d-984a-91c42064ac5b
md"""
### Map plotting using `VegaLite`

We will use the [`VegaLite`](https://www.queryverse.org/VegaLite.jl/stable/) package for plotting in this write-up. It is essentially a _grammar of graphics_ suite. 

To plot the map of California, we need to load in a JSON file of the map which we draw. There is a JSON file in the `data` sub-directory of this project. 
"""

# ╔═╡ 66ba9bf8-f235-4242-88b0-118710d20ead
cali_shape = JSON.parsefile("./data/california-counties.json")

# ╔═╡ 2fb1145a-ed83-440a-8b79-d5921e7cd4a8
cali_shape_Vega = VegaDatasets.VegaJSONDataset(cali_shape, "./data/california-counties.json")

# ╔═╡ 2df78b49-fd72-4027-a445-70777983cb18
md"""
#### Plotting the map
"""

# ╔═╡ eaf7d4cf-a000-42e3-bdf4-f63bb4be2a88
map = @vlplot(width=500, height=300) + @vlplot(
	mark = {
		:geoshape,
		fill=:black,
		stroke=:white
	},
	data = {
		values = cali_shape_Vega,
		format = {
			type=:topojson,
			feature=:cb_2015_california_county_20m
		},
	},
	projection = {
		type=:albersUsa
	},
)

# ╔═╡ 097be2ef-e742-41b5-a032-ebbd3023e24b
md"""
Let's now add the `median_house_value` to the map and see the distribution of house prices across the state.

We need to start a `@vlplot` sum with an `@vlplot()` before adding `map`. We can specify the size of the plot too. 

If instead we add the initial `@vlplot()` to the definition of `map` we can simply add new layers easily. 
"""

# ╔═╡ d79d0e9e-20cc-4f41-b6cc-c4e2e5633723
map + @vlplot(
	:circle,
	data=housedf,
	projection = {
		type=:albersUsa
	},
	longitude="longitude:q",
	latitude="latitude:q",
	size = {
		value=12
	},
	color = "median_house_value:q"
)

# ╔═╡ 9a1a19cf-6505-4291-9363-188c87fc4b73
md"""
We want to see if house prices are correlated in some way with location. In the above plot we see a tendency for higher house prices to be found inear the coast and in city regions.

Let's further highlight this by `bucket`ing the prices. We also add that to our DataFrame. We can simply do this by creating a new column. The guide uses `insertcols!` which probably does the same thing. I will use this later to demo it.
"""

# ╔═╡ cf7985ee-7681-4f78-9799-34f95f1b1539
housedf.bucketprice = Int.(div.(housedf.median_house_value, 50_000))

# ╔═╡ 2816b860-19f3-4179-944f-a4629105b512
map + @vlplot(
	:circle,
	data=housedf,
	projection = {
		type=:albersUsa
	},
	longitude="longitude:q",
	latitude="latitude:q",
	size = {
		value=12
	},
	color = "bucketprice:n"
)

# ╔═╡ a719560c-eb53-4bdc-ac03-9657fb570a21
md"""
Cheaper houses tend to be found inland, in more rural areas.

### K-Means Clustering

[K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)
"""

# ╔═╡ 876f6b2a-3278-4e46-9458-b685f8471fb6
house_loc = [housedf.latitude housedf.longitude] # We cluster on locations

# ╔═╡ 17c6f57f-f78c-418c-9fe8-8a25eac14730
house_loc_Kmeans = kmeans(house_loc', 10) # 10 clusters - we have 10 buckets above

# ╔═╡ c0153d03-e714-4784-9e84-963eb74e0633
md"""
Let's use `insertcols!` this time. We need to pass the df, optionally a column location, and a pair :symbol=>column
"""

# ╔═╡ 320892d1-8592-438b-9ec7-0cb2b2862e18
begin
	if "kmeans" in names(housedf)
		select!(housedf, Not(:kmeans))
	end
	insertcols!(housedf, :kmeans => house_loc_Kmeans.assignments)
end

# ╔═╡ 337ef205-7c0a-41bd-81a5-17c3aee3d1f5
md"""
Let's now plot the clusters
"""

# ╔═╡ ed37080d-5645-4ffc-be01-df724d8ac621
map + @vlplot(
	:circle,
	data=housedf,
	projection = {
		type=:albersUsa
	},
	longitude="longitude:q",
	latitude="latitude:q",
	size = {
		value=12
	},
	color = "kmeans:n"
)

# ╔═╡ 08623786-4b57-4d30-89ad-a38545f6f454
md"""
The above plot is a clustering of only location, price is not taken into account at all. So to analyse prices we need to compare with the `bucketprice` plot. 

If we cluster on location and median price we get:
"""

# ╔═╡ 1cb60c0e-ae86-41ee-b4fe-04de7e961b99
begin
	house_loc_price = hcat(house_loc, housedf.median_house_value)
	house_loc_price_Kmeans = kmeans(house_loc_price', 10)
	if "kmeans_price" in names(housedf)
		select!(housedf, Not(:kmeans_price))
	end
	insertcols!(housedf, :kmeans_price => house_loc_price_Kmeans.assignments)
end

# ╔═╡ e3444ec8-ca59-43b2-bfe4-3e99af5cbde9
map + @vlplot(
	:circle,
	data=housedf,
	projection = {
		type=:albersUsa
	},
	longitude="longitude:q",
	latitude="latitude:q",
	size = {
		value=12
	},
	color = "kmeans_price:n"
)

# ╔═╡ 995167aa-317c-4931-80b4-fbda7bd3db95
housedf |> @vlplot(width=500, height=300) + @vlplot(
	:circle,
	x = :kmeans,
	y = {:median_house_value, bin={maxbins=10}},
	size="count()"
)	

# ╔═╡ d90ead85-bbb3-43a5-a840-4c9136b41248
md"""
The course instructor concludes that location does affect house price, but location means proximity to water, proximity to downtown, etc.

Looking at the distribution of prices for each cluster in the above plot we can see that each cluster has a not to disimmilar distribution. Cluster 5 does have a peak at higher prices however.

We now try some different clustering methods.

### K-Medoids Clustering

For [K-Medoids Clustering](https://en.wikipedia.org/wiki/K-medians_clustering) we require a `distance` matrix. We use `Distances.jl` for this and compute pairwise Euclidean distances.
"""

# ╔═╡ 65cdb5d9-7f15-422c-bdf9-c357263901f7
loc_distance = pairwise(Euclidean(), house_loc', house_loc', dims=2)

# ╔═╡ 7bf6ee2e-600a-48ad-88df-4148bedf9e8d
house_loc_kmedoids = kmedoids(loc_distance, 10)

# ╔═╡ c28be874-3122-4f55-80a4-2f21fcccc10e
housedf.kmedoids = house_loc_kmedoids.assignments;

# ╔═╡ 6695977f-e8b7-4976-8986-9c3b5a3cc3d6
map + @vlplot(
	:circle,
	data=housedf,
	projection = {
		type=:albersUsa
	},
	longitude="longitude:q",
	latitude="latitude:q",
	size = {
		value=12
	},
	color = "kmedoids:n"
)

# ╔═╡ de836b6e-8ab6-464c-924e-010108c39bad
md"""
### Hierarchial Clustering

For [Hierarchial Clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) we need to use a distance matrix in this case. We use the `hclust` function and then need to cut the tree using `cutree`. This is the same language as in `R`.
"""

# ╔═╡ 2b648a14-449e-41db-83b8-9413cec93047
begin
	house_hclust = hclust(loc_distance)
	house_hclust10 = cutree(house_hclust; k=10)
	housedf.hclust = house_hclust10
end

# ╔═╡ 9e563561-2767-4014-aff6-716428468e5b
map + @vlplot(
	:circle,
	data=housedf,
	projection = {
		type=:albersUsa
	},
	longitude="longitude:q",
	latitude="latitude:q",
	size = {
		value=12
	},
	color = "hclust:n"
)

# ╔═╡ 9658d4af-f417-4f10-aba9-621095b882d4
md"""
`hclust` doesn't really give us any useful information.

### DBscan

For [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) we also need a distance. In this case we will take a different distance metric. 
"""

# ╔═╡ a38e51e9-d115-4dcb-b26d-9964325bc313
housedf.dbscan = dbscan(
	pairwise(SqEuclidean(), house_loc', dims=2),
	0.05,
	10
).assignments

# ╔═╡ 3e1c6fea-14b1-4697-a3f0-9f08b3d666e2
map + @vlplot(
	:circle,
	data=housedf,
	projection = {
		type=:albersUsa
	},
	longitude="longitude:q",
	latitude="latitude:q",
	size = {
		value=12
	},
	color = "dbscan:n"
)

# ╔═╡ 9a057076-4f59-4fbb-9f5b-1e0a072affb1
md"""
Interestingly, we find that `dbscan` results in 15 clusters, even though we requested 10. Note that in this case I just computed the new column in a single command, rather than building this up in multiple steps. The reason for this is that computing a distance matrix on this data saves a `20640×20640` matrix into memory. Normally this is ok as I have enough space, but I've generated a couple of these already so I didn't want to save it this time. 

### Conclusion

We find that house price is correlated slightly with location, but in the sense of proximity to features, such as water or downtown. 

The mapping by price does not match the mapping by location. We can see this better with the scatter plot:
"""

# ╔═╡ 308197fd-8d90-4ef4-8f81-6822bf88c5eb
housedf |> @vlplot(width=500, height=300) + @vlplot(
	:circle,
	x = :kmeans,
	y = {:median_house_value, bin={maxbins=10}},
	size="count()",
	color="kmeans:n"
)	

# ╔═╡ 939e5d3e-462a-497e-aa0c-2f2996d5ecf3
md"""
I am quite impressed with the `VegaLite` package. It functions quite nicely and gives us some nice looking plots - similar to `ggplot`. I need to play with it a bit more to see how useful it is in comparison to `ggplot`.

One of the challenges of working with large datasets is the amount of memory we have. If we are doing computations such as computing distances it may be better to keep memory space by running the computation within the function of interest. The risk here is that it is inefficient, especially if we need to re-run the computation due to error. If we already have the distance matrix stored we do not need to re compute it. If on the otherhand we are discarding it each time then we need to recompute. It also makes the process of debugging much more complicated. 

One approach to memory management is to use multiple notebooks & save results at each end-point. 
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Clustering = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
VegaDatasets = "0ae4a718-28b7-58ec-9efb-cded64d6d5b4"
VegaLite = "112f6efa-9a02-5b7d-90c0-432ed331239a"

[compat]
CSV = "~0.9.5"
Clustering = "~0.14.2"
DataFrames = "~1.2.2"
Distances = "~0.10.4"
HTTP = "~0.9.16"
JSON = "~0.21.2"
VegaDatasets = "~2.1.1"
VegaLite = "~2.6.0"
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

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "15b18ea098a4b5af316df529c2ff4055fcef36e9"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.9.5"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "a325370b9dd0e6bf5656a6f1a7ae80755f8ccc46"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.7.2"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

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

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

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

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "9f46deb4d4ee4494ffb5a40a27a2aced67bdd838"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.4"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[DoubleFloats]]
deps = ["GenericLinearAlgebra", "LinearAlgebra", "Polynomials", "Printf", "Quadmath", "Random", "Requires", "SpecialFunctions"]
git-tree-sha1 = "1c962cf7e75c09a5f1fbf504df7d6a06447a1129"
uuid = "497a8b3b-efae-58df-a0af-a86822472b78"
version = "1.1.23"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "3c041d2ac0a52a12a27af2782b34900d9c3ee68c"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.1"

[[FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[FilePathsBase]]
deps = ["Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "7fb0eaac190a7a68a56d2407a6beff1142daf844"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.12"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GenericLinearAlgebra]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "eddbb6ee8fe2c3244a2c973874a3179c3c4d3ac5"
uuid = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
version = "0.2.6"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "19cb49649f8c41de7fea32d089d37de917b553da"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.0.1"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Intervals]]
deps = ["Dates", "Printf", "RecipesBase", "Serialization", "TimeZones"]
git-tree-sha1 = "323a38ed1952d30586d0fe03412cde9399d3618b"
uuid = "d8418881-c3e1-53bb-8760-2df7ec849ed5"
version = "1.5.0"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterableTables]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Requires", "TableTraits", "TableTraitsUtils"]
git-tree-sha1 = "70300b876b2cebde43ebc0df42bc8c94a144e1b4"
uuid = "1c8ee90f-4401-5389-894e-7a04a3dc0f4d"
version = "1.0.0"

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

[[JSONSchema]]
deps = ["HTTP", "JSON", "URIs"]
git-tree-sha1 = "2f49f7f86762a0fbbeef84912265a1ae61c4ef80"
uuid = "7d188eb4-7ad8-530c-ae41-71a32a6d4692"
version = "0.3.4"

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

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

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

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "3927848ccebcc165952dc0d9ac9aa274a87bfe01"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.2.20"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[NodeJS]]
deps = ["Pkg"]
git-tree-sha1 = "905224bbdd4b555c69bb964514cfa387616f0d3a"
uuid = "2bd173c7-0d6d-553b-b6af-13a54713934c"
version = "1.3.0"

[[Nullables]]
git-tree-sha1 = "8f87854cc8f3685a60689d8edecaa29d2251979b"
uuid = "4d1e1d77-625e-5b40-9113-a560ec7a8ecd"
version = "1.0.0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

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
git-tree-sha1 = "a8709b968a1ea6abc2dc1967cb1db6ac9a00dfb6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.5"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[Polynomials]]
deps = ["Intervals", "LinearAlgebra", "MutableArithmetics", "RecipesBase"]
git-tree-sha1 = "029d2a5d0e6c2b5d87ac690aa58dcf40c2e2acb1"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "2.0.15"

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
git-tree-sha1 = "6330e0c350997f80ed18a9d8d9cb7c7ca4b3a880"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.2.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Quadmath]]
deps = ["Printf", "Random", "Requires"]
git-tree-sha1 = "5a8f74af8eae654086a1d058b4ec94ff192e3de0"
uuid = "be4d8f0f-7fa4-5f49-b795-2f01399ab2dd"
version = "0.5.5"

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

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "54f37736d8934a12a200edea2f9206b03bdf3159"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.7"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "fca29e68c5062722b5b4435594c3d1ba557072a3"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.7.1"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

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

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableShowUtils]]
deps = ["DataValues", "Dates", "JSON", "Markdown", "Test"]
git-tree-sha1 = "14c54e1e96431fb87f0d2f5983f090f1b9d06457"
uuid = "5e66a065-1f0a-5976-b372-e0b8c017ca10"
version = "0.2.5"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[TableTraitsUtils]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Missings", "TableTraits"]
git-tree-sha1 = "78fecfe140d7abb480b53a44f3f85b6aa373c293"
uuid = "382cd787-c1b6-5bf2-a167-d5b971a19bda"
version = "1.0.2"

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

[[TextParse]]
deps = ["CodecZlib", "DataStructures", "Dates", "DoubleFloats", "Mmap", "Nullables", "WeakRefStrings"]
git-tree-sha1 = "eb1f4fb185c8644faa2d18d14c72f2c24412415f"
uuid = "e0df1984-e451-5cb5-8b61-797a481e67e3"
version = "1.0.2"

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

[[URIParser]]
deps = ["Unicode"]
git-tree-sha1 = "53a9f49546b8d2dd2e688d216421d050c9a31d0d"
uuid = "30578b45-9adc-5946-b283-645ec420af67"
version = "0.4.1"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Vega]]
deps = ["DataStructures", "DataValues", "Dates", "FileIO", "FilePaths", "IteratorInterfaceExtensions", "JSON", "JSONSchema", "MacroTools", "NodeJS", "Pkg", "REPL", "Random", "Setfield", "TableTraits", "TableTraitsUtils", "URIParser"]
git-tree-sha1 = "43f83d3119a868874d18da6bca0f4b5b6aae53f7"
uuid = "239c3e63-733f-47ad-beb7-a12fde22c578"
version = "2.3.0"

[[VegaDatasets]]
deps = ["DataStructures", "DataValues", "FilePaths", "IterableTables", "IteratorInterfaceExtensions", "JSON", "TableShowUtils", "TableTraits", "TableTraitsUtils", "TextParse"]
git-tree-sha1 = "c997c7217f37205c5795de8c797f8f8531890f1d"
uuid = "0ae4a718-28b7-58ec-9efb-cded64d6d5b4"
version = "2.1.1"

[[VegaLite]]
deps = ["Base64", "DataStructures", "DataValues", "Dates", "FileIO", "FilePaths", "IteratorInterfaceExtensions", "JSON", "MacroTools", "NodeJS", "Pkg", "REPL", "Random", "TableTraits", "TableTraitsUtils", "URIParser", "Vega"]
git-tree-sha1 = "3e23f28af36da21bfb4acef08b144f92ad205660"
uuid = "112f6efa-9a02-5b7d-90c0-432ed331239a"
version = "2.6.0"

[[WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "c69f9da3ff2f4f02e811c3323c22e5dfcb584cfa"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.1"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─4cd10f86-2450-11ec-3a70-0bb30b3fb525
# ╠═41a65105-f9c3-4e12-a70d-10d885ec2696
# ╟─367f19de-2587-4b6b-ae51-014bc60c59fc
# ╠═d3cfdd07-6803-412a-bb50-bb70613ac9fa
# ╠═2d4af172-a17f-42da-9572-b51ee72a1f1b
# ╟─140ec774-f1bd-479d-984a-91c42064ac5b
# ╠═66ba9bf8-f235-4242-88b0-118710d20ead
# ╠═2fb1145a-ed83-440a-8b79-d5921e7cd4a8
# ╟─2df78b49-fd72-4027-a445-70777983cb18
# ╠═eaf7d4cf-a000-42e3-bdf4-f63bb4be2a88
# ╟─097be2ef-e742-41b5-a032-ebbd3023e24b
# ╠═d79d0e9e-20cc-4f41-b6cc-c4e2e5633723
# ╟─9a1a19cf-6505-4291-9363-188c87fc4b73
# ╠═cf7985ee-7681-4f78-9799-34f95f1b1539
# ╠═2816b860-19f3-4179-944f-a4629105b512
# ╟─a719560c-eb53-4bdc-ac03-9657fb570a21
# ╠═876f6b2a-3278-4e46-9458-b685f8471fb6
# ╠═17c6f57f-f78c-418c-9fe8-8a25eac14730
# ╟─c0153d03-e714-4784-9e84-963eb74e0633
# ╠═320892d1-8592-438b-9ec7-0cb2b2862e18
# ╟─337ef205-7c0a-41bd-81a5-17c3aee3d1f5
# ╠═ed37080d-5645-4ffc-be01-df724d8ac621
# ╟─08623786-4b57-4d30-89ad-a38545f6f454
# ╠═1cb60c0e-ae86-41ee-b4fe-04de7e961b99
# ╠═e3444ec8-ca59-43b2-bfe4-3e99af5cbde9
# ╠═995167aa-317c-4931-80b4-fbda7bd3db95
# ╟─d90ead85-bbb3-43a5-a840-4c9136b41248
# ╠═65cdb5d9-7f15-422c-bdf9-c357263901f7
# ╠═7bf6ee2e-600a-48ad-88df-4148bedf9e8d
# ╠═c28be874-3122-4f55-80a4-2f21fcccc10e
# ╠═6695977f-e8b7-4976-8986-9c3b5a3cc3d6
# ╟─de836b6e-8ab6-464c-924e-010108c39bad
# ╠═2b648a14-449e-41db-83b8-9413cec93047
# ╠═9e563561-2767-4014-aff6-716428468e5b
# ╟─9658d4af-f417-4f10-aba9-621095b882d4
# ╠═a38e51e9-d115-4dcb-b26d-9964325bc313
# ╠═3e1c6fea-14b1-4697-a3f0-9f08b3d666e2
# ╟─9a057076-4f59-4fbb-9f5b-1e0a072affb1
# ╠═308197fd-8d90-4ef4-8f81-6822bf88c5eb
# ╟─939e5d3e-462a-497e-aa0c-2f2996d5ecf3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
