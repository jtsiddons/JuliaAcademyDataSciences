### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ e1f1b656-268d-11ec-1272-8566a43e80f5
using Pkg; Pkg.activate(".")

# ╔═╡ 6dbc9821-50bd-4d01-894b-c716c2a9b655
using LightGraphs, MatrixNetworks, VegaDatasets, DataFrames, SparseArrays, LinearAlgebra, CairoMakie, VegaLite, Statistics

# ╔═╡ ffc28d3b-4e85-4ebd-bc84-7e784b368efe
md"""
# Julia Academy Data Sciences Course

## 08. Graphs

This is a Pluto.jl write-up of the 8th notebook for the Julia Academy Data Sciences course. In this case I am experimenting with using 

```julia
using Pkg; Pkg.activate(".")
```

in a Pluto.jl notebook in order to use the local directory `.toml` files.

---

This notebook is for **Graphs**. We will be looking at flight data. Let's start, as usual, with loading the packages.
"""

# ╔═╡ 3aa0ec48-1eac-465c-8759-25184ecf3e2a
md"""
### Airport Data
"""

# ╔═╡ 5601704c-a871-40c4-88e7-732bb7747c5d
begin
	airports = DataFrame(dataset("airports"))
	flightsairport = DataFrame(dataset("flights-airport"))
end;

# ╔═╡ 67cbdb16-4676-44c3-a02b-64d1780d19c7
describe(airports)

# ╔═╡ f39af8fb-b8df-4177-9840-326dfa72e744
describe(flightsairport)

# ╔═╡ 71d7ba7a-a8e8-4258-9183-151c28dfc477
md"""
We have two datasets here. 

#### `airports`

`airports` contains meta data for airports in the United States. It includes the airports location (both city/state and long/lat), name, and iata code.

#### `flightsairport`

`flightsairport` contains the number of flights between two airports from `airports`. I note that `count` has some non-integer values, let's correct that.
"""

# ╔═╡ 171b1737-6da1-461c-950d-68bb78392928
flightsairport.count = Int.(flightsairport.count);

# ╔═╡ 8f26dc81-28eb-4bd2-bb3d-772f2cc3b8ba
typeof(flightsairport.count)

# ╔═╡ 4ab0a828-14cf-4cfc-81d4-ca6b9a70f4bb
md"""
We want to create a subset of `airports` containing only airports in the `flightsairport` data. I will find all airports in the `flightsairport` data, and then filter the `airports` data for only `iata` values that are present.
"""

# ╔═╡ 0f0687f9-ea20-479a-811d-d112fa9a3d29
uairports = unique(vcat(flightsairport.origin, flightsairport.destination));

# ╔═╡ 2fb46f15-e9a7-4924-b216-13a965c25745
filter!(row -> row.iata in uairports, airports)

# ╔═╡ 0646e0ee-b3ea-428a-b19f-9083291554cd
md"""
### Building the graph

Let's start by building the adjacency matrix. This is a matrix that has a 1 in elements where two airports have a flight connecting them. The rows are the origin airports and the columns are destinations. However, we will symmetrise the result for convenience.
"""

# ╔═╡ 61019ad1-f954-4b09-ab2f-12ed406e50b7
begin
	origin_ids = findfirst.(isequal.(flightsairport.origin), [uairports])
	dest_ids = findfirst.(isequal.(flightsairport.destination), [uairports])
	edgeweights = flightsairport.count
end;

# ╔═╡ 4da598c3-7a03-4444-a59b-5353003d3c10
begin
	A = sparse(origin_ids, dest_ids, 1, length(uairports), length(uairports))
	A = max.(A,A')
end

# ╔═╡ 53ac14d3-73ce-433d-bb92-baa086b636c4
graph_df = DataFrame(origin=origin_ids, dest=dest_ids, weights=edgeweights)

# ╔═╡ 6695169f-2177-4feb-a421-e94f2dda7369
begin
	F_A = Figure()
	Ax_A1 = Axis(F_A[1,1])
	sc = scatter!(Ax_A1, graph_df.origin, graph_df.dest, color=graph_df.weights)
	Ax_A1.xlabel = "Origin Airport"
	Ax_A1.ylabel = "Destination Airport"
	Colorbar(F_A[1, 2], sc)
	F_A
end

# ╔═╡ 07361655-d7c4-41b3-af49-e5251224230b
md"""
### Converting to a Graph

We use the `LightGraphs` package. We also make use of `MatrixNetworks`.

Below is an example of how to build a graph using `LightGraphs`.
"""

# ╔═╡ be832d6a-cc48-4be9-94b4-5b7e5521829f
begin
	G = SimpleGraph(10) # SimpleGraph(nnodes, nedges)
	add_edge!(G,7,5) # Modifies Graph in place - adds an edge between nodes 7 & 5
end

# ╔═╡ 5a92c461-149e-49f7-b311-a50cb14f84f6
airport_graph = SimpleGraph(A)

# ╔═╡ 295c124a-b7f2-4d5a-a705-6b60f6a2c7d9
md"""
We will primarily use `MatrixNetworks` and the graph's adjacency matrix `A` to perform all operations.

#### Components of a graph

Often we want to know the connected components of the graph, especially for later performing diffusion based operations on it. If a graph is disconnected there is no way for information to pass between every node.
"""

# ╔═╡ 150697dc-416b-4b4b-9a9c-a19559926fbf
cc = scomponents(A)

# ╔═╡ 7d8626fe-0810-4cc9-9961-98ca6c8309c1
md"""
`cc.sizes` will show the size of each component that is found. In this case we can see that the `sizes` field is `[305]`, which means that every node can be reached from every other node - not necessarily directly, only that ∃ a path between every pair of nodes.

In our example, that means that we can fly to every airport, although it may require us to change.

### Degree Distribution
"""

# ╔═╡ 1ec36fb0-f237-4a83-a8a3-0f0ad8e17b82
degrees = sum(A, dims=2)[:]

# ╔═╡ ae43539e-4235-4365-b5ef-08cf918d2e59
begin
	F_deg = Figure()
	Ax_deg1 = Axis(F_deg[1,1], yscale=log10)
	lines!(Ax_deg1, sort(degrees,rev=true))
	Ax_deg1.ylabel = "log degree"
	Ax_deg2 = Axis(F_deg[1,2])
	lines!(Ax_deg2, sort(degrees,rev=true))
	Ax_deg2.ylabel = "degree"
	Ax_deg3 = Axis(F_deg[2,1:2])
	hist!(Ax_deg3, degrees)
	Ax_deg3.xlabel = "degree"
	Ax_deg3.ylabel = "n"
	F_deg
end

# ╔═╡ 1836a190-5688-47de-a416-44c3d02c54d7
md"""
The degree distribution appears to fit a powerlaw distribution. It is useful to know that our degree distribution follows a well known model. 

Which airport has highest degree?
"""

# ╔═╡ 6b18f1f3-143d-4859-93fe-62ca1ffa3d05
begin
	maxid = argmax(degrees) 
	maxid_iata = uairports[maxid]
	airports.name[airports.iata .== maxid_iata]
end

# ╔═╡ 347ba300-7ee5-4aaa-9b99-e6d7180b6700
md"""
### Plotting some maps
"""

# ╔═╡ 3b0b81fb-3ed8-40a7-af58-abe5563e8800
us10m = dataset("us-10m")

# ╔═╡ 4ce49efc-442b-47f1-9710-3521cb2fcd43
usmap = @vlplot(width=500, height=300) +
@vlplot(
    mark={
        :geoshape,
        fill=:lightgray,
        stroke=:white
    },
    data={
        values=us10m,
        format={
            type=:topojson,
            feature=:states
        }
    },
    projection={type=:albersUsa},
)

# ╔═╡ 3989350b-1669-4aa1-87f7-c17b84905aba
airp_loc = usmap + @vlplot(
    :circle,
    data=airports,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size={value=15},
    color={value=:red}
)

# ╔═╡ bfc5469e-ea99-4283-9384-b357d2328376
md"""
There is one airport located in the lop left corner of this plot. I have identified it as "STX" which is on the US Virgin Islands. I assume that it is a quirk of the mapping that it gets located there rather than its real location.
"""

# ╔═╡ 47fb45e5-e6b6-4799-84c9-abf47bad03fc
filter(row -> row.iata == airports.iata[argmin(airports.latitude)], airports)

# ╔═╡ 9d192c69-3ded-42ef-8ef6-8aa873693e6e
airp_loc + @vlplot(
    :rule,
    data=flightsairport,
    transform=[
        {filter={field=:origin,equal=:STX}},
        {
            lookup=:origin,
            from={
                data=airports,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["origin_latitude", "origin_longitude"]
        },
        {
            lookup=:destination,
            from={
                data=airports,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["dest_latitude", "dest_longitude"]
        }
    ],
    projection={type=:albersUsa},
    longitude="origin_longitude:q",
    latitude="origin_latitude:q",
    longitude2="dest_longitude:q",
    latitude2="dest_latitude:q"
)

# ╔═╡ 9e996361-538c-4976-8b02-6884acaf9ba0
md"""
Let's see all the flights out of our most connected airport.
"""

# ╔═╡ 27e37f73-56d4-4de1-a16f-2dc5706e615e
airp_loc + @vlplot(
    :rule,
    data=flightsairport,
    transform=[
        {filter={field=:origin,equal=:ATL}},
        {
            lookup=:origin,
            from={
                data=airports,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["origin_latitude", "origin_longitude"]
        },
        {
            lookup=:destination,
            from={
                data=airports,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["dest_latitude", "dest_longitude"]
        }
    ],
    projection={type=:albersUsa},
    longitude="origin_longitude:q",
    latitude="origin_latitude:q",
    longitude2="dest_longitude:q",
    latitude2="dest_latitude:q"
)

# ╔═╡ 22214dac-30e5-4f83-8ac9-42994e02dbf0
md"""
### Shortest Paths

We use [Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)

The output is a tuple containing number of stops and the route.

The route for a given node is the airport for the previous connection.

For example, if stops = 1, then the route will show the origin airport.
If stops = 2 then route will be the previous airport, that airport's route will show the origin.
"""

# ╔═╡ 629d4c2c-d01d-4a6c-9474-22f35de738df
(ATL_n_connections, ATL_route) = dijkstra(A, maxid)

# ╔═╡ a1de57e3-a5ad-458d-a01e-665b16e94bfb
md"""
We can write a function to get a route from ATL.
"""

# ╔═╡ b959c2ac-bdf0-4855-9852-e359ccd0135e
function get_route_from_ATL(dest::String, path::Vector=[dest])
	id = findfirst(uairports .== dest)
	resid = ATL_route[id]
	newdest = uairports[resid]
	
	push!(path, newdest)
	if path[end] == "ATL"
		return path
	else
		get_route_from_ATL(newdest, path)
	end
end	

# ╔═╡ 9579a5b4-85b8-4605-8006-db3388c1dd68
get_route_from_ATL("GST")[end:-1:1]

# ╔═╡ ecffdd2b-0891-41fc-9d79-5799b6f5779d
begin
	destination = "GST"
	route = get_route_from_ATL(destination);
	route_df = DataFrame(airport=Symbol.(route[end:-1:1]), order=1:length(route));
end

# ╔═╡ 2ac7fabb-1c81-4791-982b-b6f11fdeb5f6
airp_loc + @vlplot(
    :line,
    data=route_df,
    transform=[{
        lookup=:airport,
        from={
            data=airports,
            key=:iata,
            fields=["latitude","longitude"]
        }
    }],
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    order={field=:order,type=:ordinal}
)

# ╔═╡ 56838629-86ed-4c83-8a1c-2a9234c053fe
md"""
Let's improve the function to compute the route between any pair of airports.

It requires computation of the shortest path for the origin airport initially, but then we can recursively build using the route.
"""

# ╔═╡ 1b0b94e0-ed18-4da9-8e5b-82f6d589f4bf
function get_route(origin::String, dest::String, path::Vector=[dest], route=dijkstra(A,findfirst(uairports .== origin)))
	id = findfirst(uairports .== dest)
	resid = route[2][id]
	newdest = uairports[resid]
	
	push!(path, newdest)
	if path[end] == origin
		return path
	else
		get_route(origin, newdest, path, route)
	end
end	

# ╔═╡ f9f5065d-831a-4423-a0b8-d92f8e406f15
get_route("ABY", "GST")

# ╔═╡ b0ef6cff-590a-433d-bcb5-e08b25fcbb4b
md"""
An example! I can put any origin or destination into the below field and it will change the plot accordingly.
"""

# ╔═╡ b45a69d5-fe8d-4ba2-8361-33aa357713f2
begin
	origin2 = "DFW"; destination2 = "ABY"
	route2 = get_route(origin2, destination2;)
	route2_df = DataFrame(airport=Symbol.(route2[end:-1:1]), order=1:length(route2));
end

# ╔═╡ f51bcc3a-e9ee-4f65-8630-7d65a4b7edce
airp_loc + @vlplot(
    :line,
    data=route2_df,
    transform=[{
        lookup=:airport,
        from={
            data=airports,
            key=:iata,
            fields=["latitude","longitude"]
        }
    }],
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    order={field=:order,type=:ordinal},
	title="Route from $origin2 to $destination2"
)

# ╔═╡ b48fd05c-a6b5-40e4-ae20-69201c2f9f98
md"""
### Minimum Spanning Tree

The next problem is forming a minimum spanning tree on the graph. The idea of a minimum spanning tree is to connect all nodes in the graph with as few edges as possible. We will use Prim's algorithm for this problem.
"""

# ╔═╡ 06f9f1db-5584-44d1-b6b3-ca1abc1be7fd
ti, tj, tv, nverts = mst_prim(A)

# ╔═╡ 1cc017ad-e891-43f0-9056-ab8c88c24c47
span_df = DataFrame(:ei=>uairports[ti], :ej=>uairports[tj])

# ╔═╡ cadf2547-bdf1-4f90-9120-14eb01b238a1
airp_loc + @vlplot(
    :rule,
    data=span_df, #data=flightsairport,
    transform=[
        {
            lookup=:ei,
            from={
                data=airports,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["originx", "originy"]
        },
        {
            lookup=:ej,
            from={
                data=airports,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["destx", "desty"]
        }
    ],
    projection={type=:albersUsa},
    longitude="originy:q",
    latitude="originx:q",
    longitude2="desty:q",
    latitude2="destx:q"
)

# ╔═╡ 248a8ccf-6553-41c1-8baa-2abc0ed3cbf9
md"""
### PageRank

PageRank is the algorithm that got Google started. The idea is: given an network of connections between multiple nodes (web pages in the cae of Google), is there a way to return a list of ranked nodes? PageRank provides this ranking. Obviously, nodes can be ranked in several different ways but PageRank remains to be one of the most popular methods in network analysis. 

---

# pagerank

PageRank is the stationary distribution of a Markov chain defined as follows. The behavior of the chain at state `i` is:

* with probability `lpha`, randomly transition to an out-neighbor of the current node (based on a weighted probabilities if the graph has non-negative weights).
* with probability `1-lpha`, jump to a random node chosen anywhere in the network.
* if there are no out-neighbors, then jump anywhere in the network with equal probability.

The solution satisfies a linear system of equations. This function will solve that linear system to a 1-norm error of `tol` where `tol` is chosen by default to be the machine precision.

The solution is always a probability distribution.

##  Functions

* `x = pagerank(A::SparseMatrixCSC{V,Int},alpha::Float64)`
* `x = pagerank(A::MatrixNetwork{V},alpha::Float64)`
* `x = pagerank(A,alpha,eps::Float64)` specifies solution tolerance too.

## Inputs

* `A`: The sparse matrix or matrix network that you wish to use to compute PageRank. In the case of a sparse matrix, it must have non-negative values and the values will impact the computation as we will compute a stochastic normalization as part of the algorithm.
* `alpha`: the teleportation parameter given above.
* `tol`: the tolerance, the default choice is machine precision for floating point, which is more than enough for almost all applications.

## Examples

```julia
  pagerank(A,alpha)
              # return the standard PageRank vector
              # with uniform teleportation and alpha=0.85
              # computed to machine precision
```
"""

# ╔═╡ 987239f9-82be-4e75-84a0-cfe1f7f02175
airports.pagerank = MatrixNetworks.pagerank(A,0.85)

# ╔═╡ d164e465-1397-490f-af4c-e6b7414acab8
sum(airports.pagerank)

# ╔═╡ c825d51f-d476-4db8-bd6e-911f2c26f275
usmap + @vlplot(
    :circle,
    data=airports,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size="pagerank:q",
    color={value=:steelblue}
)

# ╔═╡ f908d0e8-0626-486c-90c5-418cce57a5fb
cor(airports.pagerank, degrees)

# ╔═╡ f5d9f29e-d6a6-451a-ab68-4e3cf1162da7
md"""
### Clustering Coefficients

From Wikipedia: The local clustering coefficient of a vertex (node) in a graph quantifies how close its neighbours are to being a clique (complete graph).

This means that if for example, a node is connected to two nodes that are also connected to each other, that node's clustering coeefficient is 1. This can be a good metric to find out which nodes tend to have tight clusters around them. Let's look at the documentation of `clustercoeffs` from MatrixNetworks.
"""

# ╔═╡ e1c7176d-c897-4ffe-ba31-9608ba79e2c7
airports.clusteringcoef = clustercoeffs(A)

# ╔═╡ bcc7a580-00b2-4960-b9f3-5c862ac7fcb1
begin	
	airports.clusteringcoef[findall(airports.clusteringcoef .<= eps())] .= 0
	airports.clusteringcoef[findall(isnan.(airports.clusteringcoef))] .= 0
end

# ╔═╡ 9dd79ef1-a808-48e5-b9ba-265c4b91a9c4
cor(airports.clusteringcoef, degrees)

# ╔═╡ 2585d0ae-0afa-42ba-b266-61b4e3be9c56
usmap + @vlplot(
    :circle,
    data=airports,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size="clusteringcoef:q",
    color={value=:gray}
)

# ╔═╡ 357aa7e8-47d6-4e12-8934-04b5251dc629
md"""
### PageRank + Prim's Algorithm
"""

# ╔═╡ ffb4c4e1-fe07-4067-a450-2ee8a1c80663
usmap + @vlplot(
    :rule,
    data=span_df, #data=flightsairport,
    transform=[
        {
            lookup=:ei,
            from={
                data=airports,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["originx", "originy"]
        },
        {
            lookup=:ej,
            from={
                data=airports,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["destx", "desty"]
        }
    ],
    projection={type=:albersUsa},
    longitude="originy:q",
    latitude="originx:q",
    longitude2="desty:q",
    latitude2="destx:q"
) + @vlplot(
    :circle,
    data=airports,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size="pagerank:q",
    color={value=:steelblue}
)

# ╔═╡ 365feac5-4f17-47b2-a6db-e0802c69da64
md"""
### Conclusion

We notice that the airports with high PageRank are the big hub airports. This isn't clear from the combined PageRank & Minimum Spanning Tree plot since the airports with the most number of edges are not necessarily major hubs. For example Dallas Fort Worth has high PageRank, and is a hub for Texas, but is only connected to one other airport on the minimum spanning tree. We can see that it is a hub by looking at the map for it.
"""

# ╔═╡ a95bf0f9-0c30-4c42-9316-601c39416907
airp_loc + @vlplot(
    :rule,
    data=flightsairport,
    transform=[
        {filter={field=:origin,equal=:DFW}},
        {
            lookup=:origin,
            from={
                data=airports,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["origin_latitude", "origin_longitude"]
        },
        {
            lookup=:destination,
            from={
                data=airports,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["dest_latitude", "dest_longitude"]
        }
    ],
    projection={type=:albersUsa},
    longitude="origin_longitude:q",
    latitude="origin_latitude:q",
    longitude2="dest_longitude:q",
    latitude2="dest_latitude:q"
)

# ╔═╡ Cell order:
# ╠═e1f1b656-268d-11ec-1272-8566a43e80f5
# ╟─ffc28d3b-4e85-4ebd-bc84-7e784b368efe
# ╠═6dbc9821-50bd-4d01-894b-c716c2a9b655
# ╟─3aa0ec48-1eac-465c-8759-25184ecf3e2a
# ╠═5601704c-a871-40c4-88e7-732bb7747c5d
# ╠═67cbdb16-4676-44c3-a02b-64d1780d19c7
# ╠═f39af8fb-b8df-4177-9840-326dfa72e744
# ╟─71d7ba7a-a8e8-4258-9183-151c28dfc477
# ╠═171b1737-6da1-461c-950d-68bb78392928
# ╠═8f26dc81-28eb-4bd2-bb3d-772f2cc3b8ba
# ╟─4ab0a828-14cf-4cfc-81d4-ca6b9a70f4bb
# ╠═0f0687f9-ea20-479a-811d-d112fa9a3d29
# ╠═2fb46f15-e9a7-4924-b216-13a965c25745
# ╟─0646e0ee-b3ea-428a-b19f-9083291554cd
# ╠═61019ad1-f954-4b09-ab2f-12ed406e50b7
# ╠═4da598c3-7a03-4444-a59b-5353003d3c10
# ╠═53ac14d3-73ce-433d-bb92-baa086b636c4
# ╠═6695169f-2177-4feb-a421-e94f2dda7369
# ╟─07361655-d7c4-41b3-af49-e5251224230b
# ╠═be832d6a-cc48-4be9-94b4-5b7e5521829f
# ╠═5a92c461-149e-49f7-b311-a50cb14f84f6
# ╟─295c124a-b7f2-4d5a-a705-6b60f6a2c7d9
# ╠═150697dc-416b-4b4b-9a9c-a19559926fbf
# ╟─7d8626fe-0810-4cc9-9961-98ca6c8309c1
# ╠═1ec36fb0-f237-4a83-a8a3-0f0ad8e17b82
# ╠═ae43539e-4235-4365-b5ef-08cf918d2e59
# ╟─1836a190-5688-47de-a416-44c3d02c54d7
# ╠═6b18f1f3-143d-4859-93fe-62ca1ffa3d05
# ╟─347ba300-7ee5-4aaa-9b99-e6d7180b6700
# ╠═3b0b81fb-3ed8-40a7-af58-abe5563e8800
# ╠═4ce49efc-442b-47f1-9710-3521cb2fcd43
# ╠═3989350b-1669-4aa1-87f7-c17b84905aba
# ╟─bfc5469e-ea99-4283-9384-b357d2328376
# ╠═47fb45e5-e6b6-4799-84c9-abf47bad03fc
# ╠═9d192c69-3ded-42ef-8ef6-8aa873693e6e
# ╟─9e996361-538c-4976-8b02-6884acaf9ba0
# ╠═27e37f73-56d4-4de1-a16f-2dc5706e615e
# ╟─22214dac-30e5-4f83-8ac9-42994e02dbf0
# ╠═629d4c2c-d01d-4a6c-9474-22f35de738df
# ╟─a1de57e3-a5ad-458d-a01e-665b16e94bfb
# ╠═b959c2ac-bdf0-4855-9852-e359ccd0135e
# ╠═9579a5b4-85b8-4605-8006-db3388c1dd68
# ╠═ecffdd2b-0891-41fc-9d79-5799b6f5779d
# ╠═2ac7fabb-1c81-4791-982b-b6f11fdeb5f6
# ╟─56838629-86ed-4c83-8a1c-2a9234c053fe
# ╠═1b0b94e0-ed18-4da9-8e5b-82f6d589f4bf
# ╠═f9f5065d-831a-4423-a0b8-d92f8e406f15
# ╟─b0ef6cff-590a-433d-bcb5-e08b25fcbb4b
# ╠═b45a69d5-fe8d-4ba2-8361-33aa357713f2
# ╠═f51bcc3a-e9ee-4f65-8630-7d65a4b7edce
# ╟─b48fd05c-a6b5-40e4-ae20-69201c2f9f98
# ╠═06f9f1db-5584-44d1-b6b3-ca1abc1be7fd
# ╠═1cc017ad-e891-43f0-9056-ab8c88c24c47
# ╠═cadf2547-bdf1-4f90-9120-14eb01b238a1
# ╟─248a8ccf-6553-41c1-8baa-2abc0ed3cbf9
# ╠═987239f9-82be-4e75-84a0-cfe1f7f02175
# ╠═d164e465-1397-490f-af4c-e6b7414acab8
# ╠═c825d51f-d476-4db8-bd6e-911f2c26f275
# ╠═f908d0e8-0626-486c-90c5-418cce57a5fb
# ╟─f5d9f29e-d6a6-451a-ab68-4e3cf1162da7
# ╠═e1c7176d-c897-4ffe-ba31-9608ba79e2c7
# ╠═bcc7a580-00b2-4960-b9f3-5c862ac7fcb1
# ╠═9dd79ef1-a808-48e5-b9ba-265c4b91a9c4
# ╠═2585d0ae-0afa-42ba-b266-61b4e3be9c56
# ╠═357aa7e8-47d6-4e12-8934-04b5251dc629
# ╠═ffb4c4e1-fe07-4067-a450-2ee8a1c80663
# ╟─365feac5-4f17-47b2-a6db-e0802c69da64
# ╠═a95bf0f9-0c30-4c42-9316-601c39416907
