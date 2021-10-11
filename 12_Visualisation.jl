### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 5edad594-29d1-11ec-1075-775f3b40b7fa
using Pkg; Pkg.activate(".")

# ╔═╡ 8460a040-80d5-4d82-aedd-13c2abec33b5
begin
	using CairoMakie, AlgebraOfGraphics # Plotting libraries
	using Statistics, StatsBase, MLBase
	using DataFrames, XLSX, LinearAlgebra
	using ColorSchemes, Colors
end

# ╔═╡ bb83587a-2efc-4f15-b8f7-0d761ecc0485
md"""
# Julia Academy Data Sciences

## 12. Visualisation

This is the final notebook in the Julia Academy Data Sciences course. I am working on rewriting the Jupyter notebooks into `Pluto.jl`. 

This notebook is on Visualisation, I will be using `CairoMakie` for plotting since I am trying to learn about it and how it compares to the standard `Plots` package.

Different plotting libraries have different strengths and weaknesses. As we have previously seen, `Makie` isn't able to add an `alpha` or transparency to plots, which is a significant weakness, nor is it able to easily add a label for grouped data in a scatter plot. We can do it manually, either creating a legend manually or by looping through the groups adding layers to the scatter. `VegaLite` is a very good equivalent to `ggplot_2` from `R`. It is especially good for plotting maps as we have seen in previous notebooks. 

As always, we start by loading packages, and again I am loading from my local `toml` files. This is os that I can keep a standard set of `toml`s across all notebooks in this series.
"""

# ╔═╡ 9172819f-9183-4f83-93d2-a3c50f6c3b33
md"""
### Sales Data

We will load the house sale data that we have seen in a previous notebook.

To start with, I need to create a quick dictionary with state abbreviations so that I can easily link the data & for plotting later on. I have hidden the cell for this as it's quite a long cell, but it is simply a dictionary with `state => abbrv`. You can unhide with the eye at the top left of the hidden cell.
"""

# ╔═╡ f75da571-c433-4896-9e71-cf9d078f7592
stateabbreviations = Dict("Alabama" => "AL",
    "Alaska" => "AK",
    "Arizona" => "AZ",
    "Arkansas" => "AR",
    "California" => "CA",
    "Colorado" => "CO",
    "Connecticut" => "CT",
    "Delaware" => "DE",
    "Florida" => "FL",
    "Georgia" => "GA",
    "Hawaii" => "HI",
    "Idaho" => "ID",
    "Illinois" => "IL",
    "Indiana" => "IN",
    "Iowa" => "IA",
    "Kansas" => "KS",
    "Kentucky" => "KY",
    "Louisiana" => "LA",
    "Maine" => "ME",
    "Maryland" => "MD",
    "Massachusetts" => "MA",
    "Michigan" => "MI",
    "Minnesota" => "MN",
    "Mississippi" => "MS",
    "Missouri" => "MO",
    "Montana" => "MT",
    "Nebraska" => "NE",
    "Nevada" => "NV",
    "New Hampshire" => "NH",
    "New Jersey" => "NJ",
    "New Mexico" => "NM",
    "New York" => "NY",
    "North Carolina" => "NC",
    "North Dakota" => "ND",
    "Ohio" => "OH",
    "Oklahoma" => "OK",
    "Oregon" => "OR",
    "Pennsylvania" => "PA",
    "Rhode Island" => "RI",
    "South Carolina" => "SC",
    "South Dakota" => "SD",
    "Tennessee" => "TN",
    "Texas" => "TX",
    "Utah" => "UT",
    "Vermont" => "VT",
    "Virginia" => "VA",
    "Washington" => "WA",
    "West Virginia" => "WV",
    "Wisconsin" => "WI",
    "Wyoming" => "WY", 
    "District of Columbia"=>"DC");

# ╔═╡ dfa0f4c1-baec-4271-91e4-2d50621a0602
md"""
Let's load the data. We will mostly focus on three states - New York, California, and Florida, so I will split the data into 3 dataframes, we will compare number of sales in 2020 and 2010.
"""

# ╔═╡ 7ff53e59-1ca7-420b-b21e-92bd678055b2
begin
	all_df = DataFrame(XLSX.readtable("./data/zillow_data_download_april2020.xlsx", "Sales_median_price_city", infer_eltypes=true)...);
	dropmissing!(all_df);
	# NY_df = dropmissing(filter(row -> row.StateName.=="New York", all_df));
	# CA_df = dropmissing(filter(row -> row.StateName.=="California", all_df));
	# FL_df = dropmissing(filter(row -> row.StateName.=="Florida", all_df));
	
	all_df_3 = filter(row -> row.StateName in ["New York", "California", "Florida", "Idaho"], all_df)
end

# ╔═╡ 642cf8ab-5c59-436e-b379-8c4dd90dcd0d
begin
	df_3_2020 = select(all_df_3, ["StateName", "2020-02"])
	rename!(df_3_2020, "2020-02" => :Sales20)
	
	df_3 = select(all_df_3, ["StateName", "2010-02", "2020-02"])
	rename!(df_3, "2020-02" => :Sales20)
	rename!(df_3, "2010-02" => :Sales10)
end

# ╔═╡ 528170d2-25d2-49d3-817a-ebbd7ee0ac20
md"""
#### Violin Plots

Let's start with `Makie`. We can use the `violin` function. We need to use numeric groups, so we need to convert our `StateName` field into numerical values.

We use `labelmap` and `labelencode`.

First, we plot violins for our 3 states and number of sales in Feb. 2020.

We need to then use our `labelmap` to set the axis xtick labels.
"""

# ╔═╡ ece8d9e9-3ef6-4c18-a8e3-6de666cba517
begin
	F_M_vio1 = Figure()
	Ax_M_vio1 = Axis(F_M_vio1[1,1])
	
	labs = labelmap(all_df_3.StateName)
	df_3_2020.StateNum = labelencode(labs, df_3_2020.StateName)
	
	violin!(Ax_M_vio1, df_3_2020.StateNum, df_3_2020.Sales20, datalimits=extrema)
	
	Ax_M_vio1.xticks = (1:length(labs), labs.vs)
	
	F_M_vio1
end

# ╔═╡ 6fb544b4-91c7-4083-98b3-a31582b41337
md"""
We can do a side by side comparison for each state between 2010 and 2020 by selecting the `side` option.
"""

# ╔═╡ 03f388d0-b63b-4c7a-830f-42d615bb2f0d
begin
	F_M_vio2 = Figure()
	Ax_M_vio2 = Axis(F_M_vio2[1,1])
	df_3.StateNum = labelencode(labs, df_3.StateName)
	
	violin!(Ax_M_vio2, df_3.StateNum, df_3.Sales10, datalimits=extrema, side=:left, label="2010-02")
	violin!(Ax_M_vio2, df_3.StateNum, df_3.Sales20, datalimits=extrema, side=:right,label="2020-02")
	
	Ax_M_vio2.xticks = (1:length(labs), labs.vs)
	Ax_M_vio2.ylabel = "House Price"
	
	axislegend(Ax_M_vio2)
	F_M_vio2
end

# ╔═╡ f6f077f6-ea63-4d90-8ef4-f402ffe77b25
md"""
We can add text to our violins to describe the median number of sales for each state.

First, compute a dataframe with medians.
"""

# ╔═╡ 1c2da17a-a677-4970-8611-47585f5ee434
meds = unstack(
	combine(
		groupby(
			stack(df_3, 2:3), 
			[:StateName, :StateNum, :variable]),
		:value => median => :value
		)
	)

# ╔═╡ 988f9652-53fb-416c-a183-70255725680c
begin
	meds.Med10 = string.(round.(meds.Sales10./1000, digits=1)).*"K"
	meds.Med20 = string.(round.(meds.Sales20./1000, digits=1)).*"K"
end

# ╔═╡ 440edb5f-1254-4b7e-bd2b-e8119b8b04ed
begin
	F_M_vio3 = Figure()
	Ax_M_vio3 = Axis(F_M_vio3[1,1], limits=((0.5,length(labs)+0.5), (0,nothing)), xlabel="State", ylabel="Housing Prices", aspect=1.5)
	
	violin!(Ax_M_vio3, df_3.StateNum, df_3.Sales10, datalimits=extrema, side=:left, label="2010-02")
	violin!(Ax_M_vio3, df_3.StateNum, df_3.Sales20, datalimits=extrema, side=:right,label="2020-02")
	
	# Legend
	axislegend(Ax_M_vio3,position=:rt)
	Ax_M_vio3.xticks=(1:length(labs),labs.vs)
	
	# Add the text
	text!(
		Ax_M_vio3,
		position=[(meds.StateNum[i]-0.05, meds.Sales10[i]) for i = 1:length(labs)],
		meds.Med10,
		align=(:right, :center)
		)
	
	text!(
		Ax_M_vio3,
		position=[(meds.StateNum[i]+0.05, meds.Sales20[i]) for i = 1:length(labs)],
		meds.Med20,
		align=(:left, :center)
		)

	F_M_vio3
end

# ╔═╡ 6e19f8a8-3d90-4f56-8547-87de4ecbeaf4
md"""
Producing violin plots in `Makie` is fairly easy, it takes a little time to set everything up but we've done well here. We can produce violin plots in `AlgebraOfGraphics.jl` too see the [documentation](http://juliaplots.org/AlgebraOfGraphics.jl/dev/gallery/gallery/basic%20visualizations/statistical_visualizations/#Statistical-visualizations) for an example of this.

### Bar Charts & Histograms


"""

# ╔═╡ 418bb90e-8fdb-469e-8bd0-0ceb1e6eda2a
begin
	statelabels = labelmap(all_df.StateName)
	all_df.StateNum = labelencode(statelabels, all_df.StateName)
end

# ╔═╡ 2b1f9eb2-7291-4576-94fc-ad1a8933fd8d
hist(all_df.StateNum, bins=length(statelabels))

# ╔═╡ 58b9ea5a-d3f9-467a-8eb2-ddf5b4f292bf
md"""
Let's rearrange the bars to make this easier to interpret. We `fit` a histogram to the data, rearrange by the resulting weights, plot as a bar plot.
"""

# ╔═╡ e4df649d-8ed1-49b2-a75b-83567993447d
begin
	h = fit(Histogram, all_df.StateNum, nbins=length(statelabels))
	sorted_labels = sortperm(h.weights, rev=true)
	barplot(h.weights[sorted_labels])
end

# ╔═╡ d0cca854-8084-414b-98cf-25ad759e65c5
md"""
We now want to flip the plot. We need to specify `yreversed=true` in an Axis definition - or we can do it after with `Axis.yreversed = true` where we replace `Axis` with the assigned name for the axis.
"""

# ╔═╡ e2de6465-d91f-4286-94a4-7ec489540eb5
begin
	F_bar_1 = Figure()
	Ax_bar_1 = Axis(F_bar_1[1,1], yreversed=true)
	barplot!(Ax_bar_1,h.weights[sorted_labels], direction=:x)
	F_bar_1
end

# ╔═╡ ba7dec3a-fbfa-439f-a3d6-61f8b9348848
md"""
Let's add state labels
"""

# ╔═╡ d5501bb1-f5dc-4956-86f2-19d77398aa53
stateid_sorted = statelabels.vs[sorted_labels]

# ╔═╡ c37dc0cf-7246-43bc-948d-8223ff185e0b
begin
	F_bar_2 = Figure()
	Ax_bar_2 = Axis(F_bar_2[1,1], yreversed=true, aspect = 1)
	barplot!(Ax_bar_2,h.weights[sorted_labels], direction=:x)
	
	for (i, st) in enumerate(stateid_sorted)
		pos = (h.weights[sorted_labels][i]+1, i)
		text!(Ax_bar_2, position=pos, stateabbreviations[st], textsize=11, align=(:left,:center))
	end
	
	F_bar_2
end

# ╔═╡ 964593ac-3caf-401c-b0b3-ebbadf03c91a
md"""
Looking nicer now, lets make it a bit nicer, we'll choose a colour, hide some decorations, and add some seperators along the bars for context.
"""

# ╔═╡ bbb773f4-8b6b-4558-aaa2-7150fad75e4c
begin
	F_bar_3 = Figure()
	Ax_bar_3 = Axis(F_bar_3[1,1], yreversed=true, aspect = 1, ylabel="Number of Listings")
	barplot!(Ax_bar_3,h.weights[sorted_labels], direction=:x, color=:gray80, x_gap=0.3)
	
	splt = collect(20:20:maximum(h.weights))
	vlines!(Ax_bar_3, splt, color=:white)
	
	for (i, st) in enumerate(stateid_sorted)
		pos = (h.weights[sorted_labels][i]+1, i)
		text!(Ax_bar_3, position=pos, stateabbreviations[st], textsize=11, align=(:left,:center))
	end
	
	hidedecorations!(Ax_bar_3, ticks=false, ticklabels=false)
	hideydecorations!(Ax_bar_3, ticklabels=false)
	hidespines!(Ax_bar_3, :l, :t, :r)
	F_bar_3
end

# ╔═╡ 34a5c10c-1212-4038-b58d-d9e1a0dd438b
md"""
Much nicer. I want to add an inset axis with a zoomed in bar plot for the bottom states.

Let's first do the bar plot so we can see it.
"""

# ╔═╡ 9f83a423-118d-493a-84dc-a57da9344a2d
begin
	F_bar_35 = Figure()
	Ax_bar_35 = Axis(F_bar_35[1,1], yreversed=true, aspect = 1, ylabel="Number of Listings")
	barplot!(Ax_bar_35,h.weights[sorted_labels][21:end], direction=:x, color=:gray80, x_gap=0.3)
	
	splt2 = collect(5:5:maximum(h.weights[sorted_labels][21:end]))
	vlines!(Ax_bar_35, splt2, color=:white)
	
	for (j, st) in enumerate(stateid_sorted[21:end])
		i = j + 20
		pos = (h.weights[sorted_labels][i]+0.5, j)
		text!(Ax_bar_35, position=pos, stateabbreviations[st], textsize=11, align=(:left,:center))
	end
	
	hidedecorations!(Ax_bar_35, ticks=false, ticklabels=false)
	hideydecorations!(Ax_bar_35, ticklabels=false)
	hidespines!(Ax_bar_35, :l, :t, :r)
	F_bar_35
end

# ╔═╡ a300da33-5955-4b9c-ad37-834bcbfd9c2c
md"""
Okay, we have the code for the sub-figure. We do this by choosing a location in the subplot's axis. In this example we have the main Axis spread over effectively a 5x5 grid. We can then place the second figure by specifying a location within that grid.
"""

# ╔═╡ 192bc282-ab50-4a9e-ad30-bb8e18d133c1
begin
	F_bar_4 = Figure()
	Ax_bar_4 = Axis(F_bar_4[1:5,1:5], yreversed=true, xlabel="Number of Locations", aspect = 0.7)
	barplot!(Ax_bar_4,h.weights[sorted_labels], direction=:x, color=:gray80, x_gap=0.3)
	
	vlines!(Ax_bar_4, splt, color=:white)
	
	for (i, st) in enumerate(stateid_sorted)
		pos = (h.weights[sorted_labels][i]+1, i)
		text!(Ax_bar_4, position=pos, stateabbreviations[st], textsize=11, align=(:left,:center))
	end
	
	hidedecorations!(Ax_bar_4, ticks=false, ticklabels=false, label=false)
	hideydecorations!(Ax_bar_4, label=false)
	hidespines!(Ax_bar_4, :l, :t, :r)
	
	# Now for the inset plot
	
	Ax_bar_45 = Axis(F_bar_4[2:4,3:4], yreversed=true, limits=((0,21), nothing), aspect=0.7)
	barplot!(Ax_bar_45,h.weights[sorted_labels][21:end], direction=:x, color=:gray80, x_gap=0.3)
	
	vlines!(Ax_bar_45, splt2, color=:white)
	
	for (j, st) in enumerate(stateid_sorted[21:end])
		i = j + 20
		pos = (h.weights[sorted_labels][i]+0.25, j)
		text!(Ax_bar_45, position=pos, stateabbreviations[st], textsize=11, align=(:left,:center))
	end
	
	hidedecorations!(Ax_bar_45, ticks=false, ticklabels=false)
	hideydecorations!(Ax_bar_45)
	hidespines!(Ax_bar_45, :l, :t, :r)
	Ax_bar_45.xticks=[0;splt2;20]
	
	F_bar_4
end

# ╔═╡ b62ee0c0-5f96-4a15-8b93-e2f8669095b4
md"""
Awesome. 

### Plots with Error Bars

Now we will compare state prices over the years and see how they have changed using error bars. We start with the New York data.
"""

# ╔═╡ 212d1b7f-93c0-4568-8434-5508d72023be
begin
	NY_df = filter(row -> row.StateName .== "New York", all_df)
	M_NY = Matrix(NY_df[:, 5:end-1])
	xticklabs = string.(names(NY_df)[5:end-1])
end

# ╔═╡ 3fb5ad86-112b-4393-b89a-687a8d331c00
begin
	F_lines1 = Figure();
	Ax_lines1 = Axis(F_lines1[1,1], aspect = 2);
	
	for i = 1:size(M_NY, 1)
		lines!(Ax_lines1, M_NY[i,:], label=false)
	end
	Ax_lines1.xticks = (1:4:length(xticklabs), xticklabs[1:4:end])
	Ax_lines1.xticklabelrotation = pi/2
	F_lines1
end

# ╔═╡ b4e1fdb0-187f-4425-be07-c33777839d3c
md"""
Lots of data here. We can summarise it better using a `ribbon`, which in `Makie` language is `band`. We will use a ribbon to group together the data so that we can show the median of the data and the quantiles - equivalent to error bars.

For scatter data we can add error bars to our data.
"""

# ╔═╡ bf1ee4b1-e0a7-4944-8980-a39043a43928
function get_quantile(A,q,dims)
	if dims == 1
		return [quantile(A[:,i], q) for i = 1:size(A,2)]
	elseif dims == 2
		return [quantile(A[i,:], q) for i = 1:size(A,1)]
	else
		error("Require dims to be either 1 or 2")
	end
end

# ╔═╡ 11361c6e-ce11-495d-821a-c069215ecc6a
begin
	F_lines2 = Figure();
	Ax_lines2 = Axis(F_lines2[1,1], aspect = 2);
	
	md = median(M_NY, dims=1)[:]
	q25 = get_quantile(M_NY, 0.25, 1)
	q75 = get_quantile(M_NY, 0.75, 1)
	
	# band!(Ax_lines2, 1:length(mn), mn - 2 .* sd, mn + 2 .* sd)
	band!(Ax_lines2, 1:length(md), q25, q75, color=RGBA(0.0, 0.0, 1.0, 0.4))
	lines!(Ax_lines2, md, color=:black, linewidth=2)
	
	Ax_lines2.xticks = (1:4:length(xticklabs), xticklabs[1:4:end])
	Ax_lines2.xticklabelrotation = pi/2
	F_lines2
end

# ╔═╡ c45741a0-f6e9-47e6-b4af-bca8376f620e
md"""
Let's now try to write a function for any state so we can compare states.
"""

# ╔═╡ 37cb7572-ca15-4f47-b9c4-bb68c79b597a
function plot_state!(axid::Axis, state::String, color::Tuple{Float64,Float64,Float64}, alpha::Float64)
	df = filter(row -> row.StateName .== state, all_df)
	M = Matrix(df[:,5:end-1])
	ticklabels = string.(names(df)[5:end-1])
	
	md = median(M, dims=1)[:]
	q25 = get_quantile(M, 0.25, 1)
	q75 = get_quantile(M, 0.75, 1)
	
	band!(axid, 1:length(ticklabels), q25, q75, color=RGBA(color[1],color[2],color[3],alpha))
	lines!(axid, md, color=RGB(color[1], color[2], color[3]), label=stateabbreviations[state], linewidth=3)
end

# ╔═╡ d934537b-402c-4870-b191-2ddf2dce980f
begin
	F1 = Figure()
	Ax1 = Axis(F1[1,1])
	Ax1.xticks = (1:6:length(xticklabs), xticklabs[1:6:end])
	Ax1.xticklabelrotation = pi/2
	Ax1.ylabel = "Prices"
	hidedecorations!(Ax1, label=false, ticks=false, ticklabels=false)
	hidespines!(Ax1, :t, :r)
	
	plot_state!(Ax1, "Indiana", (1.0, 0.0, 0.0), 0.2)
	plot_state!(Ax1, "Ohio", (0.0, 1.0, 0.0), 0.2)
	plot_state!(Ax1, "Idaho", (0.0, 0.0, 1.0), 0.2)
	axislegend(Ax1)
	F1
end

# ╔═╡ fde8a274-4df5-4c66-8074-1ea9c4804fd1
begin
	F2 = Figure()
	Ax2 = Axis(F2[1,1])
	Ax2.xticks = (1:6:length(xticklabs), xticklabs[1:6:end])
	Ax2.xticklabelrotation = pi/2
	Ax2.ylabel = "Prices"
	hidedecorations!(Ax2, label=false, ticks=false, ticklabels=false)
	hidespines!(Ax2, :t, :r)
	
	plot_state!(Ax2, "California", (1.0, 0.0, 0.0), 0.2)
	plot_state!(Ax2, "Idaho", (0.0, 0.0, 1.0), 0.2)
	axislegend(Ax2)
	F2
end

# ╔═╡ 90eb74ea-5c96-47ac-a16d-c19e11aa1c72
md"""
House prices are rising very rapidly in Idaho, compared to Indiana and Ohio.
"""

# ╔═╡ 1be7874b-5fdb-4262-b1a3-8ba7c29b7265
md"""
### Plot with Double Axis

In standard plots we can add an axis with `twinx`. There is no such function in `Makie`, however, we can specify a second axis in the same location and choose to have the labels on the right - or even the top!

A good thing to do is use `linkxaxis`, or `linkyaxis` as required. Note that I have set the `rightspinevisible` on the first axis to be `false`.
"""

# ╔═╡ 4bab4430-196a-4ef7-9859-f115b12878e8
begin
	x1 = rand(10)
	x2 = rand(15)*100
	
	Fdoub = Figure()
	AxDoubL = Axis(Fdoub[1,1], rightspinevisible=false)
	AxDoubR = Axis(Fdoub[1,1], yticklabelcolor = :red, yaxisposition = :right, rightspinecolor=:red)
	linkxaxes!(AxDoubL, AxDoubR)
	
	l1 = lines!(AxDoubL, x1, color=:black, label="A")
	l2 = lines!(AxDoubR, x2, color=:red)
	Legend(Fdoub[1,2], [l1, l2], ["A", "B"])
	Fdoub
end

# ╔═╡ 2f048b21-310a-4473-b530-cb8715398a39
begin
	Fdoub2 = Figure()
	asp = 1.4
	AxDoub2L = Axis(Fdoub2[1,1], rightspinevisible=false, leftspinecolor=:blue, aspect=asp)
	AxDoub2R = Axis(Fdoub2[1,1], bottomspinevisible=false, leftspinevisible=false, yticklabelcolor = :red, yaxisposition = :right, rightspinecolor=:red, aspect=asp)
	linkxaxes!(AxDoub2L, AxDoub2R)
	
	AxDoub2L.xticks = (1:nrow(NY_df), NY_df.RegionName);
	AxDoub2L.xticklabelrotation=pi/2
	
	hidexdecorations!(AxDoub2R)
		
	town_medians = median(M_NY, dims=2)[:]
	town_q25 = get_quantile(M_NY, 0.25, 2)
	town_q75 = get_quantile(M_NY, 0.75, 2)
	
	doub_bnd = band!(AxDoub2L, 1:nrow(NY_df), town_q25, town_q75, color=RGBA(0.0, 0.0, 1.0, 0.3))
	lines!(AxDoub2L, town_medians, color=:blue)
	ldoub_n = lines!(AxDoub2R, NY_df.SizeRank, color=:red)
	
	axlegs = ["Prices (right)", "Rank (left)"]
	Legend(Fdoub2[1,2], [doub_bnd, ldoub_n], axlegs)
	Fdoub2
end

# ╔═╡ 99fdf603-89f7-4399-a164-0c8417af97d4
md"""
It looks like lower rank regions (smaller towns) have higher house prices & vice-versa.

### High Dimensional Data - 2D plots

We will compare 2020 prices in Feb against those in 2010 for California using scatter plots. We will make use of colour scales to help us visualise the data.
"""

# ╔═╡ ce016a02-7062-455f-9e8c-9f09a988887b
begin
	CA_df = dropmissing(filter(row -> row.StateName.=="California", all_df));
	select!(CA_df, ["SizeRank", "2010-02", "2020-02" ])
	rename!(CA_df, "2020-02" => :Sales20)
	rename!(CA_df, "2010-02" => :Sales10)
end

# ╔═╡ 7f98da2f-24cf-4a52-9491-89ebaa8e8fab
begin
	F_sc1 = Figure()
	Ax_sc1 = Axis(F_sc1[1,1])
	scatter!(Ax_sc1,CA_df.Sales10, CA_df.Sales20)
	Ax_sc1.xlabel="2010 Prices"
	Ax_sc1.ylabel="2020 Prices"
	F_sc1
end

# ╔═╡ 9752eb48-97c4-4703-ab3c-69601d9e926b
md"""
We can colour the points by a category, or scale. In this case we will colour the points by the size of the town.
"""

# ╔═╡ e4761e7f-38ba-494e-80b6-c26b649cd73e
begin
	F_sc2 = Figure()
	Ax_sc2 = Axis(F_sc2[1,1])
	sc = scatter!(Ax_sc2,CA_df.Sales10, CA_df.Sales20, color=CA_df.SizeRank, colormap=:autumn1)
	Ax_sc2.xlabel="2010 Prices"
	Ax_sc2.ylabel="2020 Prices"
	
	Colorbar(F_sc2[1,2], limits=extrema(CA_df.SizeRank), colormap=:autumn1)
	F_sc2
end

# ╔═╡ 8d017024-907f-4a6a-aa14-e4c4085b6fb8
md"""
This agrees with our previous result from NY, that smaller towns have higher house prices.

### Summary

We have generated some good visualisations of the house price data from various US states. We've explored Violin plots which indicate the distributions of data in a way than allow us to compare across different variables or factors. Histograms and bar charts can be used to visualise relationships between categorial variables, for example we see that California, Florida have the most number of towns in the dataset. Line and band plots can be used to visualise time series, for example how house prices have changed over time. We can add addional axis to plots in order to compare two factors that may have different scales. Finally we can produce scatter plots to visualise high dimensional data, using colour scales can allow us to visualise a 3rd dimension without having to generate a 3d plot. 
"""

# ╔═╡ Cell order:
# ╠═5edad594-29d1-11ec-1075-775f3b40b7fa
# ╟─bb83587a-2efc-4f15-b8f7-0d761ecc0485
# ╠═8460a040-80d5-4d82-aedd-13c2abec33b5
# ╟─9172819f-9183-4f83-93d2-a3c50f6c3b33
# ╟─f75da571-c433-4896-9e71-cf9d078f7592
# ╟─dfa0f4c1-baec-4271-91e4-2d50621a0602
# ╠═7ff53e59-1ca7-420b-b21e-92bd678055b2
# ╠═642cf8ab-5c59-436e-b379-8c4dd90dcd0d
# ╟─528170d2-25d2-49d3-817a-ebbd7ee0ac20
# ╠═ece8d9e9-3ef6-4c18-a8e3-6de666cba517
# ╟─6fb544b4-91c7-4083-98b3-a31582b41337
# ╠═03f388d0-b63b-4c7a-830f-42d615bb2f0d
# ╟─f6f077f6-ea63-4d90-8ef4-f402ffe77b25
# ╠═1c2da17a-a677-4970-8611-47585f5ee434
# ╠═988f9652-53fb-416c-a183-70255725680c
# ╠═440edb5f-1254-4b7e-bd2b-e8119b8b04ed
# ╟─6e19f8a8-3d90-4f56-8547-87de4ecbeaf4
# ╠═418bb90e-8fdb-469e-8bd0-0ceb1e6eda2a
# ╠═2b1f9eb2-7291-4576-94fc-ad1a8933fd8d
# ╟─58b9ea5a-d3f9-467a-8eb2-ddf5b4f292bf
# ╠═e4df649d-8ed1-49b2-a75b-83567993447d
# ╟─d0cca854-8084-414b-98cf-25ad759e65c5
# ╠═e2de6465-d91f-4286-94a4-7ec489540eb5
# ╟─ba7dec3a-fbfa-439f-a3d6-61f8b9348848
# ╠═d5501bb1-f5dc-4956-86f2-19d77398aa53
# ╠═c37dc0cf-7246-43bc-948d-8223ff185e0b
# ╟─964593ac-3caf-401c-b0b3-ebbadf03c91a
# ╠═bbb773f4-8b6b-4558-aaa2-7150fad75e4c
# ╟─34a5c10c-1212-4038-b58d-d9e1a0dd438b
# ╠═9f83a423-118d-493a-84dc-a57da9344a2d
# ╟─a300da33-5955-4b9c-ad37-834bcbfd9c2c
# ╠═192bc282-ab50-4a9e-ad30-bb8e18d133c1
# ╟─b62ee0c0-5f96-4a15-8b93-e2f8669095b4
# ╠═212d1b7f-93c0-4568-8434-5508d72023be
# ╠═3fb5ad86-112b-4393-b89a-687a8d331c00
# ╠═b4e1fdb0-187f-4425-be07-c33777839d3c
# ╠═bf1ee4b1-e0a7-4944-8980-a39043a43928
# ╠═11361c6e-ce11-495d-821a-c069215ecc6a
# ╟─c45741a0-f6e9-47e6-b4af-bca8376f620e
# ╠═37cb7572-ca15-4f47-b9c4-bb68c79b597a
# ╠═d934537b-402c-4870-b191-2ddf2dce980f
# ╠═fde8a274-4df5-4c66-8074-1ea9c4804fd1
# ╟─90eb74ea-5c96-47ac-a16d-c19e11aa1c72
# ╟─1be7874b-5fdb-4262-b1a3-8ba7c29b7265
# ╠═4bab4430-196a-4ef7-9859-f115b12878e8
# ╠═2f048b21-310a-4473-b530-cb8715398a39
# ╟─99fdf603-89f7-4399-a164-0c8417af97d4
# ╠═ce016a02-7062-455f-9e8c-9f09a988887b
# ╠═7f98da2f-24cf-4a52-9491-89ebaa8e8fab
# ╟─9752eb48-97c4-4703-ab3c-69601d9e926b
# ╠═e4761e7f-38ba-494e-80b6-c26b649cd73e
# ╟─8d017024-907f-4a6a-aa14-e4c4085b6fb8
