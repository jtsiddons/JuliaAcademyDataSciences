### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 69de90d3-7874-475e-8f84-48e1afc6970d
using Lasso, RDatasets, MLBase, CairoMakie, DecisionTree, Distances, NearestNeighbors, Random, LinearAlgebra, DataStructures, DataFrames, LIBSVM, CategoricalArrays, PlutoUI

# ╔═╡ 9c9ca920-2508-11ec-2daf-0f207b54646f
md"""
# Julia Academy Data Sciences Course

## 06. Classification

This is a `Pluto.jl` write-up of the 6th lesson in the Julia Academy Data Sciences course. 

Classification is the task of predicting a label for a given observation. For example, you are given certain physical descriptions of some animal that you are to determine if it is a dog or a cat. In this tutorial we will be classifying iris flowers from the famous `iris` dataset. 

We will use multiple classifiers during this notebook and we will compare their effectiveness. 

First, we load in the necessary packages. In this case I will be replacing `GLMNet` with `Lasso.jl` - I like this package as it is a fully Julia solution to penalised regression - although I wouldn't typically associate regression with classification - unless we use some form of binomial or multinomial regression.
"""

# ╔═╡ df785c5a-b771-42e9-867a-b6c2f552af6e
md"""
### Model Accuracy

We first define a function to test the accuracy of all our models. Essentially we are comparing the predicted values with the true values for our classification.

This is a simple function that purely compares values. In reality we would want to be a bit more rigourous with our function and include some error processing.
"""

# ╔═╡ 15d251ff-2ec1-42e9-ac22-cbc69b082293
findaccuracy(predictedvals, groundtruthvals) = sum(predictedvals .== groundtruthvals)/length(groundtruthvals)

# ╔═╡ c38bd4f6-0193-4267-8242-fb19fdbd78b6
md"""
### Iris Data

Load and prepare the dataset.
"""

# ╔═╡ a3e25d91-31a3-45d1-8b5f-33adc009dd8d
irisdf = dataset("datasets", "iris")

# ╔═╡ 1bea6213-2e63-4e05-b10f-df1ff6eb6a04
describe(irisdf)

# ╔═╡ fadfde5f-b45e-4232-a383-930e768f9769
# To print into Pluto.jl we need to use with_terminal() from PlutoUI.jl
with_terminal() do 
	for (i,(n,t)) in enumerate(zip(names(irisdf),typeof.(eachcol(irisdf))))
		println([i, n, t])
	end
end

# ╔═╡ 21f476a2-8678-489d-8e69-595edbca8665
md"""
We can see that the `Species` column is of type: `CategoricalValue`. We can see the possible values it can take using `levels`. Since the original version of this class the `CategoricalArrays` methods have changed, rather than having a `CategoricalArray` of strings we have a `Vector` of `CategoricalValue`s.
"""

# ╔═╡ e9bc4d3b-23ab-4d6b-ae46-0a3d71817ac4
levels(irisdf.Species)

# ╔═╡ 2ec10458-a8f8-462a-982b-53aeacb4e40e
md"""
We have three species into which we want to classify the data. Let's split out the data into predictors and labels. We will need to encode the labels to a numeric value.

We use `labelencode` and `labelmap` to transform `irisdf.Species` to numerical values.
"""

# ╔═╡ 02e71249-4ca1-44ca-a660-7998cf580cf8
iris_predictors = Matrix(irisdf[:,1:4])

# ╔═╡ 92afc019-183f-4bea-a57b-86296cc9f207
iris_classes = labelencode(labelmap(irisdf.Species), irisdf.Species)

# ╔═╡ 34a9821f-b0ec-4bce-8d9c-9c5abbdc4a4e
md"""
#### Training and Testing data subsets

When doing classification we need to train our models. A common approach is to randomly seperate the data into `training` and `testing` subsets of the data.

In this case, we want to ensure that we have some of all possible classifications in both sets.

We do this using a function. This function takes as its arguments the vector of clasification labels (in our case `iris_classes`) and a proportion, which is how much of the data will be assigned to training data. It loops through each unique value in `labels` randomly selecting `proportion` of values from the labels and pushing it to `keepids`. For example, if we select `proportion=0.7`, we get 70% of the data for each unique classification. We return the row ids which will be used to seperate the data.
"""

# ╔═╡ 51bc494e-d0b1-4c9b-9ed6-932e83b2d836
function perclass_splits(labels, proportion)
	uids = unique(labels)
	keepids = []
	for ui in uids
		curids = findall(labels.==ui)
		rowids = randsubseq(curids, proportion)
		push!(keepids, rowids...)
	end
	return keepids
end

# ╔═╡ 1d64d87d-4335-4975-85d1-50d5b2715856
begin
	training_ids = perclass_splits(iris_classes, 0.75)
	test_ids = setdiff(1:nrow(irisdf), training_ids)
end;

# ╔═╡ 88009e78-6f3c-4eba-ae93-34b383d6ad8b
md"""
We write one additional funciton. This assigns a class to our predictions, which may not be exactly `[1,2,3]`.
"""

# ╔═╡ 243e5b08-859d-444c-b40b-901ddffc6caf
assign_class(predicted_values) = argmin(abs.(predicted_values.- [1,2,3]));

# ╔═╡ d64b714f-3468-4e2d-8311-532511e673aa
md"""
### Lasso Regression

Lasso regression is a penalised regression whereby we penalise the coefficients by adding an L1 penalty to their values. With large numbers of predictors this can have the effect of setting some of them to zero.

We first define a `LassoPath`, which generates a solution for a series of λs (the penalty parameter). We can specify a set of λs, although for Lasso the method can generate a suitable sequence (it cannot do this for Ridge regression later).

From there, we then `selectmodel` to select the best performing model according to some error function - in this case we are choosing cross-validation with a k-fold that  assigns 2/3 of the data as training and generates a series of models from random training subsets. Note these training subsets are actually subsets of our `training` data.
"""

# ╔═╡ 8ab6ed40-3973-4a4b-a382-9514178eb616
begin
	cv_n = length(training_ids)
	cv_k = Int(floor(cv_n/3))
	(cv_n,cv_k)
end

# ╔═╡ a232ed9b-eb48-4fde-800e-2cf42784724c
iris_lasso = fit(
	LassoPath, 
	iris_predictors[training_ids,:], 
	iris_classes[training_ids],
	α=1
	)

# ╔═╡ 9f4bc501-9f94-4075-96e5-d4ea4749a165
path = iris_lasso.λ

# ╔═╡ b45183ca-aa32-45d5-9647-c4780974352c
iris_lasso_select = selectmodel(iris_lasso, MinCVmse(Kfold(cv_n, cv_k)))
# iris_lasso_select = selectmodel(iris_lasso, MinBIC())

# ╔═╡ a0133190-c1af-42cf-a3a5-1e962ebf9768
lasso_predictions = StatsBase.predict(
	iris_lasso_select, 
	hcat(
		ones(length(test_ids)),
		iris_predictors[test_ids,:]
	)
)

# ╔═╡ e41f6dad-b33a-42c2-b747-7dce2aa465c7
begin
	lasso_predictions_class=assign_class.(lasso_predictions)
	lasso_accuracy = findaccuracy(lasso_predictions_class, iris_classes[test_ids])
	lasso_accuracy = round(lasso_accuracy, digits = 4)
end

# ╔═╡ 47444875-a0ee-4615-a400-ce23d6a41b97
md"""
### Ridge Regression

Ridge regression is similar to Lasso although it adds and L2 penalty to the coefficient values. In this case this reduces coefficients to a similar value - it heavily penalises large coefficients. This is a good tool to use for heavily correlated data.

Here we need to specifiy the λ-path. We use the same set of λs used in the Lasso model.
"""

# ╔═╡ c409c7ea-d276-4b82-abe6-8a2274d1d41b
iris_ridge = fit(
	LassoPath, 
	iris_predictors[training_ids,:], 
	iris_classes[training_ids],
	λ=path,
	α=0
	)

# ╔═╡ d81a13b3-e4fa-405d-8a48-607c323438c0
iris_ridge_select = selectmodel(iris_ridge, MinCVmse(Kfold(cv_n, cv_k)))

# ╔═╡ 1cd25877-7063-4677-a4b4-9ccb39fe7a04
ridge_predictions = StatsBase.predict(
	iris_ridge_select, 
	hcat(
		ones(length(test_ids)),
		iris_predictors[test_ids,:]
	)
)

# ╔═╡ aae7b504-6551-409e-b11b-207e43ef91d9
begin
	ridge_predictions_class=assign_class.(ridge_predictions)
	ridge_accuracy = findaccuracy(ridge_predictions_class, iris_classes[test_ids])
	ridge_accuracy = round(ridge_accuracy, digits = 4)
end

# ╔═╡ eb21a4ee-9e06-4921-989a-1ffdfc6d8676
md"""
### Elastic Net

Elastic Net is a combination of Lasso and Ridge and applys a combination of an L1 and an L2 penalty to the coefficients. 

We set α=0.5 here and use the same λ-path as generated using the Lasso regression model.
"""

# ╔═╡ 6f7f0468-e8bf-4027-aee9-9f7e8d599386
iris_en = fit(
	LassoPath, 
	iris_predictors[training_ids,:], 
	iris_classes[training_ids],
	λ=path,
	α=0.5
	)

# ╔═╡ 0fed509a-8593-4f2b-8f21-5f7ef8f5c777
iris_en_select = selectmodel(iris_en, MinCVmse(Kfold(cv_n, cv_k)))

# ╔═╡ bd562312-8e13-48ca-ab9a-a5087d5259a2
en_predictions = StatsBase.predict(
	iris_en_select,
	hcat(
		ones(length(test_ids)),
		iris_predictors[test_ids,:]
	)
)

# ╔═╡ a7ee7908-6cd4-4d04-9d0a-04faa161dd9a
begin
	en_predictions_class=assign_class.(en_predictions)
	en_accuracy = findaccuracy(en_predictions_class, iris_classes[test_ids])
	en_accuracy = round(en_accuracy, digits = 4)
end

# ╔═╡ 27dff5f6-5fb3-4246-859d-b1656c01f77d
md"""
### Variable Elastic Net

Here is a fun exercise, we can vary α and see how it changes the coefficients.
"""

# ╔═╡ eb2d1ff7-6363-47a3-a0bd-8f43b668e034
@bind α PlutoUI.Slider(0.0:0.05:1.0)

# ╔═╡ 0cf92df3-3ab4-4af5-94fa-b733c6cb75bb
"α = "*string(α)	

# ╔═╡ 38cc3c3f-65f4-4747-a52c-76596611f3f3
begin
	iris_var_en = fit(
		LassoPath, 
		iris_predictors[training_ids,:], 
		iris_classes[training_ids],
		λ=path,
		α=α
	)
	iris_var_en_select = selectmodel(iris_var_en, MinCVmse(Kfold(cv_n, cv_k)))
	en_var_predictions = StatsBase.predict(iris_var_en_select, hcat(ones(length(test_ids)),iris_predictors[test_ids,:]))
	en_var_predictions_class=assign_class.(en_var_predictions)
	en_var_accuracy = findaccuracy(en_var_predictions_class, iris_classes[test_ids])
	en_var_accuracy = round(en_var_accuracy, digits = 4)
	"accuracy = "*string(en_var_accuracy)
end

# ╔═╡ d3401731-17ec-4e70-bd06-6494dda28740
begin
	leg = ["Intercept"]
	push!(leg, names(irisdf)[1:4]...)
	F = Figure();
	Ax = Axis(F[1,1])
	for i = 1:5
		lines!(Ax, path, Vector(coef(iris_var_en)[i,:]),label=leg[i], linewidth=3)
	end
	hlines!(Ax,[0.0],color=:black, linestyle=:dash)
	Ax.xlabel = "λ"
	Ax.ylabel = "Coefficients"
	titlestr = α == 1 ? "Lasso" : (α == 0 ? "Ridge" : "Elastic Net: α="*string(α))
	Ax.title = titlestr
	axislegend(position = :lt)
	F
end

# ╔═╡ 30797a40-1ec5-4e73-a34a-8998d53f3189
md"""
Here for all of the Penalised Regression methods (Lasso, Ridge, and Elastic Net) we get a perfect result for the classification. This is impressive.

Results so far

| Method | Accuracy |
| --- | --- |
| Lasso | $lasso_accuracy |
| Ridge | $ridge_accuracy |
| Elastic Net | $en_accuracy |

### Decision Trees

We use the `DecisionTree.jl` package.
"""

# ╔═╡ 3e8ce39f-4da5-43c3-a00e-7042020405ef
dt_model = DecisionTreeClassifier(max_depth = 2)

# ╔═╡ ae93ea1a-b260-4d6f-94d3-bee541b4c80d
DecisionTree.fit!(dt_model, iris_predictors[training_ids,:], iris_classes[training_ids])

# ╔═╡ b7790967-56f9-4998-8455-29cad57fea0e
dt_predictions = DecisionTree.predict(dt_model, iris_predictors[test_ids,:])

# ╔═╡ a5d7ee71-f84b-4ab2-ad10-49cd7d97d779
begin
	dt_accuracy = findaccuracy(dt_predictions, iris_classes[test_ids])
	dt_accuracy = round(dt_accuracy, digits = 4)
end

# ╔═╡ d1c4c2f2-2677-45fb-bb29-ee551ee265b7
md"""
### Random Forest

We can use the `RandomForestClassifier` from `DecisionTree.jl` for this.
"""

# ╔═╡ 6e9aaac2-2899-44fd-8d0b-f26ab5a11cf9
rf_model = RandomForestClassifier(n_trees=50)

# ╔═╡ 916ee2fd-f35f-4861-9c0f-fc15c286923e
DecisionTree.fit!(rf_model, iris_predictors[training_ids,:], iris_classes[training_ids])

# ╔═╡ 3232cafd-c628-4e1b-baf5-9f6dfdb28cd4
rf_predictions = DecisionTree.predict(rf_model, iris_predictors[test_ids,:])

# ╔═╡ 54c93eb1-c739-443b-b127-2dcd7765d70f
begin
	rf_accuracy = findaccuracy(rf_predictions, iris_classes[test_ids])
	rf_accuracy = round(rf_accuracy, digits = 4)
end

# ╔═╡ 6ad975ac-c986-4470-940f-290c4ee9cb60
md"""
Adding the Decision Tree results to the table, we have

Results so far

| Method | Accuracy |
| --- | --- |
| Lasso | $lasso_accuracy |
| Ridge | $ridge_accuracy |
| Elastic Net | $en_accuracy |
| Decision Tree | $dt_accuracy |
| Random Forest | $rf_accuracy |

Note that re-running either of the last two methods gives different results.

### Nearest Neighbours

Use `NearestNeighbors.jl`. This method is weird, and I don't really follow this. It doesn't require the classes until the end. I think it tries to find samples that are similar. 

I will look at the guide [here](https://scikit-learn.org/stable/modules/neighbors.html)
"""

# ╔═╡ c95feaed-6086-4dc2-a57c-fdca6076bc3d
kdtree_iris = KDTree(iris_predictors[training_ids,:]')

# ╔═╡ f070024b-76f9-4160-8f29-e24ad8e764ee
idxs, dists = knn(kdtree_iris, iris_predictors[test_ids,:]', 5, true)

# ╔═╡ 510fc56d-b8c3-46c8-a894-0b6600155e09
md"""
I think we have taken each sample in the testing set and tried to find similar samples in the training set. We find 5 samples in the training which are similar to the sample from the test set.

We then assign classification based on votes. Each vote is the class of corresponding neighbour. The classification is the class with the most votes.
"""

# ╔═╡ 00b3d517-9716-4698-bf84-71bc612f4da3
begin
	c = iris_classes[training_ids][hcat(idxs...)]
	possible_labels = map(i->counter(c[:,i]),1:size(c,2))
	predictions_NN = map(i->parse(Int,string(string(argmax(possible_labels[i])))),1:size(c,2))
	nn_accuracy = findaccuracy(predictions_NN,iris_classes[test_ids])
	nn_accuracy = round(nn_accuracy, digits = 4)
end

# ╔═╡ 487cd4d7-28f9-4500-a0a9-f14e572d3d81
md"""
### Support Vector Machines (SVM)

We use `LIBSVM.jl`.
"""

# ╔═╡ 7e74bf71-e487-4089-9829-25b387186e1f
svm_iris = svmtrain(iris_predictors[training_ids,:]', iris_classes[training_ids])

# ╔═╡ 32a31142-a173-4b11-91f0-1817ef5df062
svm_iris_predict, svm_iris_decision = svmpredict(svm_iris, iris_predictors[test_ids,:]') 

# ╔═╡ dab885dd-9ba9-4db4-a988-7f7c4c1e6dfe
begin
	svm_accuracy = findaccuracy(svm_iris_predict, iris_classes[test_ids])
	svm_accuracy = round(svm_accuracy, digits = 4)
end

# ╔═╡ fd4f92c6-8b2e-492f-8f96-8952673837dc
begin
	models = [
		"Lasso",
		"Ridge",
		"Elastic Net",
		titlestr,
		"Decision Tree",
		"Random Forest",
		"Nearest Neighbours",
		"SVM"
	]
	accuracies = [
		lasso_accuracy,
		ridge_accuracy,
		en_accuracy,
		en_var_accuracy,
		dt_accuracy,
		rf_accuracy,
		nn_accuracy,
		svm_accuracy
	]
	modeldf = DataFrame(model=models, accuracy=accuracies)
	winids = findall(accuracies.==maximum(accuracies))
	nwinners = length(winids) == 1 ? "method is" : "methods are"
	last_prefix = length(winids) == 1 ? "" : "and "
	winner = unique(models[winids]) .* ", ";
	winner[end] = last_prefix*winner[end][1:end-2]
end;

# ╔═╡ 093f955a-5562-402e-bb30-b4ce11742757
md"""
Now we can put this together and combine all results.

$modeldf

The winning $nwinners $winner.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
DecisionTree = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
LIBSVM = "b1bec4e5-fd48-53fe-b0cb-9723c09d164b"
Lasso = "b4fcebef-c861-5a0f-a7e2-ba9dc32b180a"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLBase = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
NearestNeighbors = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
RDatasets = "ce6b1742-4840-55fa-b093-852dadbb1d8b"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
CairoMakie = "~0.6.5"
CategoricalArrays = "~0.10.1"
DataFrames = "~1.2.2"
DataStructures = "~0.18.10"
DecisionTree = "~0.10.10"
Distances = "~0.10.4"
LIBSVM = "~0.7.0"
Lasso = "~0.6.2"
MLBase = "~0.8.0"
NearestNeighbors = "~0.4.9"
PlutoUI = "~0.7.1"
RDatasets = "~0.7.5"
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

[[DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "1edc3eb6cd0ec2b5193ac6d37c1b1310044550fe"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.3"

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

[[DecisionTree]]
deps = ["DelimitedFiles", "Distributed", "LinearAlgebra", "Random", "ScikitLearnBase", "Statistics", "Test"]
git-tree-sha1 = "8b58db7954a6206399d9f66ef1a328da8c0f1d19"
uuid = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
version = "0.10.10"

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

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a9b99024b57d12fb19892d3f2230856f6d9671a4"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.17"

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

[[LIBLINEAR]]
deps = ["Libdl", "SparseArrays", "liblinear_jll"]
git-tree-sha1 = "81e40115c23acca9dfa30944050096b958271e5a"
uuid = "2d691ee1-e668-5016-a719-b2531b85e0f5"
version = "0.6.0"

[[LIBSVM]]
deps = ["LIBLINEAR", "LinearAlgebra", "ScikitLearnBase", "SparseArrays", "libsvm_jll"]
git-tree-sha1 = "729ea2db931587c983d0ef6691b62de5005c5570"
uuid = "b1bec4e5-fd48-53fe-b0cb-9723c09d164b"
version = "0.7.0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Lasso]]
deps = ["DSP", "Distributions", "GLM", "LinearAlgebra", "MLBase", "Random", "Reexport", "SparseArrays", "StatsBase", "StatsModels"]
git-tree-sha1 = "ece74b2a022918c8af0ef7724468e925a9609152"
uuid = "b4fcebef-c861-5a0f-a7e2-ba9dc32b180a"
version = "0.6.2"

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

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MLBase]]
deps = ["IterTools", "Random", "Reexport", "StatsBase", "Test"]
git-tree-sha1 = "f63a8d37429568b8c4384d76c4a96fc2897d6ddf"
uuid = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
version = "0.8.0"

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

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "3927848ccebcc165952dc0d9ac9aa274a87bfe01"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.2.20"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

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
git-tree-sha1 = "2537ed3c0ed5e03896927187f5f2ee6a4ab342db"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.14"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "Logging", "Markdown", "Random", "Suppressor"]
git-tree-sha1 = "45ce174d36d3931cd4e37a47f93e07d1455f038d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.1"

[[PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

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
deps = ["Pkg"]
git-tree-sha1 = "7b1d07f411bc8ddb7977ec7f377b97b158514fe0"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "0.2.0"

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

[[ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

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
deps = ["Rmath", "SpecialFunctions"]
git-tree-sha1 = "ced55fd4bae008a8ea12508314e725df61f0ba45"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.7"

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

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

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

[[liblinear_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "6a4a6a3697269cb2da57e698e9318972d88de0bb"
uuid = "275f1f90-abd2-5ca1-9ad8-abd4e3d66eb7"
version = "2.30.0+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libsvm_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl", "Pkg"]
git-tree-sha1 = "ac78676ee5b1707de969d68d0a39db71f222925d"
uuid = "08558c22-525a-5d2a-acf6-0ac6658ffce4"
version = "3.24.0+1"

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
# ╟─9c9ca920-2508-11ec-2daf-0f207b54646f
# ╠═69de90d3-7874-475e-8f84-48e1afc6970d
# ╟─df785c5a-b771-42e9-867a-b6c2f552af6e
# ╠═15d251ff-2ec1-42e9-ac22-cbc69b082293
# ╟─c38bd4f6-0193-4267-8242-fb19fdbd78b6
# ╠═a3e25d91-31a3-45d1-8b5f-33adc009dd8d
# ╠═1bea6213-2e63-4e05-b10f-df1ff6eb6a04
# ╠═fadfde5f-b45e-4232-a383-930e768f9769
# ╟─21f476a2-8678-489d-8e69-595edbca8665
# ╠═e9bc4d3b-23ab-4d6b-ae46-0a3d71817ac4
# ╟─2ec10458-a8f8-462a-982b-53aeacb4e40e
# ╠═02e71249-4ca1-44ca-a660-7998cf580cf8
# ╠═92afc019-183f-4bea-a57b-86296cc9f207
# ╟─34a9821f-b0ec-4bce-8d9c-9c5abbdc4a4e
# ╠═51bc494e-d0b1-4c9b-9ed6-932e83b2d836
# ╠═1d64d87d-4335-4975-85d1-50d5b2715856
# ╟─88009e78-6f3c-4eba-ae93-34b383d6ad8b
# ╠═243e5b08-859d-444c-b40b-901ddffc6caf
# ╟─d64b714f-3468-4e2d-8311-532511e673aa
# ╠═8ab6ed40-3973-4a4b-a382-9514178eb616
# ╠═a232ed9b-eb48-4fde-800e-2cf42784724c
# ╠═9f4bc501-9f94-4075-96e5-d4ea4749a165
# ╠═b45183ca-aa32-45d5-9647-c4780974352c
# ╠═a0133190-c1af-42cf-a3a5-1e962ebf9768
# ╠═e41f6dad-b33a-42c2-b747-7dce2aa465c7
# ╟─47444875-a0ee-4615-a400-ce23d6a41b97
# ╠═c409c7ea-d276-4b82-abe6-8a2274d1d41b
# ╠═d81a13b3-e4fa-405d-8a48-607c323438c0
# ╠═1cd25877-7063-4677-a4b4-9ccb39fe7a04
# ╠═aae7b504-6551-409e-b11b-207e43ef91d9
# ╟─eb21a4ee-9e06-4921-989a-1ffdfc6d8676
# ╠═6f7f0468-e8bf-4027-aee9-9f7e8d599386
# ╠═0fed509a-8593-4f2b-8f21-5f7ef8f5c777
# ╠═bd562312-8e13-48ca-ab9a-a5087d5259a2
# ╠═a7ee7908-6cd4-4d04-9d0a-04faa161dd9a
# ╟─27dff5f6-5fb3-4246-859d-b1656c01f77d
# ╟─eb2d1ff7-6363-47a3-a0bd-8f43b668e034
# ╟─0cf92df3-3ab4-4af5-94fa-b733c6cb75bb
# ╟─38cc3c3f-65f4-4747-a52c-76596611f3f3
# ╟─d3401731-17ec-4e70-bd06-6494dda28740
# ╟─30797a40-1ec5-4e73-a34a-8998d53f3189
# ╠═3e8ce39f-4da5-43c3-a00e-7042020405ef
# ╠═ae93ea1a-b260-4d6f-94d3-bee541b4c80d
# ╠═b7790967-56f9-4998-8455-29cad57fea0e
# ╠═a5d7ee71-f84b-4ab2-ad10-49cd7d97d779
# ╟─d1c4c2f2-2677-45fb-bb29-ee551ee265b7
# ╠═6e9aaac2-2899-44fd-8d0b-f26ab5a11cf9
# ╠═916ee2fd-f35f-4861-9c0f-fc15c286923e
# ╠═3232cafd-c628-4e1b-baf5-9f6dfdb28cd4
# ╠═54c93eb1-c739-443b-b127-2dcd7765d70f
# ╟─6ad975ac-c986-4470-940f-290c4ee9cb60
# ╠═c95feaed-6086-4dc2-a57c-fdca6076bc3d
# ╠═f070024b-76f9-4160-8f29-e24ad8e764ee
# ╟─510fc56d-b8c3-46c8-a894-0b6600155e09
# ╠═00b3d517-9716-4698-bf84-71bc612f4da3
# ╟─487cd4d7-28f9-4500-a0a9-f14e572d3d81
# ╠═7e74bf71-e487-4089-9829-25b387186e1f
# ╠═32a31142-a173-4b11-91f0-1817ef5df062
# ╠═dab885dd-9ba9-4db4-a988-7f7c4c1e6dfe
# ╠═fd4f92c6-8b2e-492f-8f96-8952673837dc
# ╟─093f955a-5562-402e-bb30-b4ce11742757
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
