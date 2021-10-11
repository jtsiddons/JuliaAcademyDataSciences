### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ b2c4212a-2836-11ec-390a-1d38cf9a1e17
using Pkg; Pkg.activate(".")

# ╔═╡ 98fe6ab4-40b0-4205-a585-c58e0a8e9c94
using Flux, Images, MLDatasets

# ╔═╡ b33c4a37-7c99-4b24-b1bc-012bcf68d5ea
using Flux: onehotbatch, argmax, crossentropy, throttle

# ╔═╡ 6cfa0c51-7bcf-427c-b450-a3a8caf7e0a7
using Base.Iterators: repeated

# ╔═╡ 219cf430-9b15-4739-a4be-d4a2ef763813
using StatsBase: sample	

# ╔═╡ 999d86a5-7ce7-43ec-90ef-019115c08910
md"""
# Julia Academy Data Sciences

## 10. Neural Networks

This is my write-up of notebook 10 from the Julia Academy Data Sciences course into Pluto.jl. 

In this note book we go through one of the famous neural network examples, classifying  the MNIST handwritten numbers dataset.

We will be using `Flux` which requires the ability to use `CUDA`. Arch Linux users may need to install `julia-bin` from the AUR for Julia to recognise the LLVM library correctly.

Let's start by loading the required libraries. Again I am using my local environment by calling

```julia
using Pkg; Pkg.activate(".")
```

in the first cell. This disables the Pluto notebook's own package management system. It means that sharing this notebook is slightly more complicated, but loading the packages can be faster as I have previously pre-compiled them.
"""

# ╔═╡ 82216b2f-435a-4c50-8b28-33ecfced9a68
md"""
### MNIST Data

The course loads the data using `Flux.Data.MNIST` however this method is depreciated so we load using `MLDatasets` which is loaded above. 

I cannot run `MNIST.download` here, I have run it seperately in the REPL and stored the files in `./data/MNIST`.

Let's look at one of the images.
"""

# ╔═╡ 49168ebc-3f80-4a9e-863a-2748a2a792de
if !isfile("./data/MNIST/train-labels-idx1-ubyte.gz")
	MNIST.download(dir="./data/MNIST/"; i_accept_the_terms_of_use=true)
end

# ╔═╡ 49168ebc-3f80-4a9e-863a-a448a2a792de
begin
	train_labels = Float64.(MNIST.Reader.readlabels("./data/MNIST/train-labels-idx1-ubyte.gz"));
	test_labels = Float64.(MNIST.Reader.readlabels("./data/MNIST/t10k-labels-idx1-ubyte.gz"));
	ntrain = length(train_labels);
	ntest = length(test_labels)
end

# ╔═╡ 4db480e2-61f5-4210-910f-ec9ea7adba25
md"""
Working with this data and Pluto.jl is a right pain. The download returns a 3d array and I want to have a vector of images. The training data contains $ntrain images, extracting this in a loop is a nightmare and seems to lock up Pluto - despitte it being rapid in the REPL. I need to write a function to extract the images.

I also convert them to `Float32` here as that makes `Flux` faster.
"""

# ╔═╡ 45d7721f-5824-4877-a8aa-b1608ee8dfb1
function get_imgs(file::String, n::Int)
	imgs = MNIST.convert2image(MLDatasets.MNIST.Reader.readimages(file))[:,:,:];
	imgsplit = []
	for i = 1:n
		push!(imgsplit, Float32.(imgs[:,:,i][:]))
	end
	return imgsplit
end

# ╔═╡ 2e01ce9b-7ad8-43d1-a7ac-ba8a00d21132
train_imgs = get_imgs("./data/MNIST/train-images-idx3-ubyte.gz",ntrain);

# ╔═╡ 748022ec-b086-4f94-a44b-2eff9e6a029b
test_imgs = get_imgs("./data/MNIST/t10k-images-idx3-ubyte.gz",ntest);

# ╔═╡ b2439a36-1153-4e40-86dd-4d2d76ebab0c
md"""
A quick function to show an image from the vectorised format. I know that the images are 28×28 so I can easily reshape.
"""

# ╔═╡ 091e6ce3-f611-4eb9-8604-a9c133896b22
showimg(x::Vector{Float32}) = colorview(Gray, reshape(x, 28, 28)) 

# ╔═╡ 24196d33-e5f0-4ddb-add6-2b171622c00f
showimg(train_imgs[55])

# ╔═╡ f6d116e4-81d5-41ed-a66e-370d9c1b48c5
size(train_imgs)

# ╔═╡ 1e155dcf-b0eb-4c01-86d0-2604bfdec934
md"""
### Building the network

From the labels we create an output column for each image. Each column is a vector of length 10 with a `1` in the location for the solution (between 0:9 - so 1:10 in Julia index).

For example `[0 0 0 1 0 0 0 0 0 0]` corresponds to 3 - index 1 corresponds to the result 0.
"""

# ╔═╡ e41e9edb-5782-4582-b63c-5c1b9eb3d37b
Y = onehotbatch(train_labels, 0:9);

# ╔═╡ 50bff3f6-50ed-424c-9dd1-8ee7684e4307
md"""
Now we build the model. We will have 2 layers. The hidden layer will have 32 nodes and the output layer will have 10 (the number of possible solutions 0:9) so we will go:

`(28×28 =) 784 → 32 → 10`
"""

# ╔═╡ 7015cdda-6769-47d4-a726-1438631a4223
m = Chain(
	Dense(28^2, 32, relu),
	Dense(32, 10),
	softmax)

# ╔═╡ f32fec38-c0f4-4284-9ab7-3581322fdd50
md"""
What does `m`, the neural network mean here? 

If you've worked with neural networks before you know that the solution is often not found by just one pass on the neural network. One pass happens, and a solution is generated at the output layer, then this solution is compared to the ground truth solution we already have (the columns from `Y`), and the network goes back and adjusts its weights and parameters and then try again. Here, since `m` is not "trained" yet, one pass of `m` on a figure generates the following (not-so-great) answer. We will see later how this changes after training.
"""

# ╔═╡ 1b6c9f08-786e-4ddf-894f-0039b0d8d2ba
m(train_imgs[14])

# ╔═╡ 51d802a3-0bae-4a2c-9880-cbc16401bb65
md"""
To run our neural network, we need a loss function and an accuracy function. The accuracy function is used to compare the output result from the output layer in the neural network to the groundtruth result. The loss function is used to evaluate the performance of the overall model after new weights have been recalculated at each pass.
"""

# ╔═╡ 5257fc34-3ea1-48fd-bfc9-55c24c507c17
begin
	loss(x, y) = Flux.crossentropy(m(x), y)
	accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))
end

# ╔═╡ 99863963-2bd8-4ff9-a2c2-9dc1ee271da6
md"""
I may want the images to be in a matrix where each column is the column vector corresponding to an image. Let's create those matrices.
"""

# ╔═╡ 3cafb147-a545-40b7-84d2-4e2cf0fee322
begin
	train_imgs_mat = zeros(28^2, ntrain)
	for i = 1:ntrain
		train_imgs_mat[:, i] = train_imgs[i];
	end
end;

# ╔═╡ da81dac0-ad9b-4fb0-9acb-f3c67ec20f49
begin
	test_imgs_mat = zeros(28^2, ntest)
	for i = 1:ntest
		test_imgs_mat[:, i] = test_imgs[i];
	end
end;

# ╔═╡ 6a52532b-90ca-4775-90a9-332e04c79144
md"""
We will repeat our data to have more samples to pass the neural network - more change for corrections.
"""

# ╔═╡ ac5d0ee6-7535-4bfc-bb8d-9fd5a7bc2388
dataset = collect(repeated((train_imgs_mat, Y),200));

# ╔═╡ a16b70a4-a5a2-41fe-9992-ccd230d6a086
md"""
A function to display loss at each step
"""

# ╔═╡ 83de67e5-7d0e-4664-8336-5ba56906b4ac
evalcb = () -> @show(loss(train_imgs_mat, Y))

# ╔═╡ 7c02c18b-bfd8-4791-b0c0-0beae57deae6
ps = Flux.params(m)

# ╔═╡ 1b6dd992-1796-464b-88fc-74f7025eee10
md"""
### Train the model!
"""

# ╔═╡ b0bb7c66-cc7c-4c86-93c2-b71eefbe2115
opt = ADAM()

# ╔═╡ aeec9786-1cf2-4dc3-8bbe-6165f136c12a
Flux.train!(loss, ps, dataset, opt, cb=throttle(evalcb,10))

# ╔═╡ 91bb550d-88ff-4112-9d60-940fee0cde4e
[
	showimg(test_imgs[14]),
	m(test_imgs[14]),
	argmax(m(test_imgs[14])) - 1
]

# ╔═╡ 26e1e34e-8545-4459-99d7-12db6673af3f
md"""
So, our model is trained and we can see from the above result that it performs rather well. Let's randomly select 100 test images and see if it matches.
"""

# ╔═╡ d5f38f14-f3a4-45e3-84e5-9dcc28dd15bc
begin
	n = 1000
	ids = n==ntest ? collect([1:n]) : sample(1:ntest, n, replace=false)
	result = []
	failed = []
	for id in ids
		testres = argmax(m(test_imgs[id]))-1
		successres = testres == test_labels[id]
		if !successres
			if n > 1000
				push!(failed, (id, test_labels[id], testres)) #, showimg(test_imgs[id])))
			else
				push!(failed, (id, test_labels[id], testres, showimg(test_imgs[id]), m(test_imgs[id])))
			end
		end
		push!(result, (test_labels[id], argmax(m(test_imgs[id]))-1, test_labels[id] == argmax(m(test_imgs[id]))-1))
	end
	(n-length(failed))*100/n
end

# ╔═╡ ff5e250f-c5bf-4183-8b2f-62f593f34971
failed

# ╔═╡ 314d3870-818e-4198-841e-8f66fc3964f7
md"""
It appears we are succesful $((n-length(failed))*100/n)% of the time.
"""

# ╔═╡ Cell order:
# ╠═b2c4212a-2836-11ec-390a-1d38cf9a1e17
# ╟─999d86a5-7ce7-43ec-90ef-019115c08910
# ╠═98fe6ab4-40b0-4205-a585-c58e0a8e9c94
# ╠═b33c4a37-7c99-4b24-b1bc-012bcf68d5ea
# ╠═6cfa0c51-7bcf-427c-b450-a3a8caf7e0a7
# ╟─82216b2f-435a-4c50-8b28-33ecfced9a68
# ╠═49168ebc-3f80-4a9e-863a-2748a2a792de
# ╠═49168ebc-3f80-4a9e-863a-a448a2a792de
# ╟─4db480e2-61f5-4210-910f-ec9ea7adba25
# ╠═45d7721f-5824-4877-a8aa-b1608ee8dfb1
# ╠═2e01ce9b-7ad8-43d1-a7ac-ba8a00d21132
# ╠═748022ec-b086-4f94-a44b-2eff9e6a029b
# ╟─b2439a36-1153-4e40-86dd-4d2d76ebab0c
# ╠═091e6ce3-f611-4eb9-8604-a9c133896b22
# ╠═24196d33-e5f0-4ddb-add6-2b171622c00f
# ╠═f6d116e4-81d5-41ed-a66e-370d9c1b48c5
# ╟─1e155dcf-b0eb-4c01-86d0-2604bfdec934
# ╠═e41e9edb-5782-4582-b63c-5c1b9eb3d37b
# ╟─50bff3f6-50ed-424c-9dd1-8ee7684e4307
# ╠═7015cdda-6769-47d4-a726-1438631a4223
# ╟─f32fec38-c0f4-4284-9ab7-3581322fdd50
# ╠═1b6c9f08-786e-4ddf-894f-0039b0d8d2ba
# ╟─51d802a3-0bae-4a2c-9880-cbc16401bb65
# ╠═5257fc34-3ea1-48fd-bfc9-55c24c507c17
# ╟─99863963-2bd8-4ff9-a2c2-9dc1ee271da6
# ╠═3cafb147-a545-40b7-84d2-4e2cf0fee322
# ╠═da81dac0-ad9b-4fb0-9acb-f3c67ec20f49
# ╟─6a52532b-90ca-4775-90a9-332e04c79144
# ╠═ac5d0ee6-7535-4bfc-bb8d-9fd5a7bc2388
# ╟─a16b70a4-a5a2-41fe-9992-ccd230d6a086
# ╠═83de67e5-7d0e-4664-8336-5ba56906b4ac
# ╠═7c02c18b-bfd8-4791-b0c0-0beae57deae6
# ╟─1b6dd992-1796-464b-88fc-74f7025eee10
# ╠═b0bb7c66-cc7c-4c86-93c2-b71eefbe2115
# ╠═aeec9786-1cf2-4dc3-8bbe-6165f136c12a
# ╠═91bb550d-88ff-4112-9d60-940fee0cde4e
# ╟─26e1e34e-8545-4459-99d7-12db6673af3f
# ╠═219cf430-9b15-4739-a4be-d4a2ef763813
# ╠═d5f38f14-f3a4-45e3-84e5-9dcc28dd15bc
# ╠═ff5e250f-c5bf-4183-8b2f-62f593f34971
# ╟─314d3870-818e-4198-841e-8f66fc3964f7
