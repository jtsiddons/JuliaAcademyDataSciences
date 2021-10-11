### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 187903c0-2042-11ec-0e61-e56251f0b5fc
using LinearAlgebra, SparseArrays, Images, MAT

# ╔═╡ db02be00-66ff-481b-9cf9-bbea4f7a31a5
md"""
# Julia Academy Data Sciences Course

## 02: Linear Algebra

This is my version of the Jupyter notebook for class 2. 
"""

# ╔═╡ 56e1846b-3364-441e-84b8-8e736a7da97a
md"""
### Getting started

It is useful to know that a matrix is stored in memory as a column vector

We start by creating a random matrix using `rand`.
"""

# ╔═╡ 41673b10-6cb1-4082-ae11-dd4305e54a19
# Matrix storage - graphic wasn't working in the markdown 😞
load("./data/matrix_storage.png")

# ╔═╡ bbad0f0b-be64-4988-8b68-a940f5babdae
Arand = rand(10,10) # 10 x 10 random matrix

# ╔═╡ 01179c4f-e765-4781-920f-90e1b7524f69
Atranspose = Arand'

# ╔═╡ 3321cc07-f0c0-4877-b909-a9db5026b14a
A = Arand * Atranspose

# ╔═╡ a3966ae7-e8c7-42c6-803e-7567c970ad1c
md"""
Let's show that $A[1,2] = A[11]$. Matrix rows are rotated and then concatenated for storage.

In a very quick test, I used `@btime` in the REPL to test looping through rows and columns of `A`, adding 1, against looping thought the length of the vector representation of `A`.

`@btime` returns the __minimum__ time elapsed during the benchmark.

The results are

| Representation | `@btime` | Allocations |
| --- | --- | --- | 
| Matrix | 9.063 μs | 321 |
| Vector | 8.424 μs | 301 | 

This suggests (from my understanding of `@btime`) that it may be quicker to use the vector representation - although one may need to perform additional calculations (relating vector and matrix location) to the vector method, which may slow things down.
"""

# ╔═╡ f0d3be17-c63b-420f-a185-fc21cb09d53c
A[11] == A[1,2]

# ╔═╡ d57fdd9f-d13b-47a6-9237-be81cd887216
md"""
We can solve linear systems quickly using `\`. We want `A\b` where `A` is a `Matrix` and `b` is a vector. The `\` operator will choose a suitable factorisation for the matrix `A` in order to solve it quickly.

If `A` is non-square, `\` returns the minimum-norm least squares solution computed by a pivoted `QR` factorization of `A` and a rank estimate of `A` based on the `R` factor

First, verify that the matrix `A` is positive definite.
"""

# ╔═╡ 72dafa77-8775-4c39-8df6-3d2597d0d14a
isposdef(A)

# ╔═╡ b2eb8838-195e-402a-a2d6-fa150696a571
begin
	b = rand(10); 	# Vector of length 10
	x = A\b; 		# Solve A*x = b
	norm(A*x - b)	# Demonstrate that solution is correct
end

# ╔═╡ 1316aeb3-e510-4d2a-b3e9-42d867164856
md"""
A quick look at the data types. The transpose function returns a matrix of `Adjoint` type.
"""

# ╔═╡ f9c96f1e-f010-41d6-b9a3-171793a96624
[
	typeof(A),
	typeof(b),
	typeof(Arand),
	typeof(Atranspose),
	typeof(rand(10))
]

# ╔═╡ dfd03e52-d21f-4085-a590-dd36f02c1e9e
md"""
`adjoint` is a _lazy_ adjoint. We can perform linear algebra operations such as `A*A'` without actually transposing `A`.
"""

# ╔═╡ 99661dc8-f2f3-4403-a3e0-fa5498acc886
Atranspose.parent

# ╔═╡ 11c83626-1254-4149-b2e1-ca5730efdcf6
md"""
We can look at various representations of the `size` of `A`.
"""

# ╔═╡ 89dca098-ce6a-4de2-a315-00eeee336c2e
[
	size(A),	# Standard size, returns a tuple (n: rows,m: columns)
	length(A),	# Returns the length of the vector representation (n * m)
	sizeof(A)	# Number of bytes in A - each element is Float64 which is 8 bytes
]

# ╔═╡ 430f8947-dc5a-40fa-b592-2376985efec8
md"""
We can create a copy of a matrix. Here I copy the transpose, which transforms the transpose from an `Adjoint` to a `Matrix`
"""

# ╔═╡ 9c38ef0f-b76d-49a1-9efe-c47a2adf22ff
B = copy(Atranspose)

# ╔═╡ 5bad65de-8958-4f07-b65b-7f9c53df610a
[
	typeof(Atranspose),
	typeof(B)
]

# ╔═╡ 596f2e34-fc3b-438e-9010-4b0857a3067a
md"""
### Factorisation

One of the most common tools in Linear Algebra is matrix factorisation. These tools are often used to solve systems such as `Ax=b`, a problem that comes up a lot in Data Science.

#### LU Factorisation

We have a lower $\hat{L}$ and upper triangular matrix $\hat{U}$ and a permutation matrix $\hat{P}$ such that 

$\hat{L}\times\hat{U} = \hat{P}\times\hat{A}$
"""

# ╔═╡ a62b7442-49f1-465d-a351-0e61defc7435
luA = lu(A)

# ╔═╡ 79d5f566-f5ec-425c-a683-5a50fbd657cd
md"""
We can access the lower and upper components of this factorisation by getting `luA.L` and `luA.U` respectively. We could also access the elements as elements of a tuple. We would need

```
L, U, p, P = lu(A)
```

where `p` is the permutation vector & `P` is the permutation matrix.
"""

# ╔═╡ e136aac8-e4e5-4fb8-a3d0-344f896b9a46
norm(luA.L*luA.U - luA.P*A)

# ╔═╡ c1d38175-92f4-4951-9f9e-74fc42adc2d8
md"""
#### QR Factorisation

$$\hat{A} = \hat{Q}\times\hat{R}$$

where $\hat{Q}$ is an orthogonal/unitary matrix and $\hat{R}$ is upper triangular.
"""

# ╔═╡ 9961a08a-8a83-4b39-97de-3c936a2c1d20
qrA = qr(A)

# ╔═╡ 607e5dde-373c-4567-8316-f8342bedaa10
norm(A - qrA.Q * qrA.R)

# ╔═╡ 78c904f8-cbfe-48cf-b040-2b5617240bb1
md"""
#### Cholesky Factorisation

$$\hat{L}\times\hat{L}^T = \hat{A}$$

In this case $\hat{L}$ is lower triangular and $\hat{A}$ __must__ be symmetric positive definite. Note that $\hat{L}^T$ is upper triangular.
"""

# ╔═╡ 5139b07e-e589-4f7e-94bd-5f48aece6bb1
[isposdef(A), issymmetric(A)]

# ╔═╡ 1d74f60e-150e-4d10-83d3-4e0dca29de98
cholA = cholesky(A)

# ╔═╡ 006e9d4e-c00d-4ab4-877a-b9653a4bdae7
cholA.L == cholA.U'		# confirm that L = U^T

# ╔═╡ 4be63206-0e67-4c42-bf7f-0e0dbfc30e27
[
	norm(cholA.L * cholA.U - A),
	norm(cholA.L * cholA.L' - A)
]

# ╔═╡ 3f4641ac-8082-46fd-8616-e3438c1d3802
md"""
#### Base Factorisation Function

We can also run `factorize()`. Note that it will choose an appropriate factorisation method. Also note that it uses a `z` not an `s` in the function - cursed American-English 😢
"""

# ╔═╡ 3d55e860-b040-4f0f-99b6-a2225191576a
factorize(A)	# Cholesky as A is symmetric positive definite

# ╔═╡ 7a18c388-9eb7-4d10-b2db-1a7728866b24
factorize(Arand)	# LU for Arand = rand(10,10)

# ╔═╡ f8fe8a3b-d7b7-4e74-9f24-5dd595ee9951
factorize(hcat(Arand,rand(10)))	# QR for non-square matrix

# ╔═╡ 2e02c931-ad43-473b-8442-1b618f86bd9f
md"""
### Diagonal Matrices
"""

# ╔═╡ c9e54626-93e8-41dc-8767-4b8dd289cfc5
A2 = [
	1 2 3;
	4 5 6;
	7 8 9
]

# ╔═╡ fb4b039a-21bf-45cd-a61d-d871aa59706a
md"""
We can get the diagonals of `A2` above using `diag`
"""

# ╔═╡ 66abff1f-7ed1-415e-9ce0-2936243aa8a9
[
	diag(A2),		# Main diagonal
	diag(A2,1),		# Up 1 from main
	diag(A2,2),		# Up 2 from main
	diag(A2,-1),	# Down 1 from main
	diag(A2,-2)		# Down 2 from main
]

# ╔═╡ 0356976f-f5c4-445d-9cca-88a0331662eb
md"""
#### Building diagonal matrices
"""

# ╔═╡ 85f9ff4b-872c-40a9-8275-58eafa6976b9
Diagonal(A2)	# Diagonal Matrix from A2

# ╔═╡ 6aa589e5-e5c9-4fcb-8b87-3173828bcb9b
Diagonal(diag(A2))	# Diagonal Matrix using diagonal of A2 - can just have vector here

# ╔═╡ 2eff119c-e0e5-42d8-a2e5-0733789fe401
diagm(diag(A2)) 	# Full Matrix with only diagonal elements of A2

# ╔═╡ f09f345b-f4e1-4219-9a7a-c1c70b7d5b33
I(3) 	# 3x3 Identity matrix

# ╔═╡ c62623a5-8948-4d15-a8ef-e4e61dec6e37
md"""
### Sparse Linear Algebra

Note that running `Diagonal` had $\cdot$ on the non-diagonal elements. This is a sparse matrix. They are stored in _Compressed Sparse Column (CSC)_ form.
"""

# ╔═╡ ea31a6ae-f075-45d3-9af0-d69adfe03636
S = sprand(5,5,2/5) 	# sparse random matrix with probability 2/5 for non-zero entry

# ╔═╡ 72d1f769-d9d0-4691-8038-ec1e8e5b6d2d
S.rowval 		# There are values in these rows

# ╔═╡ 82ee41c3-3fc8-49c2-a8be-d67134f61355
S.colptr 		# Move to next column on these indices

# ╔═╡ 56e03d5c-0127-4a9b-9c6e-0a6f91db4364
md"""
So in the example above, we have a vector of length 10 (`S.rowval`) which tells us the row positions for the data. We also have `S.colptr` which tells us that starting at index 4 in `S.rowval` we move across one column.
"""

# ╔═╡ 7cb75c7f-8b37-4659-8b9e-5ce922f124f2
Matrix(S)

# ╔═╡ d32fb100-5c77-4886-b88f-aedbf0bdea36
md"""
Let's try to rebuild `S` in a loop. `S` contains all the info we need.

* `S.nzval` are the values
* `S.rowval` are the rows where the above are located
* `S.colptr` tells the program which elements are on the next column
* `S.n` and `S.m` are the number of rows and columns of `S` respectively
"""

# ╔═╡ 373464fa-6382-402c-9116-71324c785b28
[
	S.nzval,
	S.rowval,
	S.colptr,
	(S.n, S.m)
]

# ╔═╡ 953863a0-c58d-4b4a-a49a-d462c7faf0e5
begin
	S2 = zeros(S.n,S.m)
	col = 0
	for (i,row) in enumerate(S.rowval)
		if i == S.colptr[col+1]
			col += 1
		end
		S2[row,col] = S.nzval[i]
	end
	S2
end

# ╔═╡ fdec5767-778f-4118-a8dc-362b2b87f882
S2 == S

# ╔═╡ a94b7f28-cfe1-430b-9413-81b291eb719f
md"""
### Images as matrices

We can load an image using `load`. Note that the type of an image is `Matrix{RGB{N0f8}}`
"""

# ╔═╡ b44ae005-f1fa-46c5-9a5d-5b46542f13ac
X = load("./data/khiam-small.jpg")

# ╔═╡ 89d6cfb1-3e0a-4334-b39c-e1547723d227
typeof(X)

# ╔═╡ 094b7b9f-4c94-45a7-8c2f-19ec7a98d187
X[1,1]

# ╔═╡ ee2553e5-aee4-4b1f-b74d-2f56b1f404c6
md"""
We can convert to grayscale.
"""

# ╔═╡ cf9998f0-3e93-4fe0-8ecf-2f94899e2325
Xgray = Gray.(X)

# ╔═╡ 079a53c6-d3a9-4c94-9d38-12f48409dbe3
typeof(Xgray)

# ╔═╡ 41b4e51e-8ba0-41d7-bfed-1f1f1db400b1
md"""
#### Extracting colours

Each pixel has an r, g, and b element corresponding to red, green, and blue respectively. We cannot automatically get the full matrix of r values for example, however we can use map to loop through pixels and reshape.

Let's extract the colours and then plot only the red.
"""

# ╔═╡ 09144ab9-351e-41b4-b43a-8a23dc6bc316
begin
	lX = length(X);
	nX, mX = size(X);
	
	rmat = map(i->X[i].r,1:lX)
	rmat = Float64.(reshape(rmat,nX,mX))
	
	gmat = map(i->X[i].g,1:lX)
	gmat = Float64.(reshape(gmat,nX,mX))
	
	bmat = map(i->X[i].b,1:lX)
	bmat = Float64.(reshape(bmat,nX,mX))
end;

# ╔═╡ 527f5ecb-35c4-4ff3-82c1-1a2f0c17d61b
begin
	Z = zeros(nX,mX)
	RGB.(rmat,Z,Z)
end

# ╔═╡ 83fe3c84-5a60-4e95-870a-cd370da95eec
md"""
We can more easily get the grayscale values.
"""

# ╔═╡ 81f5e2cb-cbbb-4f0e-8304-cd608883f80d
XgrayValues = float64.(Xgray)

# ╔═╡ be5baad3-8955-49b0-89d9-0c1ccbc76899
md"""
#### Singlular Value Decomposition

This decomposes a matrix into 

$$\hat{U}\times\hat{D}\times{V} = \hat{A}$$

where $\hat{D}$ is a diagonal matrix of the singular values.

Note that we get both `V` and `Vt` as output. We want to use `V` as `Vt` is `transpose(V)`.
"""

# ╔═╡ 5e6255b7-6ea0-4c84-9d94-0b3ba97fef4a
SVD_V = svd(XgrayValues)

# ╔═╡ d1cb0e58-4d00-4ab8-866f-8c35950da47c
fieldnames(typeof(SVD_V))

# ╔═╡ 0115f185-ef7b-489d-b652-c4e6f57a728f
norm(SVD_V.U * diagm(SVD_V.S) * SVD_V.V' - XgrayValues)

# ╔═╡ a3ac6235-b5c4-4e06-9f3c-b0760ad532a7
md"""
Let's try to recontstruct the image using 4 singular values
"""

# ╔═╡ 75108a45-c338-47ec-911c-d3cb3dfb79f5
begin
	u1 = SVD_V.U[:,1]
	v1 = SVD_V.V[:,1]
	s1 = SVD_V.S[1]
	img1 = u1 * s1 * v1'
	
	u2 = SVD_V.U[:,2]
	v2 = SVD_V.V[:,2]
	s2 = SVD_V.S[2]
	img1 += u2 * s2 * v2'
	
	u3 = SVD_V.U[:,3]
	v3 = SVD_V.V[:,3]
	s3 = SVD_V.S[3]
	img1 += u3 * s3 * v3'
	
	u4 = SVD_V.U[:,4]
	v4 = SVD_V.V[:,4]
	s4 = SVD_V.S[4]
	img1 += u4 * s4 * v4'
end

# ╔═╡ 213847b8-b943-43ee-af4f-121428ab6e69
Gray.(img1)

# ╔═╡ a1389562-07e4-4de7-a038-0d706820800a
md"""
This is not a good result, let's write a function to do it for us.

First, it is a better approach to select 1:4 rather than individually add imgs.
"""

# ╔═╡ 811621b4-2101-41e7-99b6-b0a02afd41eb
begin
	u14 = SVD_V.U[:,1:4]
	v14 = SVD_V.V[:,1:4]
	s14 = SVD_V.S[1:4]
	img14 = u14 * spdiagm(s14) * v14' # turn vector s14 into sparse diagonal
	Gray.(img14)
end

# ╔═╡ 7ccad360-6282-4ab7-9446-f0df5a9148ff
function svd_img(SVD_V, i::Int)
	if i == 1
		return SVD_V.U[:,1] * SVD_V.S[1] * SVD_V.V[:,i]'
	end
	
	i = 1:i
	u = SVD_V.U[:,i]
	v = SVD_V.V[:,i]
	s = SVD_V.S[i]
	return u * spdiagm(s) * v'
end

# ╔═╡ 55d2aef2-5aea-44b6-8e95-f1028f8f8ece
Gray.(svd_img(SVD_V,3))

# ╔═╡ 90f315fd-b6d6-4ad6-ae71-e66f3c4d4149
[
	norm(XgrayValues - svd_img(SVD_V,1)),
	norm(XgrayValues - svd_img(SVD_V,4)),
	norm(XgrayValues - svd_img(SVD_V,10)),
	norm(XgrayValues - svd_img(SVD_V,100))
]

# ╔═╡ 82261d77-26f9-44e4-81b5-210e2507a1e4
md"""
### Face Recognition

Now for something more challenging
"""

# ╔═╡ 6fe48929-c549-47bf-94fe-a63b610b8f2e
M = matread("data/face_recog_qr.mat")

# ╔═╡ afcd5db4-0a1b-45ff-a5f5-103452faed81
md"""
Each vector in `M["V2"]` is a face image. Let's extract the first, then look at the remaining faces.

We find a number of different faces, each repeated with different levels of contrast.
"""

# ╔═╡ 37af25b8-cc6f-452a-930a-bdb34219dcce
begin
	q = reshape(M["V2"][:,1],192,168)
	Gray.(q)
end

# ╔═╡ 14049d51-5a59-4413-9f2f-afcadfa6a652
begin
	qimgs = Any[]
	qs = []
	qnorms = []
	for i = 1:size(M["V2"])[2]
		qcur = reshape(M["V2"][:,i],192,168)
		push!(qs, qcur)
		push!(qnorms, norm(q - qcur))
		push!(qimgs,Gray.(qcur))
	end
	qimgs
end

# ╔═╡ 3d1ba29b-00a1-4d9e-bb43-1a2ef08e8130
qnorms

# ╔═╡ 4441ea6b-2d69-4cd3-8816-4274ad2ec8b6
md"""
Let's now do a simple test. We try to selec the imagest that are most similar to `q` (the first image). Let's define the query image
"""

# ╔═╡ 9c15515e-93f6-4b81-bfb1-56e9fa426052
bqueryface = q[:]

# ╔═╡ bf7a481c-fef0-4e6a-bbe7-f8bd7d19e413
begin
	Afaces = M["V2"][:,2:end]
	xface = Afaces\bqueryface
	Gray.(reshape(Afaces*xface,192,168))
end

# ╔═╡ 1065642b-b6c9-4fd2-bb16-074142cdb5ee
md"""
We get a very close face as output. Let's look at the norm, and `xface`
"""

# ╔═╡ d7389457-a672-4b78-979c-6723964d6122
norm(Afaces*xface - bqueryface)

# ╔═╡ cec1bd41-6f87-44c2-a21f-da30cff819b0
xface

# ╔═╡ a71ab42c-9786-4986-b8d5-1673b6fedbf9
findall(xface .> 0.5)

# ╔═╡ 144fd882-fdcc-4f8a-85c9-f496e59c68aa
md"""
We notice that we have a value of 1 for face 23 (24 in the full set)

We can add some noise to the query image now and see what happens
"""

# ╔═╡ 8729b65c-059d-46c2-96a1-b5c67b45c3fe
begin
	qnoise = q + rand(192,168)*0.5
	qnoise = qnoise ./ maximum(qnoise)
	bnoise = qnoise[:]
	Gray.(qnoise)
end

# ╔═╡ 4ebc88b2-a54b-468a-ae3d-66af7479072e
xnoise = Afaces \ bnoise

# ╔═╡ c477e769-40ca-4686-a27c-b6868d9dd3df
md"""
We get larger contributions to the query face from the other faces
"""

# ╔═╡ c5cd8ba6-99c0-4e58-b91c-3012e12f56f0
Gray.(reshape(Afaces*xnoise,192,168))

# ╔═╡ 108916d6-29f2-4cfa-b5a6-6e1f2dbb160a
norm(Afaces * xnoise - bnoise)

# ╔═╡ 94ba2145-ebf8-4ace-9a94-b30dde983195
findall(xnoise .> 0.5)

# ╔═╡ c3e3f3b3-b4e0-49aa-ae36-13eba418db93
xnoise[findall(xnoise .> 0.5)]

# ╔═╡ 0dcaa469-b057-452b-a39f-7246d4ed9f80
md"""
Let's look at the significant faces.

Note that we need to add 1 hto the index of faces as we excluded face 1 as our query face.
"""

# ╔═╡ 0c62c419-5836-4862-a5d4-b822276c3d49
begin
	resfaces = Any[]
	for (i,face) in enumerate(findall(xnoise .> 0.5).+1)
		curface = reshape(M["V2"][:,face],192,168)
		push!(resfaces, Gray.(curface))
	end
	resfaces
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MAT = "23992714-dd62-5051-b70f-ba57cb901cac"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
Images = "~0.24.1"
MAT = "~0.10.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "b8d49c34c3da35f220e7295659cd0bab8e739fed"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.33"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "d127d5e4d86c7680b20c35d40b503c74b9a39b5e"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.4"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

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

[[BufferedStreams]]
deps = ["Compat", "Test"]
git-tree-sha1 = "5d55b9486590fdda5905c275bb21ce1f0754020f"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.0.0"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e8a30e8019a512e4b6c56ccebc065026624660e8"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.7.0"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

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

[[ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "6d1c23e740a586955645500bbec662476204a52c"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.1"

[[CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

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

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "8041575f021cba5a099a456b4163c9a08b566a02"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.0"

[[FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "70a0cfd9b1c86b0209e38fbfe6d8231fd606eeaf"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.1"

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

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "2c1cf4df419938ece72de17f368a021ee162762e"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.0"

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

[[IdentityRanges]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be8fcd695c4da16a1d6d0cd213cb88090a150e3b"
uuid = "bbac6d45-d8f3-5730-bfe4-7a449cd117ca"
version = "0.3.1"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[ImageAxes]]
deps = ["AxisArrays", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "794ad1d922c432082bc1aaa9fa8ffbd1fe74e621"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.9"

[[ImageContrastAdjustment]]
deps = ["ColorVectorSpace", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "2e6084db6cccab11fe0bc3e4130bd3d117092ed9"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.7"

[[ImageCore]]
deps = ["AbstractFFTs", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "db645f20b59f060d8cfae696bc9538d13fd86416"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.8.22"

[[ImageDistances]]
deps = ["ColorVectorSpace", "Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "6378c34a3c3a216235210d19b9f495ecfff2f85f"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.13"

[[ImageFiltering]]
deps = ["CatIndices", "ColorVectorSpace", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageCore", "LinearAlgebra", "OffsetArrays", "Requires", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "bf96839133212d3eff4a1c3a80c57abc7cfbf0ce"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.6.21"

[[ImageIO]]
deps = ["FileIO", "Netpbm", "OpenEXR", "PNGFiles", "TiffImages", "UUIDs"]
git-tree-sha1 = "13c826abd23931d909e4c5538643d9691f62a617"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.5.8"

[[ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[ImageMagick_jll]]
deps = ["JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1c0a2295cca535fabaf2029062912591e9b61987"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.10-12+3"

[[ImageMetadata]]
deps = ["AxisArrays", "ColorVectorSpace", "ImageAxes", "ImageCore", "IndirectArrays"]
git-tree-sha1 = "ae76038347dc4edcdb06b541595268fca65b6a42"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.5"

[[ImageMorphology]]
deps = ["ColorVectorSpace", "ImageCore", "LinearAlgebra", "TiledIteration"]
git-tree-sha1 = "68e7cbcd7dfaa3c2f74b0a8ab3066f5de8f2b71d"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.2.11"

[[ImageQualityIndexes]]
deps = ["ColorVectorSpace", "ImageCore", "ImageDistances", "ImageFiltering", "OffsetArrays", "Statistics"]
git-tree-sha1 = "1198f85fa2481a3bb94bf937495ba1916f12b533"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.2.2"

[[ImageShow]]
deps = ["Base64", "FileIO", "ImageCore", "OffsetArrays", "Requires", "StackViews"]
git-tree-sha1 = "832abfd709fa436a562db47fd8e81377f72b01f9"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.1"

[[ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "IdentityRanges", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "e4cc551e4295a5c96545bb3083058c24b78d4cf0"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.8.13"

[[Images]]
deps = ["AxisArrays", "Base64", "ColorVectorSpace", "FileIO", "Graphics", "ImageAxes", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageShow", "ImageTransformations", "IndirectArrays", "OffsetArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "8b714d5e11c91a0d945717430ec20f9251af4bd2"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.24.1"

[[Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[IndirectArrays]]
git-tree-sha1 = "c2a145a145dc03a7620af1444e0264ef907bd44f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "0.5.1"

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

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

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

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

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

[[MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "5c62992f3d46b8dce69bdd234279bb5a369db7d5"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.1"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

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

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[Netpbm]]
deps = ["ColorVectorSpace", "FileIO", "ImageCore"]
git-tree-sha1 = "09589171688f0039f13ebe0fdcc7288f50228b52"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.1"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0e9e582987d36d5a61e650e6e543b9e44d9914b"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.7"

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

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "e14c485f6beee0c7a8dcf6128bf70b85f1fe201e"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.9"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "646eed6f6a5d8df6708f15ea7e02a7a2c4fe4800"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.10"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rotations]]
deps = ["LinearAlgebra", "StaticArrays", "Statistics"]
git-tree-sha1 = "2ed8d8a16d703f900168822d83699b8c3c1a5cd8"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.0.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

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
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

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
git-tree-sha1 = "632a8d4dbbad6627a4d2d21b1c6ebcaeebb1e1ed"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.4.2"

[[TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "59e2ad8fd1591ea019a5259bd012d7aee15f995c"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═db02be00-66ff-481b-9cf9-bbea4f7a31a5
# ╠═187903c0-2042-11ec-0e61-e56251f0b5fc
# ╠═56e1846b-3364-441e-84b8-8e736a7da97a
# ╠═41673b10-6cb1-4082-ae11-dd4305e54a19
# ╠═bbad0f0b-be64-4988-8b68-a940f5babdae
# ╠═01179c4f-e765-4781-920f-90e1b7524f69
# ╠═3321cc07-f0c0-4877-b909-a9db5026b14a
# ╠═a3966ae7-e8c7-42c6-803e-7567c970ad1c
# ╠═f0d3be17-c63b-420f-a185-fc21cb09d53c
# ╠═d57fdd9f-d13b-47a6-9237-be81cd887216
# ╠═72dafa77-8775-4c39-8df6-3d2597d0d14a
# ╠═b2eb8838-195e-402a-a2d6-fa150696a571
# ╠═1316aeb3-e510-4d2a-b3e9-42d867164856
# ╠═f9c96f1e-f010-41d6-b9a3-171793a96624
# ╠═dfd03e52-d21f-4085-a590-dd36f02c1e9e
# ╠═99661dc8-f2f3-4403-a3e0-fa5498acc886
# ╠═11c83626-1254-4149-b2e1-ca5730efdcf6
# ╠═89dca098-ce6a-4de2-a315-00eeee336c2e
# ╠═430f8947-dc5a-40fa-b592-2376985efec8
# ╠═9c38ef0f-b76d-49a1-9efe-c47a2adf22ff
# ╠═5bad65de-8958-4f07-b65b-7f9c53df610a
# ╠═596f2e34-fc3b-438e-9010-4b0857a3067a
# ╠═a62b7442-49f1-465d-a351-0e61defc7435
# ╠═79d5f566-f5ec-425c-a683-5a50fbd657cd
# ╠═e136aac8-e4e5-4fb8-a3d0-344f896b9a46
# ╠═c1d38175-92f4-4951-9f9e-74fc42adc2d8
# ╠═9961a08a-8a83-4b39-97de-3c936a2c1d20
# ╠═607e5dde-373c-4567-8316-f8342bedaa10
# ╠═78c904f8-cbfe-48cf-b040-2b5617240bb1
# ╠═5139b07e-e589-4f7e-94bd-5f48aece6bb1
# ╠═1d74f60e-150e-4d10-83d3-4e0dca29de98
# ╠═006e9d4e-c00d-4ab4-877a-b9653a4bdae7
# ╠═4be63206-0e67-4c42-bf7f-0e0dbfc30e27
# ╠═3f4641ac-8082-46fd-8616-e3438c1d3802
# ╠═3d55e860-b040-4f0f-99b6-a2225191576a
# ╠═7a18c388-9eb7-4d10-b2db-1a7728866b24
# ╠═f8fe8a3b-d7b7-4e74-9f24-5dd595ee9951
# ╠═2e02c931-ad43-473b-8442-1b618f86bd9f
# ╠═c9e54626-93e8-41dc-8767-4b8dd289cfc5
# ╠═fb4b039a-21bf-45cd-a61d-d871aa59706a
# ╠═66abff1f-7ed1-415e-9ce0-2936243aa8a9
# ╠═0356976f-f5c4-445d-9cca-88a0331662eb
# ╠═85f9ff4b-872c-40a9-8275-58eafa6976b9
# ╠═6aa589e5-e5c9-4fcb-8b87-3173828bcb9b
# ╠═2eff119c-e0e5-42d8-a2e5-0733789fe401
# ╠═f09f345b-f4e1-4219-9a7a-c1c70b7d5b33
# ╠═c62623a5-8948-4d15-a8ef-e4e61dec6e37
# ╠═ea31a6ae-f075-45d3-9af0-d69adfe03636
# ╠═72d1f769-d9d0-4691-8038-ec1e8e5b6d2d
# ╠═82ee41c3-3fc8-49c2-a8be-d67134f61355
# ╠═56e03d5c-0127-4a9b-9c6e-0a6f91db4364
# ╠═7cb75c7f-8b37-4659-8b9e-5ce922f124f2
# ╠═d32fb100-5c77-4886-b88f-aedbf0bdea36
# ╠═373464fa-6382-402c-9116-71324c785b28
# ╠═953863a0-c58d-4b4a-a49a-d462c7faf0e5
# ╠═fdec5767-778f-4118-a8dc-362b2b87f882
# ╠═a94b7f28-cfe1-430b-9413-81b291eb719f
# ╠═b44ae005-f1fa-46c5-9a5d-5b46542f13ac
# ╠═89d6cfb1-3e0a-4334-b39c-e1547723d227
# ╠═094b7b9f-4c94-45a7-8c2f-19ec7a98d187
# ╠═ee2553e5-aee4-4b1f-b74d-2f56b1f404c6
# ╠═cf9998f0-3e93-4fe0-8ecf-2f94899e2325
# ╠═079a53c6-d3a9-4c94-9d38-12f48409dbe3
# ╠═41b4e51e-8ba0-41d7-bfed-1f1f1db400b1
# ╠═09144ab9-351e-41b4-b43a-8a23dc6bc316
# ╠═527f5ecb-35c4-4ff3-82c1-1a2f0c17d61b
# ╠═83fe3c84-5a60-4e95-870a-cd370da95eec
# ╠═81f5e2cb-cbbb-4f0e-8304-cd608883f80d
# ╠═be5baad3-8955-49b0-89d9-0c1ccbc76899
# ╠═5e6255b7-6ea0-4c84-9d94-0b3ba97fef4a
# ╠═d1cb0e58-4d00-4ab8-866f-8c35950da47c
# ╠═0115f185-ef7b-489d-b652-c4e6f57a728f
# ╠═a3ac6235-b5c4-4e06-9f3c-b0760ad532a7
# ╠═75108a45-c338-47ec-911c-d3cb3dfb79f5
# ╠═213847b8-b943-43ee-af4f-121428ab6e69
# ╠═a1389562-07e4-4de7-a038-0d706820800a
# ╠═811621b4-2101-41e7-99b6-b0a02afd41eb
# ╠═7ccad360-6282-4ab7-9446-f0df5a9148ff
# ╠═55d2aef2-5aea-44b6-8e95-f1028f8f8ece
# ╠═90f315fd-b6d6-4ad6-ae71-e66f3c4d4149
# ╠═82261d77-26f9-44e4-81b5-210e2507a1e4
# ╠═6fe48929-c549-47bf-94fe-a63b610b8f2e
# ╠═afcd5db4-0a1b-45ff-a5f5-103452faed81
# ╠═37af25b8-cc6f-452a-930a-bdb34219dcce
# ╠═14049d51-5a59-4413-9f2f-afcadfa6a652
# ╠═3d1ba29b-00a1-4d9e-bb43-1a2ef08e8130
# ╠═4441ea6b-2d69-4cd3-8816-4274ad2ec8b6
# ╠═9c15515e-93f6-4b81-bfb1-56e9fa426052
# ╠═bf7a481c-fef0-4e6a-bbe7-f8bd7d19e413
# ╠═1065642b-b6c9-4fd2-bb16-074142cdb5ee
# ╠═d7389457-a672-4b78-979c-6723964d6122
# ╠═cec1bd41-6f87-44c2-a21f-da30cff819b0
# ╠═a71ab42c-9786-4986-b8d5-1673b6fedbf9
# ╠═144fd882-fdcc-4f8a-85c9-f496e59c68aa
# ╠═8729b65c-059d-46c2-96a1-b5c67b45c3fe
# ╠═4ebc88b2-a54b-468a-ae3d-66af7479072e
# ╠═c477e769-40ca-4686-a27c-b6868d9dd3df
# ╠═c5cd8ba6-99c0-4e58-b91c-3012e12f56f0
# ╠═108916d6-29f2-4cfa-b5a6-6e1f2dbb160a
# ╠═94ba2145-ebf8-4ace-9a94-b30dde983195
# ╠═c3e3f3b3-b4e0-49aa-ae36-13eba418db93
# ╠═0dcaa469-b057-452b-a39f-7246d4ed9f80
# ╠═0c62c419-5836-4862-a5d4-b822276c3d49
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
