# Haar, cHaar Calculations
Exact calculations of Haar and cHaar ensemble quantities using `sympy`, `haarpy`, and `numpy`.

This library is used in the preparation of the work:
- *Moments of quantum channel ensembles*, Duschenes, M., Garcia-Martin, D., Holmes, Z., Cerezo, M., arXiv preprint arXiv:2511XXXXX (2025), found on [arXiv](https://arxiv.org/abs/2511.XXXXX).

## matrix
- Compute `t`-order `k`-concatenated $d,d_{\mathcal{E}}=d^{n}$-dimensional cHaar ensemble localized permutation transfer matrix coefficients $\tau_{dd_{\mathcal{E}}}^{(t)}([\sigma],[\pi])$, norm $\lVert\tau_{dd_{\mathcal{E}}}^{(t)}\rVert$, and trace $\text{Tr}[\tau_{dd_{\mathcal{E}}}^{(t)}]$
```python
	./main.py <path> <t>
	./plot.py <path> <t> <k> <n>
```

## basis
- Compute `t`-order `d`-dimensional localized permutation basis change of basis matrix $\phi_{d}^{(t)}(\sigma,\pi)$
```python
	./main.py <path> <t> <d>
	./plot.py <path> <t> <d>
```

<!-- ## test
- *fisher*: tests for mixed-state fisher information as a function of noise locality.
- *haar*: tests for haar twirl coefficients.
- *matmul*: tests for symbolic matrix multiplications
- *qiskit*: initial random kraus operator simulations for scaling of expressivity with number of samples to compute twirls using qiskit and numpy.
- *scaling*: tests for leading order t-order chaar twirl coefficients scaling for k-concatenations.
- *weingarten*: tests for exact weingarten functions using external library and representation theory.
- *test*: tests for Haar,cHaar twirl. -->