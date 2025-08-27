# Haar, cHaar Calculations
Exact calculations of Haar and cHaar twirl quantities using sympy and haarpy.

## matrix
- Compute <t>-order <k>-concatenated, arbitrary $d_{\mathcal{H}},d_{\mathcal{E}}$-dimensional cHaar twirl permutation and cycle operator coefficients $\tau_{d_{\mathcal{H}}d_{\mathcal{E}}}^{(t)}(\sigma,\pi) \to \tau_{d_{\mathcal{H}}d_{\mathcal{E}}}^{(t)}(P,S)$ exactly using sympy and haarpy.
```python
	./main.py <path> <t>
	./plot.py <path> <t> <k> <n>
```

## basis
- Compute <t>-order $<d>$-dimensional localized permutation basis change of basis matrix using sympy.
```python
	./main.py <path> <t> <d>
	./plot.py <path> <t> <d>
```

## norm
- Compute <t>-order, various $d_{\mathcal{H}},d_{\mathcal{E}}$-dimensional cHaar twirl norm relative to Haar, Depolarize twirl norms exactly using sympy and numpy linear algebra.
```python
	./main.py settings.json
```

## test
- *fisher*: tests for mixed-state fisher information as a function of noise locality.
- *haar*: tests for haar twirl coefficients.
- *matmul*: tests for symbolic matrix multiplications
- *qiskit*: initial random kraus operator simulations for scaling of expressivity with number of samples to compute twirls using qiskit and numpy.
- *scaling*: tests for leading order <t>-order chaar twirl coefficients scaling for k-concatenations using sympy and numpy linear algebra.
- *weingarten*: tests for exact weingarten functions using external library and representation theory.
- *test*: tests for Haar,cHaar twirl norm using numpy.