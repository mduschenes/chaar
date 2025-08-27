# Haar, cHaar Calculations
Exact calculations of Haar and cHaar twirl quantities using `sympy`, `haarpy`, and `numpy`.

## matrix
- Compute `t`-order `k`-concatenated, arbitrary $d,d_{\mathcal{E}}=d^{n}$-dimensional cHaar twirl permutation and localized permutation transfer matrix coefficients $\tau_{dd_{\mathcal{E}}}^{(t)}(\sigma,\pi) \to \tau_{dd_{\mathcal{E}}}^{(t)}([\sigma],[\pi])$ using `sympy` and `haarpy`.
```python
	./main.py <path> <t>
	./plot.py <path> <t> <k> <n>
```

## basis
- Compute `t`-order `d`-dimensional localized permutation basis change of basis matrix $\phi_{d}^{(t)}(\sigma,\pi)$ using `sympy` and `numpy`.
```python
	./main.py <path> <t> <d>
	./plot.py <path> <t> <d>
```

## norm
- Compute `t`-order, various $d,d_{\mathcal{E}}$-dimensional cHaar twirl norm $|\tau_{dd_{\mathcal{E}}}^{(t)}|$ relative to Haar, Depolarize twirl norms exactly using `sympy` and `numpy`.
```python
	./main.py settings.json
```

<!-- ## test
- *fisher*: tests for mixed-state fisher information as a function of noise locality.
- *haar*: tests for haar twirl coefficients.
- *matmul*: tests for symbolic matrix multiplications
- *qiskit*: initial random kraus operator simulations for scaling of expressivity with number of samples to compute twirls using qiskit and numpy.
- *scaling*: tests for leading order t-order chaar twirl coefficients scaling for k-concatenations.
- *weingarten*: tests for exact weingarten functions using external library and representation theory.
- *test*: tests for Haar,cHaar twirl. -->