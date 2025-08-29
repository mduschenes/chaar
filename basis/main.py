#!/usr/bin/env python

import sys,os

import sympy as sp
from sympy.combinatorics.named_groups import SymmetricGroup,Permutation

import numpy as np
from math import prod
from itertools import product
import pickle
import logging

logger = logging.getLogger(__name__)	
logging.basicConfig(level=logging.INFO,format='%(message)s',stream=sys.stdout)
log = lambda *message,verbose=True,**kwargs: logger.info('\t'.join(str(i) for i in message) if len(message)>1 else message[0] if len(message)>0 else "") if verbose else None

def dump(path,data):
	if path is None:
		return

	mkdir(path)

	with open(path, 'wb') as file:
		file.write(pickle.dumps(data))
	return

def load(path):
	data = None
	if (path is None) or (not os.path.exists(path)):
		return data
	with open(path, 'rb') as file:
		data = pickle.loads(file.read())
	return data

def exists(path):
	try:
		return os.path.exists(path)
	except:
		return False

def mkdir(path):
	directory = os.path.dirname(path) if path else None
	if directory and not os.path.exists(directory):
		os.makedirs(directory)
	return

array = np.array
zeros = np.zeros
ones = np.ones
identity = np.eye
nan = np.nan
pi = np.pi
exp = np.exp

arrays = (np.ndarray,)

def allclose(a,b):
	return np.allclose(a,b)

def norm(a):
	return np.sum(np.absolute(a)**2)

def inner(a,b):
	subscripts = 'uij,vij->uv' if a.ndim >2 and b.ndim > 2 else 'ij,ij->'
	return np.einsum(subscripts,np.conjugate(a),b) 

def dot(a,b):
	return np.matmul(a,b)

def trace(a):
	return np.einsum('ii',a)

def det(a):
	return np.linalg.det(a)

def rank(a):
	return np.linalg.matrix_rank(a)

def inv(a):
	return np.linalg.inv(a)

def solve(a,b):
	return np.linalg.solve(a,b)

def transpose(a):
	return a.T

def conjugate(a):
	return a.conj()

def dagger(a):
	return transpose(conjugate(a))

def absolute(a):
	return np.absolute(a)

def real(a):
	return a.real

def imag(a):
	return a.imag

def astype(a,dtype):
	return a.astype(dtype)

def multiply(a):
	if not a:
		return
	i = a[0]
	for j in a[1:]:
		i = dot(i,j)
	return i

def kron(a,b):
	return np.kron(a,b)

def tensor(a):
	if not a:
		return
	i = a[0]
	for j in a[1:]:
		i = kron(i,j)
	return i

def power(a,k):
	return np.linalg.matrix_power(a,k)

def shuffle(a,i,d,t):

	l = len(i)
	i = {**{j:k for j,k in zip(i,range(l))},**{j:k for j,k in zip((j for j in range(t) if j not in i),range(l,t))}}
	
	shape = a.shape 
	s = a.ndim

	shapes = [d]*(s*t)
	axes = [i[j]+k*t for k in range(s) for j in range(t)]

	return a.reshape(shapes).transpose(axes).reshape(shape)

def flatten(i):
	if isinstance(i,(list,dict,tuple)):
		for j in i:
			yield from flatten(j)
	else:
		yield i

def tree(n):
	k = n-2
	for s in product(range(n),repeat=k):
		v = []
		e = []
		for i in range(k):
			for j in range(n):
				if j not in s and j not in v:
					v.append(j)
					e.append((s[i],j))
					break
		l = tuple(j for j in range(n) if j not in v)
		v.extend(l)
		e.append(l)

		yield e

	return

def group(t,sorting=None):
	G = SymmetricGroup(t)
	if sorting:
		G = generate(G,t)
		indices = sort(G,t)
		G = [G[i] for i in indices]
	return G

def generate(G,t):
	return list(G.generate_schreier_sims())

def partitions(x,t):
	partitions = tuple(sorted(len(cycle) for cycle in cycles(x,t)))
	return partitions

def classes(t):

	G = group(t)
	classes = {partitions(next(iter(i)),t): i for i in G.conjugacy_classes()}

	G = generate(G,t)
	indices = sort(G,t)

	G = [G[i] for i in indices]
	classes = {partition:sorted([G.index(i) for i in classes[partition]]) for partition in sorted(classes)}

	return classes

def cycles(x,t):
	return x.cyclic_form

def size(x,t):
	return t-x.cycles

def support(x,t):
	return list(flatten(ordering(x,t)))

def locality(x,t):
	return len(support(x,t))

def character(x,d,t):
	return d**(t-size(x,t))

def sorting(x,X,t):
	if not common(support(x,t),support(X,t),t):
		return False
	order = support(X,t)
	cycle = ordering(x,t,order=min,orders=lambda i:order.index(min(i)))
	length = len(cycle)
	for k in product(*(range(len(cycle[i])) for i in range(length))):
		index = [order.index(j) for i in range(length) for j in [*cycle[i][k[i]:],*cycle[i][:k[i]]]]
		if sorted(index) == index:
			return True
	return False

def ordering(x,t,order=None,orders=None):
	order = (lambda i:min(i)) if order is None else order
	orders = (lambda i:(len(i),min(i))) if orders is None else orders
	x = cycles(x,t)
	index = {i:x[i].index(order(x[i])) for i in sorted(range(len(x)),key=lambda i:orders(x[i]))}
	x = [[*x[i][index[i]:],*x[i][:index[i]]] for i in index]
	return x

def sort(G,t):

	g = len(G)
	
	indices = range(g)

	def key(i):
		cycle = ordering(G[i],t)
		indices = list(flatten(cycle))
		number = len(indices)
		length = size(G[i],t)
		key = (
			number,
			*sorted(indices),*[-t]*(t-number),
			length,
			*sorted(len(j) for j in cycle),
			*indices,*[-t]*(t-number),
			)
		return key
	
	indices = sorted(indices,key=key)

	return indices

def order(G,t):
	
	g = len(G)

	indices = range(g)

	return indices

def common(support,supports,t,equals=False):
	return (
		all(k in supports for k in support) and
		((equals and (len(supports)==len(support))) or 
		 (not equals and (len(supports)>=len(support))))
		)

def contains(x,X,t):
	return ((size(~x*X,t) == (size(X,t)-size(x,t))) and (size(X,t) >= size(x,t)))

def conditions(string):
	length = len(string)
	# return all(string[i] not in [0,string[(i-1)%(length)]*(i>0)] for i in range(length))
	# return all(string[i] not in [0,*string[:i]] for i in range(length))
	return all(string[i] not in [0,*string[:i],*[string[i+1:]]] for i in range(length))

def run(path,t,d,boolean=None,verbose=None,**kwargs):

	G = group(t,sorting=True)
	g = len(G)

	path = '%s/data.%d.%d.pkl'%(path,t,d)
	parse = lambda obj: obj.real.round(8)+0. if isinstance(obj,arrays) and not obj.dtype == int else str(obj) if not isinstance(obj,str) else obj
	disp = lambda *objs,verbose=True: log(*(parse(obj) for obj in objs),verbose=verbose)
	dtype = 'complex'

	data = [zeros((d,d),dtype=dtype) for i in range(2)]
	for i in range(d):
		data[0][i,i] = exp(i*1j*2*pi/d)
		data[1][(i+1)%d,i] = 1
	data = {index:
		dot(*(power(x,j) for x,j in zip(data,i)))
		for index,i in enumerate(product(range(d),repeat=len(data)))
		}

	datas = {**data,**{(i,j):dot(dagger(data[i]),data[j]) for i in data for j in data}}

	one = lambda t=1: tensor([identity(d)]*t)
	zero = lambda t=1: tensor([zeros((d,d),dtype=dtype)]*t)

	partitions = classes(t)

	objs = {
		't':t,'d':d,
		'strings.localized': {},
		'strings.permutations': {},
		'basis.localized': zeros((g,d**t,d**t),dtype=dtype),
		'basis.permutations': zeros((g,d**t,d**t),dtype=dtype),
		}

	for l in range(2,t+1):

		objs['strings.localized'][l] = zero(l)
		objs['strings.permutations'][l] = zero(l)

		for index,string in enumerate(product(data,repeat=l-1)):

			obj = tensor([datas[(string[(i-1)%(l-1)]*(i>0),string[(i)%(l-1)]*(i<(l-1)))] for i in range(l)])

			if conditions(string):
				objs['strings.localized'][l] += obj

			objs['strings.permutations'][l] += obj

			if verbose and index % d**(l//2) == 0:
				log(string,verbose=verbose)

	for partition in partitions:
		
		l = sum(partition)
		obj = tensor([*[objs['strings.localized'][i] for i in partition],*[datas[0]]*(t-l)])
		_obj = tensor([*[objs['strings.permutations'][i] for i in partition],*[datas[0]]*(t-l)])
		
		for i in partitions[partition]:
			j = list(flatten(cycles(G[i],t)))
			objs['basis.localized'][i] = shuffle(obj,j,d,t)
			objs['basis.permutations'][i] = shuffle(_obj,j,d,t)

	X = inner(objs['basis.localized'],objs['basis.localized'])
	x = inner(objs['basis.localized'],objs['basis.permutations'])

	data = transpose(solve(X,x))

	key = 'data'
	objs = {key:data}
	dump(path,objs)

	disp(data)

	plot(t,path=path)

	return 


def plot(data=None,path=None):
	
	import numpy as np
	import matplotlib
	import matplotlib.pyplot as plt
	np.set_printoptions(linewidth=1000)

	if data is None:
		return
	elif isinstance(data,int):
		t = data
		G = group(t,sorting=True)
		g = len(G)

		a = array([[int(contains(G[j],G[i],t)) for j in range(g)] for i in range(g)])
		b = inv(a)

		data = [a,b]

	for x in data:
		print(x)

	fig,axes = plt.subplots(1,len(data))
	
	name = 'colorbar'
	null = 0
	values = array(list(set([i for x in data for i in x[x!=null].flatten()])))
	vmin,vmax = min(values),max(values)
	lengths = {0:len(values[values<null]),1:len(values[values>null])}
	alphas = {0:0.8,1:1}
	colors = []
	for i,value in enumerate(values):
		index = value>null
		color = plt.cm.viridis
		length = lengths[index]
		alpha = alphas[index]
		color = list(color((abs(i)+1)/(length+2)))
		color[-1] = alpha
		color = tuple(color)

		colors.append(color)

	cmap = matplotlib.colors.LinearSegmentedColormap.from_list(name,colors)
	cmap.set_bad('white')

	options = dict(cmap=cmap,norm=plt.Normalize(vmin,vmax),interpolation='none', aspect=1)

	for x,ax in zip(data,axes):
		
		x = np.ma.masked_where(x==null, x)

		ax.matshow(x,**options)
		
		ax.axis(True)
		ax.set_xticks([])
		ax.set_yticks([])

	if path is not None:
		path = os.path.join(os.path.dirname(path),'plot.basis.pdf')
		mkdir(path)
		fig.subplots_adjust()
		fig.savefig(path)

	return


def main(*args,**kwargs):

	path = str(args[0] if len(args)>0 else 2)
	t = int(args[1] if len(args)>1 else 2)
	d = int(args[2] if len(args)>2 else t)

	boolean = 1
	verbose = 1
	
	run(path=path,t=t,d=d,boolean=boolean,verbose=verbose)

	return

if __name__ == '__main__':

	args = sys.argv[1:]

	main(*args)