#!/usr/bin/env python

import sys,os

import sympy as sp
from sympy import Symbol,Matrix,simplify
from sympy.combinatorics.named_groups import SymmetricGroup,Permutation

from haarpy import weingarten_element

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

def gram(x,y,d,t,local=False,supports={}):
	z = ~x*y
	return character(z,d,t)/(d**t)

def weingarten(x,y,d,t,local=False,supports={}):
	z = ~x*y
	if local:
		a,b,c = supports[x],supports[y],supports[z]
		z = Permutation([[c.index(j) for j in i] for i in cycles(z,t)])
		t = max(len(a),len(b),len(c))
	if t == 0:
		return 1
	return weingarten_element(z,t,d)*(d**t)

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

def conditions(x,X,t):
	return contains(x,X,t)# and sorting(x,X,t)

def run(path,t,d,e,boolean=None,verbose=None,**kwargs):

	G = group(t,sorting=True)
	g = len(G)

	path = '%s/data.%d.pkl'%(path,t)

	z = d*e

	orthogonal = True
	local = True
	equals = z==d

	supports = {}
	commons = {}
	index = {}
	values = {'data':{},'basis':{}}
	indices = []
	elements = []

	for i in range(g):
		supports[G[i]] = support(G[i],t)

	for i in range(g):
		for j in range(g):
			for k in [True,False]:
				commons[i,j,k] = common(supports[G[j]],supports[G[i]],t,equals=k)
			index[i,j] = conditions(G[j],G[i],t)
			
			values['data'][i,j] = (
					gram(G[0],G[i],e,t,local=local,supports=supports)*
					weingarten(G[i],G[j],e,t,local=local,supports=supports)*
					gram(G[j],G[0],d,t,local=local,supports=supports)
					)
			values['basis'][i,j] = (
					(gram(G[i],G[j],d,t))/
				   ((gram(G[0],G[i],d,t))*
					(gram(G[0],G[j],d,t)))
					) if commons[i,j,True] else 0
			
			indices.append((i,j))
			elements.append((i,j))

	data = load(path)

	if boolean or data is None:
		i,j = 0,0
		data = {attr: Matrix([[0 for j in range(g)] for i in range(g)]) for attr in values}
	else:
		i,j = list(data.keys())[-1]
		data = list(data.values())[-1]

	indices = indices[indices.index((i,j))-((i*j)>0):]
	elements = elements[:]

	for i,j in indices:

		if orthogonal and not commons[i,j,equals]:
			continue

		data['data'][i,j] = 0
	
		data['basis'][i,j] = values['basis'][i,j]

		for k,l in elements:

			if ((index[k,i] and index[l,j]) and ((not local) or (commons[i,k,False] and commons[i,l,False]))):

				data['data'][i,j] += values['data'][k,l]

			if commons[i,j,True] and (index[i,k] and index[j,l]) and ((k != i) or (l != j)):
				
				data['basis'][i,j] -= data['basis'][k,l]

		data['data'][i,j] = simplify(data['data'][i,j])

		data['basis'][i,j] = simplify(data['basis'][i,j])

		data['data'][i,j] = data['data'][i,j].subs(e,z)

		log(i,j,*(data[attr][i,j] for attr in data),verbose=verbose)

		key = (i,j)
		objs = {key:data}
		dump(path,objs)

	return

def main(*args,**kwargs):

	path = str(args[0] if len(args)>0 else '.')
	t = int(args[1] if len(args)>1 else 2)

	d = Symbol('d')
	e = Symbol('e')

	boolean = 1
	verbose = 1
	
	run(path=path,t=t,d=d,e=e,boolean=boolean,verbose=verbose)

	return


if __name__ == '__main__':

	args = sys.argv[1:]

	main(*args)