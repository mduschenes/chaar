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

	directory = os.path.dirname(path)
	if directory and not os.path.exists(directory):
		os.makedirs(directory)

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
		indices = sort(G,t)
		G = generate(G,t)
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
	
	indices = sort(G,t)
	G = generate(G,t)

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
	return d**(t-size(x,t))/(d**t)

def gram(x,y,d,t,local=False,supports={}):
	z = ~x*y
	if local:
		return character(z,d,t)
	else:
		return character(z,d,t)

def weingarten(x,y,d,t,local=False,supports={}):
	z = ~x*y
	if local:
		a,b = supports[x],supports[y]
		if (len(a)<2):
			return 1
		l = max(len(a),len(b))
		c = list(supports[z])
		z = Permutation([[c.index(j) for j in i] for i in cycles(z,t)])
		return weingarten_element(z,l,d)*(d**l)
	else:
		return weingarten_element(z,t,d)*(d**t)

def sort(G,t):
	G = generate(G,t)
	g = len(G)
	key = lambda i: (
			size(G[i],t),
			len([j for j in cycles(G[i],t)]),
			*(len(j) for j in cycles(G[i],t)),
			tuple(((tuple(j) for j in cycles(G[i],t))))
			)
	indices = list(sorted(range(g),key=key))
	return indices

def common(support,supports,t,equals=False):
	return (
		all(k in supports for k in support) and
		((equals and (len(supports)==len(support))) or 
		 (not equals and (len(supports)>=len(support))))
		)

def ordering(x,t,order=min,orders=min):
	x = cycles(x,t)
	index = {i:x[i].index(order(x[i])) for i in sorted(range(len(x)),key=lambda i:orders(x[i]))}
	x = [[*x[i][index[i]:],*x[i][:index[i]]] for i in index]
	return x

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
	indices = []
	elements = []

	for i in range(g):
		supports[G[i]] = support(G[i],t)

	for i in range(g):
		for j in range(g):
			for k in [True,False]:
				commons[i,j,k] = common(supports[G[j]],supports[G[i]],t,equals=k)
			index[i,j] = conditions(G[j],G[i],t)
			indices.append((i,j))
			elements.append((i,j))

	data = load(path)

	if boolean or data is None:
		i,j = 0,0
		data = {attr: Matrix([[0 for j in range(g)] for i in range(g)]) for attr in ['data','norm']}
	else:
		i,j = list(data.keys())[-1]
		data = list(data.values())[-1]

	indices = indices[indices.index((i,j))-((i*j)>0):]
	elements = elements[:]
	tmp = {attr:0 for attr in data}

	for i,j in indices:

		tmp['data'] = 0
	
		tmp['norm'] = ((character(~G[i]*G[j],d,t))/
					  ((character(G[i],d,t))*
					   (character(G[j],d,t)))) if commons[i,j,True] else 0

		if orthogonal and not commons[i,j,equals]:
			continue

		for k,l in elements:

			if ((index[k,i] and index[l,j]) and ((not local) or (commons[i,k,False] and commons[i,l,False]))):

				tmp['data'] += (
					gram(G[0],G[k],e,t,local=local,supports=supports)*
					weingarten(G[k],G[l],e,t,local=local,supports=supports)*
					gram(G[l],G[0],d,t,local=local,supports=supports)
					)

			if commons[i,j,True] and (index[i,k] and index[j,l]) and (k != i) and (l != j):
				
				tmp['norm'] -= data['norm'][k,l]

		tmp['data'] = simplify(tmp['data'])

		tmp['norm'] = simplify(tmp['norm'])

		tmp['data'] = tmp['data'].subs(e,z)

		data['data'][i,j] = tmp['data']

		data['norm'][i,j] = tmp['norm']

		log(i,j,*(data[attr][i,j] for attr in data),verbose=verbose)

		dump(path,{(i,j):data})

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