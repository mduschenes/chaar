#!/usr/bin/env python

import sys,os

from sympy import Symbol,Matrix,Integer,simplify,log,LT,Poly,Add
from sympy.combinatorics.named_groups import SymmetricGroup

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
matplotlib.use('pdf')

from itertools import product
import pickle
import logging

logger = logging.getLogger(__name__)	
logging.basicConfig(level=logging.INFO,format='%(message)s',stream=sys.stdout)
log = lambda *message,verbose=True,**kwargs: logger.info('\t'.join(str(i) for i in message) if len(message)>1 else message[0] if len(message)>0 else "") if verbose else None
np.set_printoptions(linewidth=1000)

def group(t,sorting=None):
	G = SymmetricGroup(t)
	if sorting:
		indices = sort(G,t)
		G = generate(G,t)
		G = [G[i] for i in indices]
	return G

def generate(G,t):
	return list(G.generate_schreier_sims())

def cycles(x,t):
	return x.cyclic_form

def size(x,t):
	return t-x.cycles

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

def order(G,t):
	g = len(G)
	partitions = {}
	for i in range(g):
		p = tuple(set(flatten(cycles(G[i],t))))
		if p not in partitions:
			partitions[p] = []

		partitions[p].append(i)
	key = lambda i: (
		len(i),
		*i
		)
	indices = list(sorted(partitions,key=key))

	partitions = {i: partitions[i] for i in indices}
	key = lambda i: (
			size(G[i],t),
			)

	indices = [j for i in partitions for j in partitions[i]]
	return indices

def number(expression,substitutions={},**kwargs):

	def substitute(expression):
		substitutions.update({attr: 
			Integer(substitutions[attr]) 
			for attr in substitutions
			if isinstance(substitutions[attr],int)
			})
		expression = expression.subs(substitutions)
		return expression

	def args(expression):
		while len(expression.args):
			expression = expression.args[-1]
		return expression

	def leading(expression):
		try:
			expression = LT(expression)
		except:
			pass
		return expression

	def terms(expression):
		
		expressions = [
			(numerator,denominator)
			for term in Add.make_args(expression)
			for numerator,denominator in [term.as_numer_denom()]
			]
		
		if expression == 0:
			return default

		while expressions and expression:
			
			expr = [
				(leading(numerator),leading(denominator))
				for numerator,denominator in expressions
				]
		
			expression = simplify(sum(numerator/denominator 
				for numerator,denominator in expr))
		
			if expression == 0:
				expressions = [(numerator - (denominator*num/den),denominator) 
					for (numerator,denominator),(num,den) in zip(expressions,expr)
					]
			else:
				numerator,denominator = expression.as_numer_denom()
				expression = leading(numerator)/leading(denominator)
				expressions = None

		return expression

	default = -1

	try:
		expression = -args(terms(substitute(expression)))
	except Exception as exception:
		expression = default

	return expression

def process(data,checkpoint,t,k,n,boolean=None,verbose=None):

	data,norm = (data['data'],data['norm']) if isinstance(data,dict) else (data,None)

	data = data.copy()
	shape = data.shape
	default = -1

	elements = group(t)
	indices = sort(elements,t)
	elements = generate(elements,t)
	elements = [elements[i] for i in indices]
	indices = order(elements,t)

	def process(data,unique=None,index=None,indices=None):
		
		d = Symbol('d')
		e = Symbol('e')

		options = dict(substitutions={e:d**n})

		data = data

		if unique is not None:
			return data,unique

		elements = [i for i in product(*(range(i) for i in shape))]

		if index is not None:
			i,j = index
		else:
			i,j = 0,0

		elements = elements[elements.index((i,j))-((i*j)>0):]		

		if index is None:
			if k is not None and norm is not None:
				tmp = data
				for i in range(1,k):
					tmp = tmp*norm*data
				data = tmp

		for i,j in elements:

			if data[i,j] not in [0,1]:
				data[i,j] = number(data[i,j],**options)
			elif data[i,j] in [1]:
				data[i,j] = 0
			elif data[i,j] in [0]:
				data[i,j] = -1
		
			if checkpoint:
				tmp = {(i,j):data}
				dump(checkpoint,tmp)

			# log(i,j,data[i,j],verbose=verbose)

		data = np.array(data).astype(int)

		unique = np.unique(data[data>=0])

		if indices is not None:
			data = np.array([[data[i,j] for j in indices] for i in indices])

		return data,unique

	
	if checkpoint:
		
		tmp = load(checkpoint)
		
		if boolean or tmp is None:
			
			index = None
			data = data
			unique = None

		else:

			if all(attr in tmp for attr in ['data','unique']):

				index = None
				data = tmp['data']
				unique = tmp['unique']

			else:

				index = list(tmp.keys())[-1]
				data = list(tmp.values())[-1]
				unique = None

	data,unique = process(data,unique,index=index,indices=indices)

	if checkpoint:
		tmp = {'data':data,'unique':unique}
		dump(checkpoint,tmp)

	return data,unique

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

def plot(path,t,k,n,boolean=None,verbose=None,**kwargs):

	data = '%s/data.%d.pkl'%(path,t)
	figure = '%s/plot.%d.%d.%d.pdf'%(path,t,k,n)
	checkpoint = '%s/data.tmp.%d.%d.%d.pkl'%(path,t,k,n)
	mplstyle = 'plot.mplstyle'

	data = load(data)

	if data is None:
		return

	i,j = list(data.keys())[-1]
	data = list(data.values())[-1]

	log(i,j,kwargs,verbose=verbose)

	data,unique = process(data,checkpoint,t=t,k=k,n=n,boolean=boolean,verbose=verbose,**kwargs)

	l = len(unique)
	L = 30

	log(data,verbose=verbose)

	with matplotlib.style.context(mplstyle):

		plt.close('all')
		fig, ax = plt.subplots()
		# cmap = matplotlib.colors.ListedColormap([plt.cm.viridis((i)/(L)) for i in range(l)])
		cmap = matplotlib.colors.ListedColormap([
				*[plt.cm.viridis(1.00*i/(0.8*L)) for i in range(0,10,1)],
				*[plt.cm.viridis(1.00*10/(0.8*L) + i/(1.7*L)) for i in range(10,L,1)]
				])
		cmap.set_under('white')
		cmap.set_over('white')
		# plot = ax.imshow(data, cmap=cmap, vmin=0, vmax=l, aspect=1,interpolation=None, rasterized=True)
		plot = ax.matshow(data, cmap=cmap, vmin=0, vmax=L,interpolation='none', aspect=1)
		cbar = fig.colorbar(plot, orientation="vertical")	
		# cbar.set_ticks(ticks=[i+0.5 for i in range(l)])
		# cbar.set_ticklabels(ticklabels=['$%d%s$'%(i,' ' if i<10 else '') for i in unique])
		cbar.set_ticks(ticks=[i+0.5 for i in range(0,L,4)])
		cbar.set_ticklabels(ticklabels=['$%d%s$'%(i,' ' if i<10 else '') for i in range(0,L,4)])		
		cbar.set_label(label="$l$",loc='center',rotation=90)
		ax.set_aspect('equal')
		ax.axis(True)
		ax.set_xticks([])
		ax.set_yticks([])
		# ax.set_title(r'$t = %d ~,~ k = %d$'%(kwargs.get('t'),kwargs.get('k'))) if kwargs.get('t') is not None and kwargs.get('k') is not None else None
		
		if figure is not None:
			# fig.subplots_adjust()
			# fig.tight_layout()
			# fig.savefig(figure,bbox_inches="tight")
			fig.subplots_adjust()
			# fig.tight_layout()
			fig.savefig(figure)			
		else:
			plt.show()

	return



def main(*args,**kwargs):

	path = str(args[0] if len(args)>0 else '.')	
	t = int(args[1] if len(args)>1 else 2)
	k = int(args[2] if len(args)>2 else 1)
	n = int(args[3] if len(args)>3 else 2)

	boolean = 1
	verbose = 1

	plot(path=path,t=t,k=k,n=n,boolean=boolean,verbose=verbose)

	return

if __name__ == '__main__':

	args = sys.argv[1:]

	main(*args)