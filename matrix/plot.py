#!/usr/bin/env python

import sys,os

from sympy import Symbol,Matrix,Integer,simplify,log,LT,Poly,Add,Trace
from sympy.combinatorics.named_groups import SymmetricGroup

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
matplotlib.use('pdf')

from itertools import product
from math import prod,factorial
import pickle
import logging

logger = logging.getLogger(__name__)	
logging.basicConfig(level=logging.INFO,format='%(message)s',stream=sys.stdout)
log = lambda *message,verbose=True,**kwargs: logger.info('\t'.join(str(i) for i in message) if len(message)>1 else message[0] if len(message)>0 else "") if verbose else None

def group(t,sorting=None):
	G = SymmetricGroup(t)
	if sorting:
		G = generate(G,t)
		indices = sort(G,t)
		G = [G[i] for i in indices]
	return G

def generate(G,t):
	return list(G.generate_schreier_sims())

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

def matrix(path,t,k,n,boolean=None,verbose=None,**kwargs):

	def process(data,checkpoint,t,k,n,boolean=None,verbose=None):

		data,basis = (data['data'],data['basis']) if isinstance(data,dict) else (data,None)

		shape = data.shape
		default = -1

		elements = group(t,sorting=True)
		indices = order(elements,t)

		def process(data,unique=None,index=None,indices=None):

			d = Symbol('d')
			e = Symbol('e')

			options = dict(substitutions={e:d**n})

			data = data

			if unique is not None:
				return data,unique

			if index is not None:
				i,j = index
			else:
				i,j = 0,0

			elements = [i for i in product(*(range(i) for i in shape))]
			elements = elements[elements.index((i,j))-((i*j)>0):]

			if index is None:

				if k is not None and basis is not None:
					tmp = data
					for i in range(1,k):
						data = data*basis*tmp

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

			if tmp is None:

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
			mkdir(figure)
			fig.subplots_adjust()
			fig.savefig(figure)			
		else:
			plt.show()

	return


def norm(path,t,k,n,boolean=None,verbose=None,**kwargs):

	def process(path,t,k,n,boolean=None,verbose=None):

		D = range(0,6+1)
		E = [-1,-2,-4,1,2]

		T = range(2,t+1)
		K = [1,k]
		N = range(n)

		values = []

		for t in T:

			data = path%(t)
			data = load(data)

			if data is None:
				return

			data = list(data.values())[-1]
			data,basis = (data['data'],data['basis']) if isinstance(data,dict) else (data,None)

			for k in K:

				tmp = data
				for i in range(1,k):
					data = data*basis*tmp

				value = data*basis
				value = Trace(value.T*value)

				for d in D:
					d = 2**d
					if d < t:
						continue
					for e in E:
						options = dict(d=d,e=d**e if e>0 else -e)
						values.append({'t':t,'k':k,'d':d,'e':e,'data':float(number(value,options))/factorial(t)})

		data = values

		return data

	def number(expression,substitutions={},**kwargs):

		def substitute(expression):
			for substitution in substitutions:
				expression = expression.subs({substitution:substitutions[substitution]})
			return expression

		number = simplify(substitute(expression))

		return number

	def permute(kwargs):
		return [dict(zip(kwargs,values)) for values in product(*(kwargs[key] for key in kwargs))]

	data = '%s/data.%%d.pkl'%(path)
	figure = '%s/norm.%d.%d.%d.pdf'%(path,t,k,n)
	mplstyle = 'plot.mplstyle'

	data = process(data,t=t,k=k,n=n,boolean=boolean,verbose=verbose,**kwargs)

	attributes = {attr:sorted(set(i[attr] for i in data),key=lambda i:i if attr not in ['e'] else 100+i if i>0 else -i) for attr in ['t','k','e']}
	size = prod(len(attributes[attr]) for attr in attributes)

	with matplotlib.style.context(mplstyle):

		fig,ax = plt.subplots()
		obj = dict(fig=fig,ax=ax)

		for index,attrs in enumerate(permute(attributes)):

			indices = [i for i in data if all(i[attr]==attrs[attr] and type(i[attr])==type(attrs[attr]) for attr in attrs)]
			indices = sorted(indices,key=lambda i:i['d'])

			x = [i['d'] for i in indices]
			y = [i['data'] for i in indices]

			options = dict(
				label='$%s$'%('~'.join([
					'%d'%(attrs['t']),
					'%d'%(attrs['k']),
					'1' if attrs['e']==0 else
					'd' if attrs['e']==1 else
					'%s'%(-attrs['e']) if attrs['e']<0 else
					'd^{%s}'%(attrs['e'] if isinstance(attrs['e'],int) else '%s/2'%(int(2*attrs['e'])))])),
				color=getattr(plt.cm,'viridis')((attributes['e'].index(attrs['e'])/len(attributes['e']))),
				marker={2:'o',3:'^',4:'s'}.get(attrs['t']),
				linestyle={1:'-',2:':',3:'dashdot'}.get(attrs['k']),
				markersize=8,linewidth=2.5,alpha=0.8,zorder=50,
				)

			ax.plot(x,y,**options)

			if index == (size-1):

				for t in attributes['t']:

					options = dict(
						label=None,
						color='grey',
						marker={2:'o',3:'^',4:'s'}.get(t),
						linestyle='',
						markersize=8,linewidth=2.5,alpha=0.8,zorder=100,
						)

					ax.plot([1.75,75],[1/factorial(t),1/factorial(t)],**options)


					options = dict(
						color='grey',
						linestyle='--',
						linewidth=2.5,alpha=0.8,zorder=100,
						)

					ax.hlines(y=1/factorial(t),xmin=min(x)*(-4),xmax=max(x)*1.5,**options)

				options = dict(
					s="$\\scriptstyle{||{\\Tau^{(t)}_{\\mathcal{D}(d)}}||^{2}}~\\Huge{/}~\\scriptstyle{||{\\Tau^{(t)}_{\\mathcal{U}(d)}}||^{2}}$",
					color="grey"
					)

				ax.text(x=max(x)*0.535,y=1/factorial(t)*0.7,**options)


			ax.set_ylabel(ylabel="$\\norm{\\Tau^{(t)}_{\\mathcal{C}(d,d_{\\mathcal{E}})}}^{2}/\\norm{\\Tau^{(t)}_{\\mathcal{U}(d)}}^{2}$")
			ax.set_xlabel(xlabel="$d$")

			ax.set_xscale(value="log",base=2)
			ax.set_yscale(value="log",base=10)
			ax.set_xlim(xmin=1.4,xmax=90)
			ax.set_ylim(ymin=5e-3,ymax=2e0)
			ax.set_xticks(ticks=[2**i for i in [1,2,3,4,5,6]])
			ax.set_xticklabels(labels=['$2^{%d}$'%(i) if i>1 else '$2$' if i>0 else '$1$' for i in [1,2,3,4,5,6]])
			ax.set_yticks(ticks=[1/factorial(i) for i in [*sorted(attributes['t'],reverse=True),1]])
			ax.set_yticklabels(labels=['$1/%d!$'%(i) if i>1 else '$%d$'%(i) for i in [*sorted(attributes['t'],reverse=True),1]])

			ax.grid(visible=True)

			ax.legend(
				title="$t~,k~,d_{\\mathcal{E}}$",
				loc="lower left",
				ncol=len(attributes['t'])*len(attributes['k']),
				title_fontsize=22,
				prop={"size":20},
				markerscale=1,
				handlelength=3
			)

			if index == (size-1):
				fig.set_size_inches(w=20,h=12)
				fig.subplots_adjust()
				fig.tight_layout()
				fig.savefig(fname=figure)

	return


def trace(path,t,k,n,boolean=None,verbose=None,**kwargs):

	def process(path,t,k,n,boolean=None,verbose=None):

		D = range(0,6+1)
		E = [-1,-2,-4,1,2]

		T = range(2,t+1)
		K = [1,k]
		N = range(n)

		values = []

		for t in T:

			data = path%(t)
			data = load(data)

			if data is None:
				return

			data = list(data.values())[-1]
			data,basis = (data['data'],data['basis']) if isinstance(data,dict) else (data,None)

			for k in K:

				tmp = data
				for i in range(1,k):
					data = data*basis*tmp

				value = data*basis
				value = Trace(value)

				for d in D:
					d = 2**d
					if d < t:
						continue
					for e in E:
						options = dict(d=d,e=d**e if e>0 else -e)
						values.append({'t':t,'k':k,'d':d,'e':e,'data':float(number(value,options))/factorial(t)})

		data = values

		return data

	def number(expression,substitutions={},**kwargs):

		def substitute(expression):
			for substitution in substitutions:
				expression = expression.subs({substitution:substitutions[substitution]})
			return expression

		number = simplify(substitute(expression))

		return number

	def permute(kwargs):
		return [dict(zip(kwargs,values)) for values in product(*(kwargs[key] for key in kwargs))]

	data = '%s/data.%%d.pkl'%(path)
	figure = '%s/trace.%d.%d.%d.pdf'%(path,t,k,n)
	mplstyle = 'plot.mplstyle'

	data = process(data,t=t,k=k,n=n,boolean=boolean,verbose=verbose,**kwargs)

	attributes = {attr:sorted(set(i[attr] for i in data),key=lambda i:i if attr not in ['e'] else 100+i if i>0 else -i) for attr in ['t','k','e']}
	size = prod(len(attributes[attr]) for attr in attributes)

	with matplotlib.style.context(mplstyle):

		fig,ax = plt.subplots()
		obj = dict(fig=fig,ax=ax)

		for index,attrs in enumerate(permute(attributes)):

			indices = [i for i in data if all(i[attr]==attrs[attr] and type(i[attr])==type(attrs[attr]) for attr in attrs)]
			indices = sorted(indices,key=lambda i:i['d'])

			x = [i['d'] for i in indices]
			y = [i['data'] for i in indices]

			options = dict(
				label='$%s$'%('~'.join([
					'%d'%(attrs['t']),
					'%d'%(attrs['k']),
					'1' if attrs['e']==0 else
					'd' if attrs['e']==1 else
					'%s'%(-attrs['e']) if attrs['e']<0 else
					'd^{%s}'%(attrs['e'] if isinstance(attrs['e'],int) else '%s/2'%(int(2*attrs['e'])))])),
				color=getattr(plt.cm,'viridis')((attributes['e'].index(attrs['e'])/len(attributes['e']))),
				marker={2:'o',3:'^',4:'s'}.get(attrs['t']),
				linestyle={1:'-',2:':',3:'dashdot'}.get(attrs['k']),
				markersize=8,linewidth=2.5,alpha=0.8,zorder=50,
				)

			ax.plot(x,y,**options)

			if index == (size-1):

				for t in attributes['t']:

					options = dict(
						label=None,
						color='grey',
						marker={2:'o',3:'^',4:'s'}.get(t),
						linestyle='',
						markersize=8,linewidth=2.5,alpha=0.8,zorder=100,
						)

					ax.plot([1.75,75],[1/factorial(t),1/factorial(t)],**options)


					options = dict(
						color='grey',
						linestyle='--',
						linewidth=2.5,alpha=0.8,zorder=100,
						)

					ax.hlines(y=1/factorial(t),xmin=min(x)*(-4),xmax=max(x)*1.5,**options)

				options = dict(
					s="$\\scriptstyle{\\textrm{Tr}[\\Tau^{(t)}_{\\mathcal{D}(d)}]}~\\Huge{/}~\\scriptstyle{\\textrm{Tr}[\\Tau^{(t)}_{\\mathcal{U}(d)}]}$",
					color="grey"
					)

				ax.text(x=max(x)*0.535,y=1/factorial(t)*0.7,**options)


			ax.set_ylabel(ylabel="$\\textrm{Tr}[\\Tau^{(t)}_{\\mathcal{C}(d,d_{\\mathcal{E}})}]~/~\\textrm{Tr}[\\Tau^{(t)}_{\\mathcal{U}(d)}]$")
			ax.set_xlabel(xlabel="$d$")

			ax.set_xscale(value="log",base=2)
			ax.set_yscale(value="log",base=10)
			ax.set_xlim(xmin=1.4,xmax=90)
			ax.set_ylim(ymin=5e-3,ymax=2e0)
			ax.set_xticks(ticks=[2**i for i in [1,2,3,4,5,6]])
			ax.set_xticklabels(labels=['$2^{%d}$'%(i) if i>1 else '$2$' if i>0 else '$1$' for i in [1,2,3,4,5,6]])
			ax.set_yticks(ticks=[1/factorial(i) for i in [*sorted(attributes['t'],reverse=True),1]])
			ax.set_yticklabels(labels=['$1/%d!$'%(i) if i>1 else '$%d$'%(i) for i in [*sorted(attributes['t'],reverse=True),1]])

			ax.grid(visible=True)

			ax.legend(
				title="$t~,k~,d_{\\mathcal{E}}$",
				loc="lower left",
				ncol=len(attributes['t'])*len(attributes['k']),
				title_fontsize=22,
				prop={"size":20},
				markerscale=1,
				handlelength=3
			)

			if index == (size-1):
				fig.set_size_inches(w=20,h=12)
				fig.subplots_adjust()
				fig.tight_layout()
				fig.savefig(fname=figure)

	return

def plot(path,t,k,n,boolean=None,verbose=None,**kwargs):

	# matrix(path,t,k,n,boolean=None,verbose=None,**kwargs)

	# norm(path,t,k,n,boolean=None,verbose=None,**kwargs)

	trace(path,t,k,n,boolean=None,verbose=None,**kwargs)

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