#!/usr/bin/env python

import sys
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sympy import Symbol,Matrix,Inverse, simplify, fraction
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.named_groups import SymmetricGroup
from scipy.special import comb as binom
from scipy.special import factorial as factorial
from natsort import natsorted
from itertools import product
from functools import reduce,partial
from copy import deepcopy
from math import prod

import threading
import multiprocessing
from joblib import Parallel, delayed
from time import time

import pickle
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(message)s',stream=sys.stdout)
log = lambda *message,verbose=True,**kwargs: logger.info('\t'.join(str(i) for i in message) if len(message)>1 else message[0] if len(message)>0 else "") if verbose else None

class Function(object):
	def __init__(self,func,*args,**kwargs):
		self.func = func
		self.args = args
		self.kwargs = kwargs
		return

	def __call__(self,index):
		return self.func(index,*self.args,**self.kwargs)


class Callback(object):
	def __init__(self,func,*args,**kwargs):
		self.func = func
		self.args = args
		self.kwargs = kwargs

		self.lock = threading.Lock()
		return

	def __call__(self,data):
		with self.lock:
			self.func(data,*self.args,**self.kwargs)
		return

class Error(object):
	def __init__(self,*args,**kwargs):
		self.args = args
		self.kwargs = kwargs
		return

	def __call__(self,error):
		log(error)
		return

class Parallelize(object):
	def __init__(self,processes,method,function,callback,*args,**kwargs):
		self.processes = processes if processes is not None else multiprocessing.cpu_count()-1
		self.method = method if method is not None else None
		self.function = Function(function,*args,**kwargs)
		self.callback = Callback(callback,*args,**kwargs)
		self.error_callback = Error(*args,**kwargs)
		self.args = args
		self.kwargs = kwargs
		return

	def __call__(self,indices):


		if isinstance(indices,int):
			size = indices
			indices = range(size)
		elif isinstance(indices,tuple):
			size,repeat = indices
			indices = product(range(size),repeat=repeat)

		with multiprocessing.Pool(processes=self.processes) as parallel:

			if self.method is None or not hasattr(parallel,self.method):
				def func(function,args=(),kwargs={},callback=None,error_callback=None):
					callback(function(*args,**kwargs))
					return
			else:
				func = getattr(parallel,self.method) 

			for index in indices:
				tmp = func(self.function,(index,),callback=self.callback,error_callback=self.error_callback)

			parallel.close()
			parallel.join()

		# with Parallel(n_jobs=self.processes) as parallel:

		# 	func = delayed(self.function)

		# 	tmp = parallel(func(index) for index in indices)

		# 	# func = self.function

		# 	# tmp = []
		# 	# for index in indices:
		# 	# 	tmp.append(func(index))

		# 	data = sum(tmp)

		return



def function(index,*args,**kwargs):

	i,j,k,l = index
	W,G,t,data = kwargs['W'],kwargs['G'],kwargs['t'],kwargs['data']
	G = list(SymmetricGroup(t).generate_schreier_sims())
	data = {attr: (W[attr][i,j]*W[attr][k,l]*
		((data[attr]['d'])**((((~G[i])*G[k]).cycles)+(((~G[j])*G[l]).cycles)))*
		((data[attr]['e'])**(G[i].cycles + G[k].cycles))
		) for attr in W}
	return data

def callback(tmp,*args,**kwargs):
	for attr in tmp:
		kwargs['data'][attr]['data'] += tmp[attr]
	return

def func(*args,**kwargs):
	def catalan(k):
		return binom(2*k,k)/(k+1)

	def mobius(permutation):
		return prod((-1)*(len(cycle)-1)*catalan(len(cycle)-1) for cycle in permutation.cyclic_form)

	def group(t):
		return list(SymmetricGroup(t).generate_schreier_sims())

	defaults = {'t':1,'d':1}
	kwargs.update({default:kwargs.get(default,defaults[default]) 
		for default in defaults})

	for kwarg in kwargs:
		if kwarg in ['t']:
			if kwargs[kwarg] is None:
				kwargs[kwarg] = defaults[kwarg]
		elif kwarg in ['d']:
			if kwargs[kwarg] is None:
				kwargs[kwarg] = {'d':1,'e':1}
			if isinstance(kwargs[kwarg],int):
				kwargs[kwarg] = {'d':kwargs[kwarg],'e':kwargs[kwarg]}
			if not isinstance(kwargs[kwarg],dict):
				kwargs[kwarg] = {attr: i for attr,i in zip(['d','e'],kwargs[kwarg])}
			if isinstance(kwargs[kwarg],dict):
				kwargs[kwarg] = {attr: (*kwargs[kwarg][attr],) if not isinstance(kwargs[kwarg][attr],int) else (kwargs[kwarg][attr],) for attr in kwargs[kwarg]}

	attrs = {attr: kwargs[kwarg][attr] 
		for kwarg in kwargs 
		if isinstance(kwargs[kwarg],(dict,list,tuple))
		for attr in kwargs[kwarg] 
		}
	attrs = permute(attrs)

	t = kwargs['t']
	G = group(t)
	size = len(G)
	repeat = 4
	W  = {attr: np.array([[
		prod((attrs[attr][k] for k in attrs[attr] if k in ['d','e']))**(((~G[i])*G[j]).cycles) 
		for j in range(size)] 
		for i in range(size)],
		dtype=float)
		for attr in attrs}

	G = {attr: G for attr in attrs}
	W = {attr: np.linalg.pinv(W[attr]) for attr in attrs}

	data = {attr: {'data':0,**kwargs,**attrs[attr]} for attr in attrs}

	processes = 10
	method = 'apply_async'
	indices = (size,repeat)
	args = tuple()
	kwargs = dict(data=data,W=W,G=G,t=t)
	parallelize = Parallelize(processes,method,function,callback,*args,**kwargs)
	parallelize(indices)

	return data


def setter(obj,key,value,delimiter='.'):
	key = key.split(delimiter) if isinstance(key,str) else key
	try:
		obj = obj.get(key[0],None)
		for i in key[1:]:
			obj = getattr(obj,i)
		if callable(obj):
			obj(**value)
		else:
			obj = value
	except Exception as exception:
		pass
	return

def load(path,default=None):
	ext = path.split('.')[-1] if path is not None else None
	data = default
	if ext in ['json']:
		try:
			with open(path,'r') as obj:
				data = json.load(obj)
		except Exception as exception:
			pass
	return data

def dump(data,path):
	ext = path.split('.')[-1] if path is not None else None
	if ext in ['json']:
		try:
			with open(path,'w') as obj:
				json.dump(data,obj,indent=4)
		except Exception as exception:
			pass
	else:
		pass
	return

def permute(kwargs,permutations=None,func=None):

	if permutations is None:
		permutations = kwargs

	if func is None:
		def func(index,kwargs,permutations=None):
			if permutations is None:
				permutations = kwargs
			kwargs = dict(zip(permutations,kwargs))
			for kwarg in kwargs:
				if isinstance(kwargs[kwarg],str) and kwargs[kwarg] in kwargs:
					kwargs[kwarg] = kwargs[kwargs[kwarg]]
				elif callable(kwargs[kwarg]):
					kwargs[kwarg] = kwargs[kwarg](kwargs)

			key = str(index)
			value = kwargs

			data = [(key,value)]

			return data

	kwargs = {key:value
		for index,kwarg in enumerate(
		product(*(kwargs[key] for key in permutations)))
		for key,value in func(index,kwarg,permutations=permutations)
		}
	return kwargs

def process(obj,data):

	def func(obj,data):
		return

	func(obj,data)

	return

def plot(data,attributes=None,options=None,settings=None,mplstyle=None):

	if data is None:
		return

	attributes = {} if attributes is None else attributes
	options = {} if options is None else options
	settings = {} if settings is None else settings

	mplstyle = 'plot.mplstyle' if mplstyle is None else mplstyle
	
	variables = {attr: list(natsorted(set(data[i][attr] for i in data)) )
					  for attr in set((attr for i in data for attr in data[i] 
					  if isinstance(data[i][attr],(int,float,str))))}


	attributes = {attr: attributes[attr] if attributes[attr] is not None else variables[attr] for attr in attributes}
	attributes = permute(attributes)

	values = {attr: options.pop(attr,None) for attr in ['x','y']}

	with matplotlib.style.context(mplstyle):
		
		fig,ax = plt.subplots()
		obj = dict(fig=fig,ax=ax)

		for i,attr in enumerate(attributes):

			attrs = attributes[attr]

			keys = [key for key in data if all(
				(data[key][attr] == attrs[attr]) or 
				(isinstance(attrs[attr],str) and not isinstance(data[key][attr],str))
				for attr in attrs)]

			if not keys:
				continue

			opts = {}
			defaults = {'color':'viridis'}
			for attr in options:
				if attr in variables:
					if isinstance(options[attr],list):
						opts.update(
							options[attr][variables[attr].index(attrs[attr] 
							if attrs[attr] in variables[attr] 
							else data[keys[0]][attr])%len(options[attr])]
						)
				else:
					opts.update({attr:options[attr]})

			opts.update({default: opts.get(default,defaults[default]) for default in defaults})

			tmp = {
				**{values[attr]: [data[key][values[attr]] for key in keys] for attr in values},
				**{attr: attrs[attr] for attr in attrs}	
			}

			process(tmp,data)


			x = tmp[values['x']]
			y = tmp[values['y']]/factorial(attrs['t'])
			xerr = None
			yerr = None

			log('Plot',attrs,y)

			properties = dict(
			)
			
			attr = 'e'
			opts.update(dict(
				label='$%s$'%('~'.join(['%s'%(str(attrs[attr])) for attr in attrs])),
				color=getattr(plt.cm,opts.get('color'))((variables[attr].index(attrs[attr] if attrs[attr] in variables[attr] else data[keys[0]][attr]))/(len(variables['e'])))
			))
			
			ax.errorbar(x,y,xerr,yerr,**opts);

			attr = 't'
			opts = dict(
				color='grey',linestyle='',linewidth=2.5,alpha=0.8,
				marker=opts.get('marker'),markersize=opts.get('markersize'),
				zorder=100
				)
			ax.plot([min(x),max(x)],[1/factorial(attrs[attr]),1/factorial(attrs[attr])],**opts)

			attr = 't'
			opts = dict(
				color='grey',linestyle='--',linewidth=2.5,alpha=0.8,
				)
			ax.hlines(y=1/factorial(attrs[attr]),xmin=min(x)*(-4),xmax=max(x)*1.2,**opts)
			
			if i == (len(attributes)-1):
				ax.text(x=max(x)*0.78,y=1/factorial(attrs[attr])*0.68,s='$\\Large{1}/\\scriptstyle{||{\\Tau^{(t)}_{\\mathcal{C}_{\\textrm{Haar}}^{d}}}||^{2}}$',color='grey')

			for setting in settings:
				setter(obj,setting,settings[setting])


	return


def run(func,permutations,**kwargs):
	
	keywords = permute(kwargs,permutations=permutations)

	data = {}
	for key in keywords:
		
		args = {**kwargs,**keywords[key]}

		log('Model',args)

		tmp = func(**args)

		data.update({'%s_%s'%(str(key),str(attr)):tmp[attr] for attr in tmp})

	return data


def main(settings=None,**kwargs):

	default = {}
	settings = load(settings,default=default)	

	defaults = {'path':None,'boolean':{},'permutations':None,'model':{},'plot':{}}
	if settings is None:
		settings = {}
	settings.update({default:settings.get(default,defaults[default]) 
		for default in defaults})

	data = None

	if settings['boolean'].get('load'):
		data = load(settings['path'])

	if settings['boolean'].get('run'):
		data = run(func,settings['permutations'],**settings['model'])
		
	if settings['boolean'].get('dump'):
		dump(data,settings['path'])

	if settings['boolean'].get('plot'):
		plot(data,**settings['plot'])

	return


if __name__ == '__main__':
	args = sys.argv[1:]
	main(*args)
