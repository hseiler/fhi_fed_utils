import numpy as np
from decorator import decorator
from scipy.interpolate import interp1d

#Convlution, usually with gaussian Kernel.
def convolve(arr,kernel):
	"""
	Convolution of array with kernel.
	"""
	#logger.debug("Convolving...")
	npts = min(len(arr), len(kernel))
	pad	 = np.ones(npts)
	tmp	 = np.concatenate((pad*arr[0], arr, pad*arr[-1]))
	norm = np.sum(kernel)
	out	 = np.convolve(tmp, kernel, mode='valid')
	noff = int((len(out) - npts)/2)
	return out[noff:noff+npts]/norm
	
def gauss_kernel(x, t0, s):
	"""
	Gaussian convolution kernel.

	Parameters
	----------
	x : array-like
		Independant variable
	t0 : array-like
		t0 offset
	irf : array-like
		Irf gaussian width (sigma)
	"""
	midp = 0.5*(np.max(x)+np.min(x))
	return (0.398942/s)*np.exp(-(x-(midp+t0))**2/(2*s**2))


def resamp_weights(t0,t1):
	"""
	Given an unevenly spaced time array t0, you want to make an evenly spaced array t1 with smaller steps.
	The new data points should have a weight that is proportional to the number of 'real' data points they're made up of .
	If you don't, your regridded data will artificially weight the fit to places where 'real' data is sparse.
	
	i.e. take unvenely spaced points [0,1,2,3,4,5,10,15,20,25] and map it to np.arange(25). 
	First 6 points in new array (0,1,2,3,4,5) should have weight 1. Next 20 points (6,7,8,...,24,25) should have weight 1/5.
	Inputs:
	t0 - (1,N) array : initial unevenly spaced array
	t1 - (1,M) array : final, evenly spaced array
	
	returns:
	weights - (1,M) array : array of weights to be used in any fitting algorithm. Doesn't take into account instrumental error.
	"""
	weights = np.ones(t1.shape)
	dt0 = np.diff(t0)
	dt1 = np.diff(t1)
	for i in range(len(t1)-1):
		tmp_sum=0
		j = np.argmin(np.abs(t1[i]-t0))
		k = 0
		while tmp_sum < dt0[j]:
			try:
				tmp_sum += dt1[i+k]
				weights[i] = dt1[i+k]/dt0[j]
			except IndexError:
				tmp_sum += dt1[i]
				weights[i] = dt1[i]/dt0[j]
				k += 1
		weights[-1] = weights[-2]
	return weights


#For irregular grids we made a decorator
def regrid(idx):
	"""
	Decorator factory to compute a model on a constant grid, then interpolate.

	This is to be used for reconvolution fits when the independant axis isn't
	evently spaced. This function returns a decorator. You should call the
	result of this function with the model to regrid. The constant grid

	Parameters
	----------
	idx : int
		Index of variable to regrid in client function.

	Returns
	-------
	regridder : decorator

	Example
	-------
	```
	def model(x, amp, tau, t0, sig):
		# Convolution assumes constant grid spacing.
		return convolve(step(x)*exp_decay(x, amp, tau), gauss_kernel(x, t0, sig))

	deco = regrid(1)
	regridded = deco(model)
	# Or, on a single line
	regridded = regrid(1)(model) # compute on first axis
	# Or, during definition
	@regrid(1)
	def model(x, *args):
		...
	```
	"""
	#logger.debug("Applying 'regrid' decorator")
	def _regrid(func, *args, **kw):
		#logger.debug("Regridding func {}".format(func.__name__))
		x = args[idx]
		#print("regridding...")
		mn, mx = np.min(x), np.max(x)
		extension=1
		margin = (mx-mn)*extension
		dx = np.abs(np.min(x[1:]-x[:-1]))
		#print("regrid args", args)
		#print("regrid kw", kw)
		#print("regrid func", func)
		grid = np.arange(mn-margin, mx+margin+dx, dx)
		args = list(args)
		args[idx] = grid
		y = func(*args, **kw)
		#print("y", y)
		intrp = interp1d(grid, y, kind=3, copy=False, assume_sorted=True)
		return intrp(x)
	return decorator(_regrid)

#Some model functions
def step(x):
	"""Heaviside step function."""
	step = np.ones_like(x, dtype='float')
	step[x<0] = 0
	step[x==0] = 0.5
	return step

def exp_decay(t, a, tau):
	return step(t)*a*np.exp(-t/tau)

def exp_rise(t, a, tau):
	return step(t)*a*(1- np.exp(-t/tau))
	
def exp_conv(t,a,tau,t0,irf):
	return convolve(
		exp_decay(t,a,tau),
		gauss_kernel(t,t0,irf))

def biexp_decay(t, a0, a1, tau0, tau1):
	return exp_decay(t, a0, tau0)+exp_decay(t, a1, tau1)


def biexp_rise(t, a0, a1, tau0, tau1):
	return exp_rise(t, a0, tau0) + exp_rise(t, a1, tau1)
	
def biexp_conv(t, a0, a1, tau0, tau1, t0, irf):
	return convolve(
		biexp_decay(t, a0, a1, tau0, tau1),
		gauss_kernel(t, t0, irf))
		
def biexp_rise_conv(t, a0, a1, tau0, tau1, t0, irf):
	return convolve(
		biexp_rise(t, a0, a1, tau0, tau1),
		gauss_kernel(t, t0, irf))
		
def n_exp_decay(t,amp_arr,t_arr,n_exp):
	y = np.zeros(len(t))
	for i in range(n_exp):
		y += exp_decay(t,amp_arr[i],t_arr[i])
	return y