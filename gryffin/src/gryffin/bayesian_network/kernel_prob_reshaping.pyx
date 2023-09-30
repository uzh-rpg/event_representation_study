#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import cython

from cython.parallel import prange

import  numpy as np 
cimport numpy as np 

from libc.math cimport exp, sqrt

#========================================================================

cdef class KernelReshaper:

	cdef int num_samples, num_obs, num_kernels, num_descriptors
	cdef np.ndarray np_recomputed_probs
	cdef np.ndarray np_all_distances

	def __init__(self):

		pass


	@cython.cdivision(True)
	@cython.boundscheck(False)
	cdef double [:, :, :] _reshape_probs(self, double [:, :, :] cat_probs, double [:, :] descriptors):

		cdef double [:, :, :] recomputed_probs = self.np_recomputed_probs
		cdef double [:, :, :] all_distances    = self.np_all_distances

		cdef double ds2, dyi, sum_distances
		cdef double averaged_descriptor

		cdef int sample_index, obs_index, target_cat_index, desc_index, kernel_index

		for sample_index in prange(self.num_samples, nogil = True):

			for obs_index in range(self.num_obs):

				for desc_index in range(self.num_descriptors):
	
					averaged_descriptor = 0.
					for kernel_index in range(self.num_kernels):
						averaged_descriptor = cat_probs[sample_index, obs_index, kernel_index] * descriptors[kernel_index, desc_index] + averaged_descriptor	


				for target_cat_index in range(self.num_kernels):

					ds2 = 0.
					for desc_index in range(self.num_descriptors):
					
						dyi = self.num_kernels * (descriptors[target_cat_index, desc_index] - averaged_descriptor)
						ds2 = ds2 + dyi * dyi

					all_distances[sample_index, obs_index, target_cat_index] = sqrt(ds2 / self.num_descriptors)

				# got all distances, compute probs from distances
				sum_distances = 0.
				for kernel_index in range(self.num_kernels):
					sum_distances = sum_distances + exp( - all_distances[sample_index, obs_index, kernel_index])
				
				for kernel_index in range(self.num_kernels):
					recomputed_probs[sample_index, obs_index, kernel_index] = exp( - all_distances[sample_index, obs_index, kernel_index]) / sum_distances

		return recomputed_probs



	cpdef reshape_probs(self, np.ndarray cat_probs, np.ndarray descriptors):

		self.num_samples     = cat_probs.shape[0]
		self.num_obs         = cat_probs.shape[1]
		self.num_kernels     = cat_probs.shape[2]
		self.num_descriptors = descriptors.shape[1]

		self.np_recomputed_probs = np.zeros((self.num_samples, self.num_obs, self.num_kernels))
		self.np_all_distances    = np.zeros((self.num_samples, self.num_obs, self.num_kernels))

		cdef double [:, :, :] cat_probs_memview   = cat_probs
		cdef double [:, :]    descriptors_memview = descriptors

		reshaped_probs = self._reshape_probs(cat_probs_memview, descriptors_memview)		
		return np.array(reshaped_probs)

