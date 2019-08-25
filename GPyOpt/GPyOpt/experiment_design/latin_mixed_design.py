import numpy as np

from ..core.errors import InvalidConfigError

from .base import ExperimentDesign
from .random_design import RandomDesign
from pyDOE import lhs, doe_lhs

class LatinMixedDesign(ExperimentDesign):
    """
    Latin experiment design modified to work with non-continuous variables.
    Neglect bandit variables (simply do what LatinDesign does).
    """
    def __init__(self, space):
        if space.has_constraints():
            raise InvalidConfigError('Sampling with constraints is not allowed by latin design')
        super(LatinMixedDesign, self).__init__(space)
        
    def get_samples(self, init_points_count, iterations = None, verbose = False):

        samples = np.empty((init_points_count, self.space.dimensionality))
        if iterations is None:
            iterations = min(30, 2 * samples.shape[0])

        def _lhs_discrete(dimensions, samples=None, iterations=5):
            H = None
            retries = 10
            retry = 0
            while H is None and retry < retries: # stop if not at par with expectations but the retries have been exhausted to avoid an infinite loop
                maxdist = 0
                for iteration in range(iterations):
                    if verbose is True:
                        print('[LHD-mv] Iteration:', iteration+1, 'of', iterations, '| Retry:', retry, 'of (max)', retries)
                    Hcandidate = _lhs_noRandom(dimensions, samples)              
                    d = doe_lhs._pdist(Hcandidate)
                    if maxdist<np.min(d):
                        test_minimum_representation = _check_representation(Hcandidate, discrete_values, display_check = verbose)
                        if test_minimum_representation is True:
                            maxdist = np.min(d)
                            H = Hcandidate.copy()
                retry += 1
            H = _map_to_discrete_values(H, discrete_values)
            return H
        
        def _lhs_noRandom(dimensions, samples=None):
            interval_starting_values = np.linspace(0, 1, samples, endpoint = False)
            H = np.zeros((samples, dimensions))
            for j in range(dimensions):
                order = np.random.permutation(range(samples))
                H[:, j] = interval_starting_values[order]            
            return H

        def _map_to_discrete_values(H, discrete_values):
            mappedH = np.full_like(H, 0)
            dimensions = H.shape[1]
            for dimension in range(dimensions):
                numLevels = len(discrete_values[dimension])
                levelindices = (H[:, dimension] * numLevels).astype(int)
                for i, index in enumerate(levelindices):
                    mappedH[i, dimension] = discrete_values[dimension][index]
            return mappedH
        
        def _check_representation(H, discrete_values, minimum = None, display_check = False):
            H = _map_to_discrete_values(H, discrete_values)
            samples = H.shape[0]
            dimensions = H.shape[1]
            test = True
            given_minimum = minimum
            for dimension in range(dimensions):
                levels = len(discrete_values[dimension])
                unique, count = np.unique(H[:, dimension], return_counts = True)
                if display_check is True:
                    unique_count_dict = {int(unique):str(count) + ' times' for unique, count in zip(unique, count)}
                    print('Discrete Dimension #{}'.format(dimension))
                    print('Dimension levels:', discrete_values[dimension])
                    print('Samples taken:', H[:, dimension])
                    print('Instance counts:', unique_count_dict, '\n')
                if samples < levels:
                    continue #skip checking since it is futile
                if given_minimum == None:
                    minimum = np.maximum(np.floor(0.8 * samples / levels), 1)
                if (min(count) < minimum) or not np.all(np.isin(discrete_values[dimension], unique)):
                    test = False
                    break
            return test
        
        if self.space.has_discrete():
            discrete_dimensions = self.space.get_discrete_dims()
            discrete_values = self.space.get_discrete_values()
            discrete_design = _lhs_discrete(len(discrete_dimensions), init_points_count, iterations)
            samples[:, discrete_dimensions] = discrete_design
            
        if self.space.has_continuous():
            continuous_dimensions = self.space.get_continuous_dims()
            continuous_bounds = self.space.get_continuous_bounds()
            lower_bound = np.asarray(continuous_bounds)[:,0].reshape(1, len(continuous_bounds))
            upper_bound = np.asarray(continuous_bounds)[:,1].reshape(1, len(continuous_bounds))
            diff = upper_bound - lower_bound
            X_design_aux_c = lhs(len(continuous_dimensions), init_points_count, criterion = 'maximin', iterations = iterations)
            I = np.ones((X_design_aux_c.shape[0], 1))
            continuous_design = np.dot(I, lower_bound) + X_design_aux_c * np.dot(I, diff)
            samples[:, continuous_dimensions] = continuous_design     
            
        return samples