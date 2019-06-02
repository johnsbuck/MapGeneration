"""Poisson Disc Sampling Generator

This script uses a generator to create random points with a minimum distance of r from all other points
within a defined space.

This is often called Poisson Disc Sampling or blue noise.

The script used is Bridson's Algorithm to generate the points.

"""


import numpy as np
from Utilities import metric


class PoissonDiscSampleGenerator(object):
    """Poisson Disc Sample Generator

    A generator used to create a poisson disc sample with given user parameters.

    """

    def __init__(self, radius=5, k=30, extent=[100, 100], seed=None):
        """ PoissonDiscSampleGenerator Constructor

        Initializes the Poisson Disc Sample Generator.

        Arguments:
            radius (float): Radial distance for each random sample. (Default: 1)
            metric (function): Function used to calculate distance. (Default: distance.euclidean)
            k (int): Number of points used to generate a sample.
            extent (list<int>): The maximum length of each dimension.
            seed (int): The seed used when generating the random sample. (Optional) (Default: None)
        """
        # --------------------------------
        # User Parameters
        # --------------------------------

        # Defines self._radius
        self.radius = radius
        # Defines self._k
        self.k = k
        # Defines both self._extent and self._dim
        self.extent = np.array(extent)
        # Defines self._seed
        self.seed = seed

        self._metric = metric.euclidean
        self._changes = self._create_neighbor_distances()

        # --------------------------------
        # Grid Parameters
        # --------------------------------
        self._cell_length = self._radius / np.sqrt(self._dim)
        self._grid_shape = np.array([int(np.ceil(
            self._extent[i] / self._cell_length))
            for i in range(self._dim)], dtype=int)

        # Define Grid
        self._grid = np.empty(shape=self._grid_shape, dtype=int)
        self._grid.fill(-1)

        # --------------------------------
        # Sample List
        # --------------------------------
        self._samples = []

    def __call__(self):
        """Calls the generate function.

        Returns:
            (np.ndarray)
        """
        return self.generate()

    def generate(self):
        """Generates the samples with the given user parameters.

        Returns:
            (np.ndarray) Returns an array of samples of shape (n_samples, dim) where dim
            is the length of the extent given.
        """
        # --------------------------------
        # Initializing Variables
        # --------------------------------

        # Set NumPy Random Seed
        np.random.seed(self._seed)

        # Create Active List
        active = []

        # Clear previously generated examples
        if len(self._samples) > 0:
            self._clear_previous_samples()

        # --------------------------------
        # Begin Generating Samples
        # --------------------------------

        # Create the first sample
        self._samples.append(np.random.uniform(low=np.zeros(shape=(self._dim,)),
                                               high=self._extent, size=(self._dim,)))
        active.append(self._samples[0])
        self._grid[self._get_grid_coord(self._samples[0])] = 0

        while active:
            # Choose Random Active Sample
            idx = np.random.choice(len(active))

            # Make new point & confirm it is valid
            new_point = self._make_point(active[idx])
            if new_point is None:
                active.pop(idx)
            else:
                # Add sample to listings and store in grid for neighboring locations.
                self._samples.append(new_point)
                active.append(new_point)
                self._grid[self._get_grid_coord(new_point)] = len(self._samples) - 1

        # Return samples as numpy array
        self._samples = np.array(self._samples)
        return self._samples

    @property
    def radius(self):
        """The minimum distance between any two points.

        Returns:
            (float)
        """
        return self._radius

    @property
    def extent(self):
        """The measurements of the plane where the sample will be produced (length, width, height, depth, etc.)

        Returns:
            (np.ndarray)
        """
        return np.array(self._extent)

    @property
    def k(self):
        """The number of attempts each active point to make a new point.

        Returns:
            (int)
        """
        return self._k

    @property
    def metric(self):
        """The distance function used to measure the distance between two points.

        Returns:
            (function)
        """
        return self._metric

    @property
    def seed(self):
        """The random seed used for generating samples.

        Returns:
            (int) Returns None if no seed is set, otherwise a user-defined int.
        """
        return self._seed

    @property
    def dimension_size(self):
        """The length of the extent and all of the generated sample point.

        Returns:
            (int)
        """
        return self._dim

    @property
    def samples(self):
        """The samples generated by the generator. May not match specs if parameters were re-adjusted after use.

        Returns:
            (np.ndarray)
        """
        if len(self._samples) == 0:
            return np.empty((0, 2))
        return self._samples

    @radius.setter
    def radius(self, radius):
        """Setter for radius

        Args:
            radius (float): Minimum distance between two points.
        """
        if radius <= 0:
            raise ValueError("Radius must be a number that is greater than 0.")
        self._radius = radius

    @extent.setter
    def extent(self, extent):
        """Setter for extent

        Args:
            extent (list<float>): The dimension lengths.
        """
        if len(extent) < 2:
            raise ValueError("Extent must have a length of at least 2.")
        extent = np.array(extent, dtype=float)
        if np.any(extent <= 0):
            raise ValueError("All extents must be greater than 0.")
        self._extent = extent
        self._dim = len(extent)

    @k.setter
    def k(self, k):
        """Setter for k

        Args:
            k (int): Number of attempts for each active point to generate a new point.
        """
        if k <= 0:
            raise ValueError("K must be greater than 0.")
        if int(k) != k:
            raise ValueError("K must be an integer.")
        self._k = k

    @seed.setter
    def seed(self, seed):
        if seed is not None:
            if type(seed) is not int:
                raise ValueError("Seed must be integer.")
            if seed < 0:
                raise ValueError("Seed must be non-negative.")
        self._seed = seed

    def _clear_previous_samples(self):
        """Clears grid and samples for generating new samples.
        """
        del self._grid
        del self._samples

        # --------------------------------
        # Grid Parameters
        # --------------------------------
        self._cell_length = self._radius / np.sqrt(self._dim)
        self._grid_shape = np.array([int(np.ceil(
            self._extent[i] / self._cell_length))
            for i in range(self._dim)], dtype=int)

        # Define Grid
        self._grid = np.empty(shape=self._grid_shape, dtype=int)
        self._grid.fill(-1)

        # --------------------------------
        # Sample List
        # --------------------------------
        self._samples = []

    def _get_grid_coord(self, point):
        """Returns the grid coordinate of the point.

        Args:
            point (np.ndarray): An array of size (number_of_dimensions,).

        Returns:
            The grid coordinate that the point is separated in.
        """
        return tuple([int(point[i] / self._cell_length) for i in range(self._dim)])

    def _make_point(self, active_point):
        """ Attempts to make a random point in proximity of active_point.

        Attempts to make a random point around the active_point k times.
        If the new point is too close to another point, it will discard and try.
        If it fails k times, the function returns None.

        Args:
            active_point (np.ndarray): An array of size (number_of_dimensions,).

        Returns:
            (np.ndarray). Returns an array of size (number_of_dimensions,) if succeeds. Otherwise, returns None.
        """
        # --------------------------------
        # Create Random Parameters
        # --------------------------------
        for _ in range(self._k):
            # Defines radial distance from active_point.
            rho = np.random.uniform(self._radius, 2 * self._radius)
            # Defines angle from active_point. Requires multiple angles for higher dimensional planes.
            theta = [np.random.uniform(0, 2 * np.pi) for _ in range(self._dim - 1)]

            # --------------------------------
            # Create New Point
            # --------------------------------

            # Create a 2D point using first theta angle.
            new_point = [active_point[0] + rho * np.cos(theta[0]), active_point[1] + rho * np.sin(theta[0])]
            # Generate more components of the coordinate for higher dimensional planes.
            new_point.extend([active_point[i] + rho * np.sin(theta[i-1]) for i in range(2, active_point.shape[0])])
            new_point = np.array(new_point)

            # Confirm point is valid
            if self._valid_point(new_point):
                return new_point
        return None

    def _valid_point(self, point):
        """Confirms that a point is valid.

        If a point is too close to another point or is outside of bounds, it will fail.
        Otherwise, it will succeed.

        Args:
            point (np.ndarray): An array of size (number_of_dimensions,).

        Returns:
            (bool) If succeeds, returns True. Otherwise, returns False.
        """
        # --------------------------------
        # Check Bounds
        # --------------------------------
        # Get grid point and confirm it is within range
        coord = self._get_grid_coord(point)
        if np.logical_or(np.any(point < 0), np.any(point >= self._extent)):
            return False

        # --------------------------------
        # Check Distance of Neighbors
        # --------------------------------
        for idx in self._get_neighbors(coord):
            # No points in grid cell
            if self._grid[idx] == -1:
                continue

            # Obtains point in grid cell and confirms its distance is less than the radius.
            near_point = self._samples[self._grid[idx]]
            if metric.euclidean(near_point, point) < self._radius:
                return False
        return True

    def _get_neighbors(self, coord):
        """Obtains neighboring cells that are surrounding a given cell coordinate.

        Args:
            coord (tuple): A coordinate whose length is equal to the number of dimensions.

        Returns:
            (list<tuple>). A list of neighboring cells.
        """
        neighbors = []
        for change in self._changes:
            neighbor_coord = np.array(coord) + change
            if np.logical_or(np.any(neighbor_coord < 0), np.any(neighbor_coord >= self._grid_shape)):
                continue
            neighbors.append(tuple(neighbor_coord))
        return neighbors

    def _create_neighbor_distances(self):
        """Creates distance vectors for calculating all neighbors from a given point.

        A neighbor can be only so far away from the original point and is dependent on the number of dimensions.
        We calculate every possible coordinate on the grid that the neighbor can be part of relative to a point
        and store that information for later use.

        Returns:
            (np.ndarray<int>) The array is of shape (number_of_vectors, number_of_dimensions).
        """
        # --------------------------------
        # Create Directions from Point
        # --------------------------------
        diff = [[0 for _ in range(self._dim)]]
        curr = diff[0][:]
        for i in range(self._dim):
            # Each diff is a unit vector, only having one value at +1 or -1 and all others at 0.
            curr[i] = 1
            diff.append(curr[:])
            curr[i] = -1
            diff.append(curr[:])
            curr[i] = 0
        # Remove initial blank unit vector with all values at 0.
        diff.pop(0)
        del curr

        # --------------------------------
        # Breadth First Search
        # --------------------------------
        distances = []
        queue = [[0 for _ in range(self._dim)]]

        while queue:
            # Get latest distance
            curr = queue.pop()

            # The distance from any possible point should be less than or equal to the number of dimensions.
            # This can be shown using basic calculations.
            if self._metric(np.array(curr), np.zeros(shape=(len(curr),))) >= 2 * np.sqrt(self._dim) or \
                    np.any(np.abs(np.array(curr)) > self._extent / 2) or curr in distances:
                continue

            # Calculate all distances from child and add to queue
            queue.extend([list(np.array(curr) + np.array(diff[i])) for i in range(len(diff))])

            # Add current distance to distances
            distances.append(curr)

        # Return all possible neighbor distances
        return np.array(distances, dtype=int)
