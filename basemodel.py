import numpy as np
import scipy.spatial as spatial
from tqdm import tqdm


class AbstractTotalModel:    #(Parent class which the other class of models inherit from this class)
    """
    This class is the base model for all the different interaction models implemented. All the common functions such as
    self.get_position or self.total_movement are written in this class. This class is defined as abstract because it
    cannot be instantiated. it it designed to be a parent class.

    :param n_particles: Number of particles in the box
    :type n_particles: int
    :param dt: Increment of time for each step.
    :type dt: float
    :param radius: radius of the particles. It as constant for all the particles
    :type radius: float
    :param contact_radius: distance from which we consider a contact between two particles
    :type contact_radius: float
    :param surface: Surface of the box. We consider the box as a square, hence the length of the side is equal to the square root of the surface.
    :type surface: float
    :param n_steps: Number of steps that we consider for the total movement of the particles.
    :type n_steps: int
    """

    def __init__(self, n_particles, dt, radius, detection_radius, surface, n_steps, janus):
        self.n_particles = n_particles
        self.dt = dt
        self.radius = radius
        self.detection_radius = detection_radius
        self.side = np.sqrt(surface)
        self.n_steps = n_steps
        self.janus = janus
        self.tij = np.empty((0, 3))

    def get_position(self):
        """
        Returns the positions of all the particles.

        :return: An array of the positions of all the particles. It is of shape (n_particles, 2).`
        :rtype: np.array
        """
        return self.position_array

################################################################################################
    def get_detection_vector(self):
        """
        Returns the detection vectors of all the particles.

        :return: An array of the detection vectors of all the particles. It is of shape (n_particles, 2).`
        :rtype: np.array
        """
        return self.detection_vector_array

################################################################################################

    def get_radius(self):
        """
        Returns the radius of all the particles.

        :return: Radius of the particles. It is the same for all the particles.`
        :rtype: float
        """
        return self.radius
##################################################################################################

    def get_detection_radius(self):
        """
        Returns the radius of all the particles.

        :return: Radius of the particles. It is the same for all the particles.`
        :rtype: float
        """
        return self.detection_radius
###################################################################################################

    def get_side(self):
        """
        Returns the length of the side of the box.

        :return: Length of the side of the box.`
        :rtype: float
        """
        return self.side

    def get_janus(self):
        """
        Returns the radius of all the particles.

        :return: Radius of the particles. It is the same for all the particles.`
        :rtype: float
        """
        return self.janus

    def get_velocities(self):
        """
        Returns the velocities of all the particles.

        :return: Velocities of all the particles.
        :rtype: np.array
        """
        return self.velocities_array

    def get_velocities_norm(self):
        """
        Returns the velocities norm of all the particles.

        :return: Velocities norm of all the particles.
        :rtype: np.array
        """
        return np.linalg.norm(self.velocities_array, axis=1)

    def contact(self):
        """
        This function determines if one particle is in contact with other particles in the system.

        :return: A tuple of length 2. First, the neighbors array for each particle in contact and second, the respective
         index of each element in the main self.position_array.
        :rtype: tuple of np.arrays.
        """
        point_tree = spatial.cKDTree(self.position_array)
        eps = 10 ** (-3) * self.radius
        contact_pairs = point_tree.query_pairs(self.detection_radius+eps , output_type='ndarray')

        #contact_index = np.unique(contact_pairs)
        AB=self.position_array[contact_pairs[:,1]]- self.position_array[contact_pairs[:, 0]]
        U_A=self.detection_vector_array[contact_pairs[:,0]]
        U_B=self.detection_vector_array[contact_pairs[:,1]]
        #index1=contact_pair[:,0][np.einsum('ij,ij->i', AB, U_A)>=0]
        #index2=contact_pair[:,0][np.einsum('ij,ij->i', AB, U_B)<=0]
        #contact=np.concatenate((index1, index2))
        indices=np.logical_and(np.einsum('ij,ij->i', AB, U_A)>=0, np.einsum('ij,ij->i', AB, U_B)<=0 )


        contact_pairs2=contact_pairs[indices]
        contact_index2 = np.unique(contact_pairs2)




        return contact_pairs2, contact_index2  # Find all pairs of points in self whose distance is at most r.
        




    def total_movement(self):
        """
        This function iterates all the Brownian motion throughout the n_steps and returns the tij array to be analyzed

        :return: Returns the tij array. It represents all the interactions between particles i and j at time t
        :rtype: np.array
        """
        for step in tqdm(range(self.n_steps)):
            self.iter_movement(step)

        return np.array(self.tij)

    def creation_tij(self, step, pairs):
        """
        This function extend the tij array of all the interactions between particle i and j at time step*dt.
        This function principal role is to find the array of neighbors in a 2 * self.radius radius.

        :param step: step of the iteration. It ranges from 0 to self.n_steps-1
        :type step: int
        :param pairs: array of all the pairs of particles.
        :type pairs: np.array
        """
        # For Janus particles we only consider the contact if the particles face each other.
        if self.janus:
            contact_i_array = pairs[:, 0]
            contact_j_array = pairs[:, 1]
            velocity_i_array, velocity_j_array = self.velocities_array[contact_i_array], \
                self.velocities_array[contact_j_array]
            # Calculation of the particles that are face to face.
            truth_array = np.einsum('ij,ij->i', velocity_j_array, velocity_i_array) <= 0
            pairs = pairs[truth_array]

        time_pairs = np.append(step * self.dt * np.ones((pairs.shape[0], 1)), pairs, axis=1)
        self.tij = np.append(self.tij, time_pairs, axis=0)


class AbstractBwsAbpModel(AbstractTotalModel):
    """
    This class is the common model for both the ballistic with stop and the active brownian particle models implemented.
    The common functions for these two models that cannot be implemented in the AbstractTotalModel are written here.
    This class is defined as abstract because it cannot be instantiated. it it designed to be a parent class.

    :param v: Speed of the particle
    :type v: float or int
    :param dt: Increment of time for each step. Constant * dt is the variance of the normal distribution that we use to calculate the increment of all the positions at each step.
    :type dt: float or int
    :param radius: radius of the particles. It as constant for all the particles
    :type radius: float or int
    :param contact_radius: distance from which we consider a contact between two particles
    :type contact_radius: float
    :param n_particles: Number of particles in the box
    :type n_particles: int
    :param surface: Surface of the box. We consider the box as a square, hence the length of the side is equal to the square root of the surface.
    :type surface: float or int
    :param n_steps: Number of steps that we consider for the total movement of the particles.
    :type n_steps: int
    :param noise: Adds noise to the angle of the particle
    :type noise: float
    :param stop: stop the particle each time it encounters another one.

    """
    def __init__(self, v, n_particles, dt, radius, detection_radius, surface, n_steps, janus, stop):
        self.v = v
        self.stop = stop
        super().__init__(n_particles, dt, radius, detection_radius, surface, n_steps, janus)
        self.position_array = self.initial_positions()
        self.velocities_array = np.zeros((self.n_particles, 2))
        self.random_velocities(np.arange(0, n_particles, dtype=int))

    def initial_positions(self):
        """
        This functions returns the initial position array of the particles. They are chosen randomly from an uniform
        distribution: 0 + self.radius <= x, y <= self.side - self.radius. One of the complications is that we do not
        want overlapping of the particles.

        :return: array of the initial positions of shape (self.n_particles, 2)
        :rtype: np.array
        """
        initial_positions_array = np.random.rand(self.n_particles, 2) * (self.side - 2 * self.radius) + self.radius
        point_tree = spatial.cKDTree(initial_positions_array)
        neighbors = point_tree.query_ball_point(initial_positions_array, 2 * self.radius)

        for i, elt in enumerate(neighbors):

            if len(elt) > 1:
                condition = False

                while not condition:
                    new_position = np.random.rand(2) * (self.side - 2 * self.radius) + self.radius   #r<x,y<side-r
                    new_point_tree = spatial.cKDTree(np.delete(initial_positions_array, i, axis=0))
                    neighbors = new_point_tree.query_ball_point(new_position, 2 * self.radius + 0.1)

                    if len(neighbors) == 0:
                        initial_positions_array[i] = new_position
                        condition = True

        return initial_positions_array

    def random_velocities(self, index):
        """
        This function updates the velocities of the particles in index. The x and y components are chosen randomly but
        are subject to one constraint, the norm of all the new velocities is equal to self.v.

        :param index: index of the velocities to update
        :type index: np.array
        """
        index_size = index.size
        self.velocities_array[index, 0] = (np.random.rand(index_size, 1).flatten() - 0.5) * 2 * self.v   #vx
        alg = np.random.choice([-1, 1], index_size)
        self.velocities_array[index, 1] = alg * np.sqrt(self.v ** 2 - self.velocities_array[index, 0] ** 2)  #vy


    def border(self):
        """
        This function sets to 0 the velocity of a particle that goes out of the box. The particle doesn't move out of
        the border until the randomly chosen perpendicular velocity to the border makes the particle reenter the box.
        The particle can glide on the border.
        """
        index_min = np.where(self.position_array - self.radius <= 0) #gives the the index of row and columns when  position-r<=0
        index_max = np.where(self.position_array + self.radius >= self.side) # gives the the index of row and columns when  position+r >=side
        self.random_velocities(index_min[0])  #then we calculate the random velocities for those particels that are over the edges  (x-r, y-r)<0
        self.random_velocities(index_max[0])   ##then we calculate the random velocities for those particels that are over the edges  (x+r, y+r)>side
        mask_min = np.logical_and(self.position_array - self.radius <= 0, self.velocities_array <= 0) 
#these search for the indices of particles that simultaneusly satsfies 2 condition (x-r , y-r<0 and velocities<0) simultaneously
        mask_max = np.logical_and(self.position_array + self.radius >= self.side, self.velocities_array >= 0)
        mask_min_size = np.count_nonzero(mask_min)
        mask_max_size = np.count_nonzero(mask_max)

        if mask_min_size > 0:
            alg = np.random.choice([-1, 1], mask_min_size)
            self.velocities_array[mask_min] = 0
            self.velocities_array[mask_min[:, [1, 0]]] = alg * self.v

        if mask_max_size > 0:
            alg = np.random.choice([-1, 1], mask_max_size)
            self.velocities_array[mask_max] = 0
            self.velocities_array[mask_max[:, [1, 0]]] = alg * self.v

    def update_velocities_stop(self, contact_pairs, contact_index):
        """
        This function updates the velocities of the particles in the case where self.stop is set to True. If two
        particles are in contact they stop. They choose a new random velocity vector. If the two vectors are not facing
        opposite directions, then they stay in contact. Otherwise they move until their next contact.

        :param contact_pairs: array of pairs of contacts. shape = (n_contact, 2)
        :type contact_pairs: np.array
        :param contact_index: array of indexes of all the particles in contact
        :type contact_index: np.array
        """
        # Calculation of random velocities for the particles in contact
        self.random_velocities(contact_index)

        # Computation of all the necessary arrays for the call of the projection function
        i_index_array, j_index_array = contact_pairs[:, 0], contact_pairs[:, 1]
        velocity_i_array, velocity_j_array = self.velocities_array[i_index_array], self.velocities_array[
            j_index_array]
        position_i_array, position_j_array = self.position_array[i_index_array], self.position_array[j_index_array]
        centers_array = position_j_array - position_i_array
        distance_array = np.linalg.norm(centers_array, axis=-1).reshape(-1, 1)
        self.position_array[i_index_array] = position_i_array - (
                2 * self.radius - distance_array) * centers_array / distance_array

        # Call of the projection function -> if velocity vectors face each other, the particles stay in contact.
        truth_array = self.projection(centers_array, velocity_i_array, velocity_j_array)
        velocity_pairs_null = np.where(np.logical_not(truth_array))[0]
        velocity_null_index = contact_pairs[velocity_pairs_null].ravel()
        self.velocities_array[velocity_null_index] = 0

    def iter_movement(self, step, animation=False):
        """
        This function updates the self.position_array at time step*dt. The function takes the position of the array
        (x, y) and adds an infinitesimal step (dx, dy). (dx, dy) is equal to (vx*dt, vy*dt). (vx, vy) is updated at
        each step in function of what model is used.  Hence the new position is (x+dx, y+dy). The borders of the box are
        also considered with the self.border() function.

        :param step: step of the iteration. It ranges from 0 to self.n_steps-1
        :type step: int
        :param animation: This parameter is set to False by default. This means that the creation_tij array is stored and can be analyzed. It is set to true only when the animation is run. As the animation can run indefinitely, too much data can be stored
        :type animation: bool, optional
        """
        contact_pairs, contact_index = self.contact() 
        if self.stop:  #if this is true it means that it is balistic with stop

            if contact_index.size > 0:
                self.update_velocities(contact_pairs, contact_index) #update_velocities is defined in balistic with stop code and inside it we 

        # if self.stop is set top stop, the contact_pairs don't matter for the update_velocities array.
        else:
            self.update_velocities(None, None)

        self.border()
        self.position_array = self.position_array + self.velocities_array * self.dt

        if not animation:
            self.creation_tij(step, contact_pairs)

    @staticmethod
    def projection(centers_array, velocity_i_array, velocity_j_array):
        """
        This function returns True if velocity_i and velocity_j face opposite directions considering the vector that
        links the center of particle i and particle j.

        :param centers_array: all the vectors that link the centers of particle i and particle j.
        :type centers_array: np.array
        :param velocity_i_array: vector of the velocity of particle i
        :type velocity_i_array: np.array
        :param velocity_j_array: array of vectors of the velocity of particle j (neighbors of i)
        :type velocity_j_array: np.array
        :return: True if the two velocity vectors face opposite directions
        :rtype: bool
        """
        # np.einsum('ij,ij->i', arr1, arr2) is the dot product between each rows of arr1 and arr2
        a = np.einsum('ij,ij->i', centers_array, velocity_i_array)
        b = np.einsum('ij,ij->i', centers_array, velocity_j_array)
        return np.logical_and(a <= 0, b >= 0)

