import numpy as np
from numba import njit
from actuations.actuation import _internal_to_external_load
from actuations.muscles.muscle import (
    local_strain,
    MuscleForce,
)


class TransverseMuscle(MuscleForce):
    def __init__(
            self,
            muscle_radius_ratio,
            max_force,
            strain_weighted_function=None,
            strain_rate_weighted=None
    ):
        MuscleForce.__init__(self, muscle_radius_ratio.shape[1])
        self.muscle_radius_ratio = np.zeros((3, self.n_elements))
        self.muscle_radius_ratio[:2, :] = muscle_radius_ratio.copy()
        self.max_force = -max_force.copy()
        self.strain_weighted_function = strain_weighted_function
        self.strain_rate_weighted = False
        if strain_rate_weighted is not None:
            self.strain_rate_weighted = True
            self.strain_rate_weighted_function = strain_rate_weighted['function']
            self.sigma_rate = strain_rate_weighted['sigma_rate']
            self.kappa_rate = strain_rate_weighted['kappa_rate']
        self.count = 0

    def __call__(self, system, count_flag=True):

        magnitude_for_force = self.get_activation() * self.max_force
        transverse_muscle_function(
            magnitude_for_force,
            system.director_collection, system.kappa, system.tangents,
            system.rest_lengths, system.rest_voronoi_lengths,
            system.dilatation, system.voronoi_dilatation,
            self.internal_forces, self.internal_couples,
            self.external_forces, self.external_couples
        )
        if count_flag:
            self.count += 1


    @staticmethod
    @njit(cache=True)
    def longitudinal_muscle_stretch_mag(off_center_displacement, strain, curvature):
        nu = local_strain(
            off_center_displacement, strain, curvature
        )
        blocksize = nu.shape[1]
        stretch_mag = np.ones(blocksize)
        for i in range(blocksize):
            stretch_mag[i] = np.linalg.norm(nu[:, i])
        return stretch_mag

    @staticmethod
    @njit(cache=True)
    def longitudinal_muscle_stretch_rate_mag(off_center_displacement, sigma_rate, curvature_rate):
        nu_rate = local_strain(
            off_center_displacement, sigma_rate, curvature_rate
        )
        blocksize = nu_rate.shape[1]
        stretch_mag = np.ones(blocksize)
        for i in range(blocksize):
            stretch_mag[i] = nu_rate[2, i]
        return stretch_mag



@njit(cache=True)
def transverse_muscle_function(
        magnitude_for_force,
        director_collection, kappa, tangents,
        rest_lengths, rest_voronoi_lengths,
        dilatation, voronoi_dilatation,
        internal_forces, internal_couples,
        external_forces, external_couples
):
    internal_forces[2, :] = magnitude_for_force.copy()

    _internal_to_external_load(
        director_collection, kappa, tangents,
        rest_lengths, rest_voronoi_lengths,
        dilatation, voronoi_dilatation,
        internal_forces, internal_couples,
        external_forces, external_couples
    )
