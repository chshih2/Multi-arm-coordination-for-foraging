from __future__ import division
from __future__ import print_function

import numpy as np
from numba import njit

from elastica._linalg import _batch_cross
from elastica._calculus import quadrature_kernel
from elastica.external_forces import inplace_addition

from actuations.actuation import ContinuousActuation, ApplyActuation

class Muscle(ContinuousActuation):
    def __init__(self, n_elements):
        ContinuousActuation.__init__(self, n_elements)
        self.n_elements = n_elements
        self.activation = np.zeros(self.n_elements-1)
        self.s = np.linspace(0, 1, self.n_elements+1)[1:-1]

    def set_activation(self, activation):
        self.activation[:] = activation.copy()
        return
    
    def get_activation(self):
        raise NotImplementedError

class MuscleForce(Muscle):
    def __init__(self, n_elements):
        Muscle.__init__(self, n_elements)
        self.distributed_activation = np.zeros(self.n_elements)

    def get_activation(self):
        redistribute_activation(self.activation, self.distributed_activation)
        return self.distributed_activation

@njit(cache=True)
def redistribute_activation(activation, distributed_activation):
    distributed_activation[0] = activation[0]/2
    distributed_activation[-1] = activation[-1]/2
    distributed_activation[1:-1] = (activation[1:] + activation[:-1])/2
    return

class MuscleCouple(Muscle):
    def __init__(self, n_elements):
        Muscle.__init__(self, n_elements)
    

    def get_activation(self):
        return self.activation

class MuscleFibers(object):
    def __init__(self, n_elements, control_numbers):
        self.controls = np.zeros(control_numbers)
        # self.G_activations = np.zeros((control_numbers, n_elements))
        self.G_internal_forces = np.zeros((control_numbers, 3, n_elements))
        self.G_internal_couples = np.zeros((control_numbers, 3, n_elements-1))
        self.G_external_forces = np.zeros((control_numbers, 3, n_elements+1))
        self.G_external_couples = np.zeros((control_numbers, 3, n_elements))
        self.G_flag = False

    @staticmethod
    @njit(cache=True)
    def muscle_fibers_function(
        controls,
        G_internal_forces, G_internal_couples,
        G_external_forces, G_external_couples,
        internal_forces, internal_couples,
        external_forces, external_couples
    ):
        for index in range(controls.shape[0]):
            inplace_addition(internal_forces, controls[index] * G_internal_forces[index, :, :])
            inplace_addition(internal_couples, controls[index] * G_internal_couples[index, :, :])
            inplace_addition(external_forces, controls[index] * G_external_forces[index, :, :])
            inplace_addition(external_couples, controls[index] * G_external_couples[index, :, :])

class ApplyMuscle(ApplyActuation):
    def __init__(self, muscles, step_skip: int, callback_params_list: list):
        ApplyActuation.__init__(self, None, step_skip, None)
        self.muscles = muscles
        self.callback_params_list = callback_params_list


    def apply_torques(self, system, time: np.float = 0.0):
        for muscle in self.muscles:
            muscle(system)
            inplace_addition(system.external_forces, muscle.external_forces)
            inplace_addition(system.external_torques, muscle.external_couples)

        self.callback()

    def callback_func(self):
        for muscle, callback_params in zip(self.muscles, self.callback_params_list):
            callback_params['activation'].append(muscle.activation.copy())
            callback_params['internal_force'].append(muscle.internal_forces.copy())
            callback_params['internal_couple'].append(muscle.internal_couples.copy())
            callback_params['external_force'].append(muscle.external_forces.copy())
            callback_params['external_couple'].append(muscle.external_couples.copy())

    def apply_muscles(self, muscle, system):
        muscle(system)
        inplace_addition(system.external_forces, muscle.external_forces)
        inplace_addition(system.external_torques, muscle.external_couples)

@njit(cache=True)
def local_strain(off_center_displacement, strain, curvature):
    return strain + _batch_cross(
        quadrature_kernel(curvature),
        off_center_displacement)