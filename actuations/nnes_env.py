from elastica import BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, PositionVerlet, CosseratRod
from elastica.boundary_conditions import ConstraintBase
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper import extend_stepper_interface

from actuations.muscles import LongitudinalMuscle, TransverseMuscle
from actuations.muscles import ApplyMuscle
from actuations.drag_force import DragForce

from collections import defaultdict
import numpy as np
from numba import njit


class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


class RodCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["dilatation"].append(system.dilatation.copy())
            self.callback_params["voronoi_dilatation"].append(system.voronoi_dilatation.copy())
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["director"].append(system.director_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["omega"].append(system.omega_collection.copy())
            self.callback_params["kappa"].append(system.kappa.copy())
            self.callback_params["sigma"].append(system.sigma.copy())
            return


class OneEndFixedRod_with_Flag(ConstraintBase):

    def __init__(self, pointer, **kwargs):
        super().__init__(**kwargs)
        self.pointer = pointer

    def constrain_values(self, rod, time):
        # if self.pointer.flag:
        self.compute_contrain_values(
            rod.position_collection,
            rod.position_collection[..., self.pointer.index],  # self.pointer.fixed_position,
            rod.director_collection,
            rod.director_collection[..., self.pointer.index],  # self.pointer.fixed_directors,
            self.pointer.index
        )

    def constrain_rates(self, rod, time):
        self.compute_constrain_rates(rod.velocity_collection, rod.omega_collection, self.pointer.index)

    @staticmethod
    @njit(cache=True)
    def compute_contrain_values(
            position_collection, fixed_position, director_collection, fixed_directors, index
    ):
        position_collection[..., index] = fixed_position
        director_collection[..., index] = fixed_directors

    @staticmethod
    @njit(cache=True)
    def compute_constrain_rates(velocity_collection, omega_collection, index):
        velocity_collection[..., index] = 0.0
        omega_collection[..., index] = 0.0


class PointerBC():
    def __init__(self, flag, pos, dirs, index):
        self.flag = flag
        self.fixed_position = pos
        self.fixed_directors = dirs
        self.index = index


class Environment:
    def __init__(self, final_time, time_step=1.0e-5, recording_fps=40, early_termination=False):
        # Integrator type
        self.StatefulStepper = PositionVerlet()

        self.final_time = final_time
        self.time_step = time_step
        self.total_steps = int(self.final_time / self.time_step)
        self.recording_fps = recording_fps
        self.step_skip = int(1.0 / (self.recording_fps * self.time_step))
        self.early_termination = early_termination

    def reset(self, cylinder_params=None):
        self.simulator = BaseSimulator()

        """ Set up arm params """
        n_elem = 100
        L0 = 0.2
        radius_base = 0.012  # radius of the arm at the base
        radius_tip = 0.001  # radius of the arm at the tip
        radius = np.linspace(radius_base, radius_tip, n_elem + 1)
        radius_mean = (radius[:-1] + radius[1:]) / 2
        damp_coefficient = 0.1
        self.shearable_rod = CosseratRod.straight_rod(
            n_elements=n_elem,
            start=np.zeros((3,)),
            direction=np.array([1.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            base_length=L0,
            base_radius=radius_mean.copy(),
            density=700,
            nu=damp_coefficient * ((radius / radius_base) ** 2),
            youngs_modulus=1e4,
        )
        self.simulator.append(self.shearable_rod)

        """ Set up boundary condition """
        init_fixed_index = 0
        init_fixed_pos = self.shearable_rod.position_collection[..., init_fixed_index]
        init_fixed_dir = self.shearable_rod.director_collection[..., init_fixed_index]
        self.BC = PointerBC(True, init_fixed_pos, init_fixed_dir, init_fixed_index)
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedRod_with_Flag, pointer=self.BC
        )

        self.rod_parameters_dict = defaultdict(list)
        self.simulator.collect_diagnostics(self.shearable_rod).using(
            RodCallBack,
            step_skip=self.step_skip,
            callback_params=self.rod_parameters_dict
        )

        """ Add muscle actuation """
        self.muscle_layers = [
            LongitudinalMuscle(
                muscle_radius_ratio=np.stack(
                    (np.zeros(radius_mean.shape),
                     6 / 9 * np.ones(radius_mean.shape)),
                    axis=0),
                max_force=0.5 * (radius_mean / radius_base) ** 2,
            ),
            LongitudinalMuscle(
                muscle_radius_ratio=np.stack(
                    (np.zeros(radius_mean.shape),
                     -6 / 9 * np.ones(radius_mean.shape)),
                    axis=0),
                max_force=0.5 * (radius_mean / radius_base) ** 2,
            ),
            TransverseMuscle(
                muscle_radius_ratio=np.stack(
                    (np.zeros(radius_mean.shape),
                     4 / 9 * np.ones(radius_mean.shape)),  # outer radius 1/3 inner radius 1/6
                    axis=0),
                max_force=1.0 * (radius_mean / radius_base) ** 2,
            )
        ]

        self.muscles_parameters = []
        for _ in self.muscle_layers:
            self.muscles_parameters.append(defaultdict(list))

        self.simulator.add_forcing_to(self.shearable_rod).using(
            ApplyMuscle,
            muscles=self.muscle_layers,
            step_skip=self.step_skip,
            callback_params_list=self.muscles_parameters,
        )

        # """ Set up a cylinder object """
        if cylinder_params != None:
            self.cylinder = Cylinder(
                start=cylinder_params['start'] * L0,  # np.array([0.7,0.45, -0.2]) * L0,
                # start=np.array([0.8,0.65, -0.2]) * L0,
                direction=cylinder_params['direction'],  # np.array([0, 0, 1]),
                normal=cylinder_params['normal'],  # np.array([1, 0, 0]),
                base_length=cylinder_params['length'] * L0,  # 0.4 * L0,
                base_radius=cylinder_params['radius'] * L0,  # 0.2 * L0,
                # base_radius=0.4 * L0,
                density=200,
            )

            self.simulator.append(self.cylinder)
            self.simulator.constrain(self.cylinder).using(
                OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
            )
            self.simulator.connect(self.shearable_rod, self.cylinder).using(
                ExternalContact, k=0.001, nu=0.1
            )

        """ Add drag force """
        dl = L0 / n_elem
        fluid_factor = 1
        r_bar = (radius_base + radius_tip) / 2
        sea_water_dentsity = 1022
        c_per = 0.41 / sea_water_dentsity / r_bar / dl * fluid_factor
        c_tan = 0.033 / sea_water_dentsity / np.pi / r_bar / dl * fluid_factor

        self.simulator.add_forcing_to(self.shearable_rod).using(
            DragForce,
            rho_environment=sea_water_dentsity,
            c_per=c_per,
            c_tan=c_tan,
            system=self.shearable_rod,
            step_skip=self.step_skip,
            callback_params=self.rod_parameters_dict
        )

        self.simulator.finalize()

        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        systems = [self.shearable_rod]

        return self.total_steps, systems

    def step(self, time, muscle_activations):

        # set muscle activations
        scale = 1.0
        for muscle_count in range(len(self.muscle_layers)):
            self.muscle_layers[muscle_count].set_activation(muscle_activations[muscle_count] * scale)

        # run the simulation for one step
        time = self.do_step(
            self.StatefulStepper,
            self.stages_and_updates,
            self.simulator,
            time,
            self.time_step,
        )

        if self.early_termination:
            done = self.check_early_termination()
        else:
            """ Done is a boolean to reset the environment before episode is completed """
            done = False
            # Position of the rod cannot be NaN, it is not valid, stop the simulation
            invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)

            if invalid_values_condition == True:
                print(" Nan detected, exiting simulation now")
                done = True
            """ Done is a boolean to reset the environment before episode is completed """

        # systems = [self.shearable_rod]
        systems = [self.shearable_rod, self.cylinder]

        return time, systems, done

    def save_data(self, dir, eps):

        print("Saving data to pickle files...")

        import pickle

        with open(dir + "/simulation_data%03d.pickle" % eps, "wb") as f:
            data = dict(
                rods=[self.rod_parameters_dict],
                muscles=self.muscles_parameters
            )
            pickle.dump(data, f)

        with open(dir + "/simulation_systems%03d.pickle" % eps, "wb") as f:
            data = dict(
                rods=[self.shearable_rod],
                muscles=self.muscle_layers
            )
            pickle.dump(data, f)

    def set_algo_data(self, algo_data):
        self.desired_sigma = algo_data['sigma']
        self.desired_kappa = algo_data['kappa']

    def check_early_termination(self, cutoff_error=1e-7):
        desired_Hamiltonian = self.cal_desired_Hamiltonian()
        if desired_Hamiltonian < cutoff_error:
            return True
        return False

    def cal_desired_Hamiltonian(self):
        kinetic_energy = (
                compute_translational_energy(
                    self.shearable_rod.mass,
                    self.shearable_rod.velocity_collection) +
                compute_rotational_energy(
                    self.shearable_rod.mass_second_moment_of_inertia,
                    self.shearable_rod.omega_collection,
                    self.shearable_rod.dilatation
                )
        )
        desired_potential_energy = (
                compute_shear_energy(
                    self.shearable_rod.sigma,
                    self.desired_sigma,
                    self.shearable_rod.shear_matrix,
                    self.shearable_rod.dilatation * self.shearable_rod.rest_lengths
                ) +
                compute_bending_energy(
                    self.shearable_rod.kappa,
                    self.desired_kappa,
                    self.shearable_rod.bend_matrix,
                    self.shearable_rod.voronoi_dilatation * self.shearable_rod.rest_voronoi_lengths
                )
        )
        return kinetic_energy + desired_potential_energy

    def post_processing(self, algo_data, cutoff_error=1e-7):
        for k in range(len(self.rod_parameters_dict['time'])):

            # calculate the desired Hamiltonian for every time frame
            kinetic_energy = (
                    compute_translational_energy(
                        self.shearable_rod.mass,
                        self.rod_parameters_dict['velocity'][k]) +
                    compute_rotational_energy(
                        self.shearable_rod.mass_second_moment_of_inertia,
                        self.rod_parameters_dict['omega'][k],
                        self.rod_parameters_dict['dilatation'][k]
                    )
            )
            desired_potential_energy = (
                    compute_shear_energy(
                        self.rod_parameters_dict['sigma'][k],
                        algo_data['sigma'],
                        self.shearable_rod.shear_matrix,
                        self.rod_parameters_dict['dilatation'][k] * self.shearable_rod.rest_lengths
                    ) +
                    compute_bending_energy(
                        self.rod_parameters_dict['kappa'][k],
                        algo_data['kappa'],
                        self.shearable_rod.bend_matrix,
                        self.rod_parameters_dict['voronoi_dilatation'][k] * self.shearable_rod.rest_voronoi_lengths
                    )
            )
            desired_Hamiltonian = kinetic_energy + desired_potential_energy

            # calculate the control energy
            self.muscles_parameters[0]['control_energy'].append(
                0.5 * np.sum(self.muscles_parameters[0]['activation'][k] ** 2)
            )

            # check if the desired Hamiltonian is smaller then cutoff error
            flag = False
            if desired_Hamiltonian < cutoff_error:
                flag = True
            self.rod_parameters_dict['stable_flag'].append(flag)

        for k in range(len(self.rod_parameters_dict['time'])):
            if self.rod_parameters_dict['stable_flag'][-k - 1] is False:
                self.rod_parameters_dict['stable_flag'][:-k - 1] = (
                    [False for kk in range(len(self.rod_parameters_dict['stable_flag'][:-k - 1]))]
                )
                break

        energy_cost = 0
        reach_time = self.rod_parameters_dict['time'][-1]
        for k in range(len(self.rod_parameters_dict['time'])):
            if self.rod_parameters_dict['stable_flag'][k] is False:
                energy_cost += self.muscles_parameters[0]['control_energy'][k]
            else:
                reach_time = self.rod_parameters_dict['time'][k]

        return reach_time, energy_cost


def compute_translational_energy(mass, velocity):
    return (0.5 * (mass * np.einsum("ij, ij-> j", velocity, velocity)).sum())


def compute_rotational_energy(mass_second_moment_of_inertia, omega, dilatation):
    J_omega_upon_e = (_batch_matvec(mass_second_moment_of_inertia, omega) / dilatation)
    return 0.5 * np.einsum("ik,ik->k", omega, J_omega_upon_e).sum()


def compute_bending_energy(kappa, target_kappa, bend_matrix, rest_voronoi_lengths):
    kappa_diff = kappa - target_kappa
    bending_internal_torques = _batch_matvec(bend_matrix, kappa_diff)

    return (0.5 * (_batch_dot(kappa_diff, bending_internal_torques) * rest_voronoi_lengths).sum())


def compute_shear_energy(sigma, target_sigma, shear_matrix, rest_lengths):
    sigma_diff = sigma - target_sigma
    shear_internal_torques = _batch_matvec(shear_matrix, sigma_diff)

    return (0.5 * (_batch_dot(sigma_diff, shear_internal_torques) * rest_lengths).sum())
