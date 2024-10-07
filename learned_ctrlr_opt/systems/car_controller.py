# Nominal controller for the car racing environment
# based on a natcar-like line-following steering setup,
# with a fixed gas/brake. Definitely not optimal, should
# get an MPC-type controller working if possible.

import numpy as np
import h5py
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from learned_ctrlr_opt.systems.car_racing import CarRacing
from learned_ctrlr_opt.systems.car_dynamics import CarParams

def find_nearest_point(point, list_of_points):
    diffs = np.linalg.norm(list_of_points - point, axis=1)
    return list_of_points[np.argmin(diffs)]


@dataclass
class CarControllerParams:
    steer_kp: float = 0.1
    steer_kd: float = 0.2
    gas: float = 0.1
    brake: float = 0.1
    max_speed: int = 75
    braking_threshold: float = 0.4
    cone_height: int = 20  # maybe these two should actually be intrinsic params?
    cone_width: int = 20

    def get_list(self):
        return np.array([self.steer_kp,
                self.steer_kd,
                self.gas,
                self.brake,
                self.max_speed,
                self.braking_threshold,
                self.cone_height,
                self.cone_width])

    @staticmethod
    def get_names():
        return ["Steer KP",
                "Steer KD",
                "Gas",
                "Brake",
                "Max Speed",
                "Braking Threshold",
                "Cone Height",
                "Cone Width"]

    @staticmethod
    def get_bounds():
        return np.array([[0.01, 0.5],
                         [0.01, 0.5],
                         [0.02, 1.0],
                         [0.02, 1.0],
                         [20, 200],
                         [0.01, 1.0],
                         [10, 30],
                         [10, 30]])

    @staticmethod
    def get_num():
        return 8

    @staticmethod
    def generate_random(to_randomize=None):
        if to_randomize is None:
            to_randomize = [i for i in range(CarControllerParams.get_num())]
        params = CarControllerParams().get_list()
        for idx in to_randomize:
            b = CarControllerParams.get_bounds()[idx]
            params[idx] = np.random.uniform(b[0], b[1])
        return CarControllerParams(*params)


class CarController:
    metric_names = ["Tracking Error", "Wheelslip", "Avg. Speed", "Laptime"]
    def __init__(self, gains):
        self.gains = gains

        self.prev_lat_dist = 0
        self.prev_fwd_dist = 0
        self.debug = False 
        # print(f"Initializing controller: params={self.gains.get_list()}")

    # PD control for steering
    def compute_steering(self, lat_dist):
        steer = self.gains.steer_kp*lat_dist + self.gains.steer_kd*(lat_dist - self.prev_lat_dist)
        self.prev_lat_dist = lat_dist
        return -np.clip(steer, -1.0, 1.0)


    def compute_gas_brake_dumb(self, lat, true_speed, steering, target, lookahead):
        brake = 0
        gas = 0
        # Should max speed be a car parameter instead of a control parameter?
        if true_speed < self.gains.max_speed and np.abs(steering) <= self.gains.braking_threshold:
            gas = self.gains.gas
        elif np.abs(steering) > self.gains.braking_threshold and true_speed >= 10:
            brake = self.gains.brake * true_speed/self.gains.max_speed
        return gas, brake


    def _is_point_in_cone(self, point, tip, dir):
        diff = point - tip
        cone_dist = np.dot(dir, diff)
        if cone_dist > self.gains.cone_height or cone_dist < 0:
            return False
        cone_r = cone_dist/self.gains.cone_height * self.gains.cone_width
        orth_dist = np.linalg.norm(diff - cone_dist * dir)
        if orth_dist < cone_r:
            return True
        return False


    def tracking_error(self, setpoint, actual_point):
        return np.linalg.norm(setpoint - actual_point)
    

    def is_wheelslip(self, car):
        if car.is_skidding:
            return 1
        return 0 


    def true_speed(self, car):
        true_speed = np.sqrt(
            np.square(car.hull.linearVelocity[0])
            + np.square(car.hull.linearVelocity[1])
        )
        return true_speed


    def test_on_track_return_traj(self, car_env, num_steps=50000, seed=42, length=None):
        s, info = car_env.reset()
        track = np.array([[car_env.track[i][2], car_env.track[i][3]] for i in range(len(car_env.track))])
        i = 0
        target = track[2]
        # metrics = np.zeros((num_steps, 4))
        metrics = []
        first_target = -1
        prev_target_idx = -1
        going_fwd = True
        track_visited = []
        action_dim = 3
        state_dim = 3
        # this ain't gonna work. But I do need consistent lengths...
        trajectory = np.zeros((num_steps, action_dim+state_dim))
        while i < num_steps:
            # instead of doing this bs, just find the nearest point in the track that is
            # within some cone in front of the car.
            p = np.array([car_env.car.hull.position[0], car_env.car.hull.position[1]])
            car_angle = -car_env.car.hull.angle
            cone_dir = np.array([np.sin(car_angle), np.cos(car_angle)])
            lat_dir = np.array([np.cos(car_angle), -np.sin(car_angle)])

            # Super inefficient way of doing this-scanning the whole track
            targets = []
            target_idxs = []
            for j, t in enumerate(track):
                if self._is_point_in_cone(t, p+cone_dir, cone_dir):
                    targets.append(t)
                    target_idxs.append(j)

            if len(targets) > 0:
                furthest = np.argmax(np.linalg.norm(np.array(targets) - p, axis=1))
                if prev_target_idx != -1:
                    prev_target_idx = target_idx
                target_idx = target_idxs[furthest]
                if target_idx > prev_target_idx and target_idx < prev_target_idx+10:
                    track_visited.append(target_idx)
                    going_fwd = True
                else:
                    going_fwd = False
                target = targets[furthest]
                lookahead = targets[furthest]
                if prev_target_idx == -1:
                    prev_target_idx = target_idx
                # print(f"target_idx = {target_idx}, prev_idx = {prev_target_idx}")
            if i == 0:
                first_target = target_idx
                # print(f"first target = {first_target}")
                # final_target = (first_target + 200) % len(track)
                if length is None:
                    final_target = len(track) - 10
                else:
                    final_target = length
                # print(f"final target = {final_target}")

            # Done with this lap
            if (final_target < first_target and
                    target_idx < first_target and
                    target_idx > final_target and
                    going_fwd and
                    (len(track_visited) > 0.8 * final_target or length is not None)):
                break
            elif (final_target > first_target and
                  target_idx > final_target and
                  going_fwd and
                  (len(track_visited) > 0.8 * final_target or length is not None)):
                break

            dist = np.array(target) - p
            # dist = pygame.math.Vector2(dist[0], dist[1]).rotate_rad(car_angle)
            fwd_dist = dist[1]
            lat_dist = -np.dot(dist, lat_dir)

            steering = self.compute_steering(lat_dist)
            true_speed = self.true_speed(car_env.car)
            gas, brake = self.compute_gas_brake_dumb(lat_dir, true_speed, steering, target, lookahead)

            action = [steering, gas, brake]
            trajectory[i][0] = true_speed
            trajectory[i][1] = car_angle
            trajectory[i][2] = lat_dist
            print(lat_dist)
            trajectory[i][3:] = np.array(action)
            s, r, d, info = car_env.step(action)
            # car_env.render()
            metrics.append([self.tracking_error(p, target), self.is_wheelslip(car_env.car), self.true_speed(car_env.car)])
            i += 1
            # if i % 20 == 0:
            # print(f"target idx = {target_idx}")
            if i % 15 == 0 and self.debug:
                print(f"gas, brake = {gas, brake}")
                print(f"car pos = {p}, track target= {target}")
                print(f"cone_dir = {cone_dir}")
                print(f"fwd_dist = {fwd_dist}, lat_dist = {lat_dist}")
                print(f"actions: {[steering, gas, brake]}")
        # print(f"i = {i}")
        return metrics, i, trajectory



    def test_on_track(self, car_env, num_steps=50000, seed=42, length=None):
        s, info = car_env.reset()
        track = np.array([[car_env.track[i][2], car_env.track[i][3]] for i in range(len(car_env.track))])
        i = 0
        target = track[2]
        # metrics = np.zeros((num_steps, 4))
        metrics = []
        first_target = -1
        prev_target_idx = -1
        going_fwd = True
        track_visited = []
        while i < num_steps:
            # instead of doing this bs, just find the nearest point in the track that is
            # within some cone in front of the car.
            p = np.array([car_env.car.hull.position[0], car_env.car.hull.position[1]])
            car_angle = -car_env.car.hull.angle
            cone_dir = np.array([np.sin(car_angle), np.cos(car_angle)])
            lat_dir = np.array([np.cos(car_angle), -np.sin(car_angle)])

            # Super inefficient way of doing this-scanning the whole track
            targets = []
            target_idxs = []
            for j, t in enumerate(track):
                if self._is_point_in_cone(t, p+cone_dir, cone_dir):
                    targets.append(t)
                    target_idxs.append(j)

            if len(targets) > 0:
                furthest = np.argmax(np.linalg.norm(np.array(targets) - p, axis=1))
                if prev_target_idx != -1:
                    prev_target_idx = target_idx
                target_idx = target_idxs[furthest]
                if target_idx > prev_target_idx and target_idx < prev_target_idx+10:
                    track_visited.append(target_idx)
                    going_fwd = True
                else:
                    going_fwd = False
                target = targets[furthest]
                lookahead = targets[furthest]
                if prev_target_idx == -1:
                    prev_target_idx = target_idx
                # print(f"target_idx = {target_idx}, prev_idx = {prev_target_idx}")
            if i == 0:
                first_target = target_idx
                # print(f"first target = {first_target}")
                # final_target = (first_target + 200) % len(track)
                if length is None:
                    final_target = len(track) - 10
                else:
                    final_target = length
                # print(f"final target = {final_target}")

            # Done with this lap
            if (final_target < first_target and
                target_idx < first_target and
                target_idx > final_target and
               going_fwd and
                    (len(track_visited) > 0.8 * final_target or length is not None)):
                break
            elif (final_target > first_target and
                  target_idx > final_target and
                  going_fwd and
                  (len(track_visited) > 0.8 * final_target or length is not None)):
                break

            dist = np.array(target) - p
            # dist = pygame.math.Vector2(dist[0], dist[1]).rotate_rad(car_angle)
            fwd_dist = dist[1]
            lat_dist = -np.dot(dist, lat_dir)

            steering = self.compute_steering(lat_dist)
            true_speed = self.true_speed(car_env.car)
            gas, brake = self.compute_gas_brake_dumb(lat_dir, true_speed, steering, target, lookahead)

            action = [steering, gas, brake]
            s, r, d, info = car_env.step(action)
            # car_env.render()
            metrics.append([self.tracking_error(p, target), self.is_wheelslip(car_env.car), self.true_speed(car_env.car)])
            i += 1
            # if i % 20 == 0:
                # print(f"target idx = {target_idx}")
            if i % 15 == 0 and self.debug:
                print(f"gas, brake = {gas, brake}")
                print(f"car pos = {p}, track target= {target}")
                print(f"cone_dir = {cone_dir}")
                print(f"fwd_dist = {fwd_dist}, lat_dist = {lat_dist}")
                print(f"actions: {[steering, gas, brake]}")
        # print(f"i = {i}")
        return metrics, i

    def test_on_track_reset_free(self, car_env, num_steps=1000, length=None):
        s = car_env.get_current_state()
        track = np.array([[car_env.track[i][2], car_env.track[i][3]] for i in range(len(car_env.track))])
        current_position = car_env.get_current_state()[:2]
        # find the closest idx to the car and say that's our starting index.
        min_dist = 100
        closest_idx = 0
        for t in range(track.shape[0]):
            dist = np.linalg.norm(track[t] - current_position)
            if dist < min_dist:
                min_dist = dist
                closest_idx = t

        # shift the track forward so the starting idx is now 0
        track_rot = np.roll(track, -closest_idx, axis=0)

        # set the final target based on length
        if length is None:
            raise NotImplementedError("In reset-free, length cannot be None")
        else:
            if length > len(track):
                raise NotImplementedError(f"length {length} is longer than track length {len(track)}")
            final_target = length

        # find the next target, do all the action stuff, etc.
        i = 0
        track_visited = []
        going_fwd = True
        action_dim = 3
        state_dim = 3
        trajectory = np.zeros((num_steps, action_dim+state_dim))
        metrics = []
        prev_target_idx = 0
        # target = current_position
        target_idx = 5   # closest_idx
        target = track_rot[target_idx]
        while i < num_steps:
            # instead of doing this bs, just find the nearest point in the track that is
            # within some cone in front of the car.
            p = np.array([car_env.car.hull.position[0], car_env.car.hull.position[1]])
            car_angle = -car_env.car.hull.angle
            cone_dir = np.array([np.sin(car_angle), np.cos(car_angle)])
            lat_dir = np.array([np.cos(car_angle), -np.sin(car_angle)])

            # Super inefficient way of doing this-scanning the whole track
            targets = []
            target_idxs = []
            for j, t in enumerate(track_rot):
                if self._is_point_in_cone(t, p+cone_dir, cone_dir):
                    targets.append(t)
                    target_idxs.append(j)

            if len(targets) > 0:
                furthest = np.argmax(np.linalg.norm(np.array(targets) - p, axis=1))
                prev_target_idx = target_idx
                target_idx = target_idxs[furthest]
                if target_idx > prev_target_idx and target_idx < prev_target_idx+10:
                    track_visited.append(target_idx)
                    going_fwd = True
                else:
                    going_fwd = False
                if not going_fwd:
                    target_idx = prev_target_idx
                target = track_rot[target_idx]
                # print(f"target_idx = {target_idx}, prev_idx = {prev_target_idx}")
            # if i == 0:
            #     if len(targets) == 0:
            #         first_target = 1  # we've rotated the track
            #         target_idx = 1
            #         target = track_rot[1]
            #     else:
            #         first_target = target_idx
            #         target = track_rot[first_target]

            if prev_target_idx < final_target and target_idx >= final_target and going_fwd:
                # print(f"Done! prev target was {prev_target_idx}, target_idx = {target_idx}, final_target = {final_target}")
                break

            # Done with this lap
            # if (final_target < first_target and
            #    final_target < target_idx < first_target and
            #    going_fwd):
            #     break
            # elif (target_idx > final_target > first_target and
            #       going_fwd):
            #     break

            dist = np.array(target) - p
            fwd_dist = dist[1]
            lat_dist = -np.dot(dist, lat_dir)

            steering = self.compute_steering(lat_dist)
            true_speed = self.true_speed(car_env.car)
            gas, brake = self.compute_gas_brake_dumb(lat_dir, true_speed, steering, target, 0)

            # have to get angular velocity
            action = [steering, gas, brake]
            trajectory[i][0] = true_speed
            # trajectory[i][1] = car_angle
            trajectory[i][1] = car_env.car.hull.angularVelocity
            trajectory[i][2] = lat_dist
            trajectory[i][3:] = np.array(action)
            out = car_env.step(action)
            s = out[0]
            r = out[1]
            d = out[2]
            trunc = out[3]
            # car_env.render()
            metrics.append([self.tracking_error(p, target), self.is_wheelslip(car_env.car), self.true_speed(car_env.car)])
            i += 1
            # if i % 20 == 0:
            # print(f"target idx = {target_idx}")
            if i % 15 == 0 and self.debug:
                print(f"gas, brake = {gas, brake}")
                print(f"car pos = {p}, track target= {target}")
                print(f"cone_dir = {cone_dir}")
                print(f"fwd_dist = {fwd_dist}, lat_dist = {lat_dist}")
                print(f"actions: {[steering, gas, brake]}")
        # print(f"i = {i}")
        return metrics, i, trajectory

    def get_track_ahead_from_current_state(self, car_env):
        track = np.array([[car_env.track[i][2], car_env.track[i][3]] for i in range(len(car_env.track))])
        p = np.array([car_env.car.hull.position[0], car_env.car.hull.position[1]])
        car_angle = -car_env.car.hull.angle
        cone_dir = np.array([np.sin(car_angle), np.cos(car_angle)])

        first_idx = 0
        for j, t in enumerate(track):
            if self._is_point_in_cone(t, p+cone_dir, cone_dir):
                first_idx = j
                break
        return track[first_idx:, ...]

def get_ref_track_pcas_and_scaler(path_to_dataset, ref_track_key, n_components_per):
    dset_f = h5py.File(path_to_dataset, 'r')
    ref_trajs = np.array(dset_f[ref_track_key])
    ref_trajs_all = np.reshape(ref_trajs, (ref_trajs.shape[0]*ref_trajs.shape[1], ref_trajs.shape[-2], 2))
    ref_trajs_all[:,:, 0] -= ref_trajs_all[:,0:1,0]  # subtract x of first waypoint
    ref_trajs_all[:,:, 1] -= ref_trajs_all[:,0:1,1]  # subtract y of first waypoint
    # Even this is not ideal as it is not rotation invariant...
    # Do PCA on x components
    ref_trajs_x = ref_trajs_all[:,:,0]
    pca_x = PCA(n_components=n_components_per)
    ref_trajs_x_pca = pca_x.fit_transform(ref_trajs_x)
    # Do PCA on y components
    ref_trajs_y = ref_trajs_all[:,:,1]
    pca_y = PCA(n_components=n_components_per)
    ref_trajs_y_pca = pca_y.fit_transform(ref_trajs_y)
    ref_trajs_pca = np.hstack((ref_trajs_x_pca, ref_trajs_y_pca))
    ref_track_scaler = MinMaxScaler(clip=True).fit(ref_trajs_pca)
    dset_f.close()
    return pca_x, pca_y, ref_track_scaler

# track_ahead: (length, 2)
def pp_track(track_ahead, pca_x, pca_y, scaler):
    track_ahead[:, 0] -= track_ahead[0:1,0]  # subtract x of first waypoint
    track_ahead[:, 1] -= track_ahead[0:1,1]  # subtract y of first waypoint
    return scaler.transform(np.hstack([pca_x.transform(track_ahead[:,0].reshape(1, -1)),
                                       pca_y.transform(track_ahead[:,1].reshape(1, -1))])).squeeze()


def pp_track_curvature(track_ahead, scaler, ds_factor):
    if len(track_ahead.shape) < 3:
        track_ahead = np.expand_dims(track_ahead, 0)
    angs = np.zeros((track_ahead.shape[0], track_ahead.shape[1]))
    for i in range(1, track_ahead.shape[1]-1):
        ang1 = np.arcsin((track_ahead[:, i, 0] - track_ahead[:, i-1, 0])/np.linalg.norm(track_ahead[:, i] - track_ahead[:, i-1], axis=-1))
        ang2 = np.arcsin((track_ahead[:, i+1, 0] - track_ahead[:, i, 0])/np.linalg.norm(track_ahead[:, i+1] - track_ahead[:, i], axis=-1))
        angs[:,i] = np.abs(ang1 - ang2)
    angs_flat = np.reshape(angs, (-1, 1))
    return scaler.transform(angs_flat).reshape(angs.shape)[...,::ds_factor]
        

def main():
    gains = CarControllerParams(0.4, 0.5, 0.7, 0.02, 75.0, 20.0, 20.0)
    ctrlr = CarController(gains)
    env = CarRacing(CarParams(), render_mode="human")
    env.reset(seed=20)
    # ctrlr.test_on_track(env, num_steps=2500, seed=1)
    ctrlr.test_on_track_reset_free(env, 200)


if __name__ == "__main__":
    main()
