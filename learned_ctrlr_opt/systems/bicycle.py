# sourced from: https://github.com/winstxnhdw/KinematicBicycleModel/
from math import cos, sin, tan, atan2
import numpy as np

normalise_angle = lambda angle: atan2(sin(angle), cos(angle))

class KinematicBicycleModel:
    """
    Summary
    -------
    This class implements the 2D Kinematic Bicycle Model for vehicle dynamics
    Attributes
    ----------
    dt (float) : discrete time period [s]
    wheelbase (float) : vehicle's wheelbase [m]
    max_steer (float) : vehicle's steering limits [rad]
    Methods
    -------
    __init__(wheelbase: float, max_steer: float, delta_time: float=0.05)
        initialises the class
    update(x, y, yaw, velocity, acceleration, steering_angle)
        updates the vehicle's state using the kinematic bicycle model
    """

    def __init__(self, wheelbase: float, max_steer: float, delta_time: float = 0.05):
        self.delta_time = delta_time
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.state = [0, 0, 0, 0]

    def reset(self):
        self.state = [0, 0, 0, 0]
        return np.array(self.state)

    def step(self, u):
        """
        Summary
        -------
        Updates the vehicle's state using the kinematic bicycle model
        Parameters
        ----------
        x (int) : vehicle's x-coordinate [m]
        y (int) : vehicle's y-coordinate [m]
        yaw (int) : vehicle's heading [rad]
        velocity (int) : vehicle's velocity in the x-axis [m/s]
        acceleration (int) : vehicle's accleration [m/s^2]
        steering_angle (int) : vehicle's steering angle [rad]
        Returns
        -------
        new_x (int) : vehicle's x-coordinate [m]
        new_y (int) : vehicle's y-coordinate [m]
        new_yaw (int) : vehicle's heading [rad]
        new_velocity (int) : vehicle's velocity in the x-axis [m/s]
        steering_angle (int) : vehicle's steering angle [rad]
        angular_velocity (int) : vehicle's angular velocity [rad/s]
        """
        velocity = self.state[3]
        acceleration = u[0]
        steering_angle = u[1]

        # Compute the local velocity in the x-axis
        new_velocity = velocity + self.delta_time * acceleration

        # Limit steering angle to physical vehicle limits
        steering_angle = -self.max_steer if steering_angle < -self.max_steer else self.max_steer if steering_angle > self.max_steer else steering_angle

        # Compute the angular velocity
        angular_velocity = new_velocity * tan(steering_angle) / self.wheelbase

        # Compute the final state using the discrete time model
        x, y, yaw = self.state[0], self.state[1], self.state[2]
        new_x = x + velocity * cos(yaw) * self.delta_time
        new_y = y + velocity * sin(yaw) * self.delta_time
        new_yaw = normalise_angle(yaw + angular_velocity * self.delta_time)

        self.state = [new_x, new_y, new_yaw, new_velocity]
        return np.array(self.state)
