"""
Top-down car dynamics simulation.

Some ideas are taken from this great tutorial http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
This simulation is a bit more detailed, with wheels rotation.

Created by Oleg Klimov
"""

import math

import Box2D
import numpy as np
from dataclasses import dataclass
from gym.error import DependencyNotInstalled

try:
    from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef
except ImportError:
    raise DependencyNotInstalled("box2D is not installed, run `pip install gym[box2d]`")


'''
# TODO: incorporate mass changes, friction for each wheel, etc
SIZE = 0.02
default_params = {"size": SIZE,
                  "engine_power":100000000 * (SIZE**2),
                  "friction_limit":1000000 * (SIZE**2),
                  "wheel_inertia":4000 * (SIZE**2)}

# TODO(hersh500): use dataclass for this. better than throwing around strings.
def build_params(size=0.02, engine_coeff=100000000, friction_coeff=1000000, wheel_coeff=4000):
    params = {"size": size,
              "engine_power":engine_coeff * (size**2),
              "friction_limit":friction_coeff * (size**2),
              "wheel_inertia":wheel_coeff* (size**2)}
    return params
'''

@dataclass
class CarParams:
    size: float = 0.02
    engine_power:float = 100000000 * (0.02**2)
    friction_limit: float = 1000000 * (0.02**2)
    wheel_inertia: float = 4000 * (0.02**2)

    # These are bogus and can't actually be controlled. But useful to just have them for indexing
    # So these are technically "always" randomized, but fixed when operating with a fixed seed.
    avg_curvature: float = 0.02
    track_length: float = 300

    def get_list(self):
        return np.array([self.size, self.engine_power, self.friction_limit, self.wheel_inertia,
                self.avg_curvature, self.track_length])

    def get_dict(self):
        return {CarParams.get_names()[i]: self.get_list()[i] for i in range(CarParams.get_num())}

    @staticmethod
    def get_names():
        return ["size", "engine_power", "friction_limit", "wheel_inertia", "avg_curvature", "track_length"]

    @staticmethod
    def get_num():
        return 6

    @staticmethod
    def get_bounds():
        params_box = np.array([[0.005, 0.04],
                               [20000, 50000],
                               [200, 500],
                               [1.6, 1.6],
                               [0.00, 0.08],
                               [200, 350]])
        return params_box

    @staticmethod
    def generate_random(to_randomize=None):
        if to_randomize is None:
            to_randomize = [i for i in range(CarParams.get_num())]
        params = CarParams().get_list()
        for idx in to_randomize:
            b = CarParams.get_bounds()[idx]
            params[idx] = np.random.uniform(b[0], b[1])
        return CarParams(*params)


@dataclass
class CarParamsTrain(CarParams):
    @staticmethod
    def get_bounds():
        params_box = np.array([[0.01, 0.03],
                               [25000, 45000],
                               [250, 450],
                               [1.6, 1.6],
                               [0.00, 0.08],
                               [200, 350]])
        return params_box


    @staticmethod
    def generate_random(to_randomize=None):
        if to_randomize is None:
            to_randomize = [i for i in range(CarParamsTrain.get_num())]
        params = CarParamsTrain().get_list()
        for idx in to_randomize:
            b = CarParamsTrain.get_bounds()[idx]
            params[idx] = np.random.uniform(b[0], b[1])
        return CarParamsTrain(*params)


@dataclass
class CarParamsTest(CarParams):
    @staticmethod
    def lower_half_bounds():
        return np.array([[CarParams.get_bounds()[i,0], CarParamsTrain.get_bounds()[i,0]] for i in range(6)])

    @staticmethod
    def upper_half_bounds():
        return np.array([[CarParamsTrain.get_bounds()[i,1], CarParams.get_bounds()[i,1]] for i in range(6)])

    @staticmethod
    def generate_random(to_randomize=None, half=False):
        if to_randomize is None:
            to_randomize = [i for i in range(CarParamsTest.get_num())]
        params = CarParamsTest().get_list()
        if half:
            lower = np.random.randint(2) * np.ones(len(to_randomize))
        else:
            lower = np.random.randint(2, size=len(to_randomize))
        for idx in to_randomize:
            if lower[idx] == 0:
                bounds = CarParamsTest.lower_half_bounds()[idx]
            else:
                bounds = CarParamsTest.upper_half_bounds()[idx]
            params[idx] = np.random.uniform(bounds[0], bounds[1])
        return CarParamsTest(*params)


WHEEL_R = 27
WHEEL_W = 14
WHEELPOS = [(-55, +80), (+55, +80), (-55, -82), (+55, -82)]
HULL_POLY1 = [(-60, +130), (+60, +130), (+60, +110), (-60, +110)]
HULL_POLY2 = [(-15, +120), (+15, +120), (+20, +20), (-20, 20)]
HULL_POLY3 = [
    (+25, +20),
    (+50, -10),
    (+50, -40),
    (+20, -90),
    (-20, -90),
    (-50, -40),
    (-50, -10),
    (-25, +20),
]
HULL_POLY4 = [(-50, -120), (+50, -120), (+50, -90), (-50, -90)]
WHEEL_COLOR = (0, 0, 0)
WHEEL_WHITE = (77, 77, 77)
MUD_COLOR = (102, 102, 0)


class Car:
    def __init__(self, world, init_angle, init_x, init_y, params=CarParams()):
        self.params = params
        self.size = self.params.size

        self.is_skidding = False
        self.world: Box2D.b2World = world
        self.hull: Box2D.b2Body = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * self.size, y * self.size) for x, y in HULL_POLY1]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * self.size, y * self.size) for x, y in HULL_POLY2]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * self.size, y * self.size) for x, y in HULL_POLY3]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * self.size, y * self.size) for x, y in HULL_POLY4]
                    ),
                    density=1.0,
                ),
            ],
        )
        self.hull.color = (0.8, 0.0, 0.0)
        self.wheels = []
        self.fuel_spent = 0.0
        WHEEL_POLY = [
            (-WHEEL_W, +WHEEL_R),
            (+WHEEL_W, +WHEEL_R),
            (+WHEEL_W, -WHEEL_R),
            (-WHEEL_W, -WHEEL_R),
        ]
        for wx, wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position=(init_x + wx * self.size, init_y + wy * self.size),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[
                            (x * front_k * self.size, y * front_k * self.size)
                            for x, y in WHEEL_POLY
                        ]
                    ),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0,
                ),
            )
            w.wheel_rad = front_k * WHEEL_R * self.size 
            w.color = WHEEL_COLOR
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.skid_start = None
            w.skid_particle = None
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx * self.size, wy * self.size),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180 * 900 * self.size * self.size,
                motorSpeed=0,
                lowerAngle=-0.4,
                upperAngle=+0.4,
            )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)
        self.drawlist = self.wheels + [self.hull]
        self.particles = []

    def set_params(self, params):
        prev_size = self.params.size
        self.params = params
        self.size = params.size
        self.friction_limit = self.params.friction_limit
        self.engine_power = self.params.engine_power
        self.wheel_inertia = self.params.wheel_inertia
        for w in self.wheels:
            w.wheel_rad  = w.wheel_rad/prev_size * self.params.size


    def gas(self, gas):
        """control: rear wheel drive

        Args:
            gas (float): How much gas gets applied. Gets clipped between 0 and 1.
        """
        gas = np.clip(gas, 0, 1)
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.1:
                diff = 0.1  # gradually increase, but stop immediately
            w.gas += diff

    def brake(self, b):
        """control: brake

        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation"""
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        """control: steer

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side"""
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def step(self, dt, use_grass=True):
        self.is_skidding = False
        for w in self.wheels:
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = dir * min(50.0 * val, 3.0)

            # Position => friction_limit
            grass = True
            # Grass causes bifurcations in the friction behavior...
            if use_grass:
                friction_limit = self.params.friction_limit * 0.6  # Grass friction if no tile
            else:
                friction_limit = self.params.friction_limit
            for tile in w.tiles:
                friction_limit = max(
                    friction_limit, self.params.friction_limit * tile.road_friction
                )
                grass = False

            # Force
            forw = w.GetWorldVector((0, 1))
            side = w.GetWorldVector((1, 0))
            v = w.linearVelocity
            vf = forw[0] * v[0] + forw[1] * v[1]  # forward speed
            vs = side[0] * v[0] + side[1] * v[1]  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega

            # add small coef not to divide by zero
            w.omega += (
                dt
                * self.params.engine_power
                * w.gas
                / self.params.wheel_inertia 
                / (abs(w.omega) + 5.0)
            )
            self.fuel_spent += dt * self.params.engine_power * w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15  # radians per second
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE * w.brake
                if abs(val) > abs(w.omega):
                    val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir * val
            w.phase += w.omega * dt

            vr = w.omega * w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr  # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.

            # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            f_force *= 205000 * self.size * self.size 
            p_force *= 205000 * self.size * self.size
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            # Skid trace
            if abs(force) > 2.0 * friction_limit:
                if (
                    w.skid_particle
                    and w.skid_particle.grass == grass
                    and len(w.skid_particle.poly) < 30
                ):
                    w.skid_particle.poly.append((w.position[0], w.position[1]))
                elif w.skid_start is None:
                    w.skid_start = w.position
                else:
                    w.skid_particle = self._create_particle(
                        w.skid_start, w.position, grass
                    )
                    w.skid_start = None
                # adding additional state variables for cost fn tracking
                self.is_skidding = True
            else:
                w.skid_start = None
                w.skid_particle = None

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            w.omega -= dt * f_force * w.wheel_rad / self.params.wheel_inertia

            w.ApplyForceToCenter(
                (
                    p_force * side[0] + f_force * forw[0],
                    p_force * side[1] + f_force * forw[1],
                ),
                True,
            )
        return np.array([self.hull.position.x, self.hull.position.y,
                         self.hull.linearVelocity.x, self.hull.linearVelocity.y,
                         self.hull.angle, self.wheels[0].joint.angle])

    def draw(self, surface, zoom, translation, angle, draw_particles=True):
        import pygame.draw

        if draw_particles:
            for p in self.particles:
                poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in p.poly]
                poly = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1],
                    )
                    for coords in poly
                ]
                pygame.draw.lines(
                    surface, color=p.color, points=poly, width=2, closed=False
                )

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                path = [(coords[0], coords[1]) for coords in path]
                path = [pygame.math.Vector2(c).rotate_rad(angle) for c in path]
                path = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1],
                    )
                    for coords in path
                ]
                color = [int(c * 255) for c in obj.color]

                pygame.draw.polygon(surface, color=color, points=path)

                if "phase" not in obj.__dict__:
                    continue
                a1 = obj.phase
                a2 = obj.phase + 1.2  # radians
                s1 = math.sin(a1)
                s2 = math.sin(a2)
                c1 = math.cos(a1)
                c2 = math.cos(a2)
                if s1 > 0 and s2 > 0:
                    continue
                if s1 > 0:
                    c1 = np.sign(c1)
                if s2 > 0:
                    c2 = np.sign(c2)
                white_poly = [
                    (-WHEEL_W * self.size, +WHEEL_R * c1 * self.size),
                    (+WHEEL_W * self.size, +WHEEL_R * c1 * self.size),
                    (+WHEEL_W * self.size, +WHEEL_R * c2 * self.size),
                    (-WHEEL_W * self.size, +WHEEL_R * c2 * self.size),
                ]
                white_poly = [trans * v for v in white_poly]

                white_poly = [(coords[0], coords[1]) for coords in white_poly]
                white_poly = [
                    pygame.math.Vector2(c).rotate_rad(angle) for c in white_poly
                ]
                white_poly = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1],
                    )
                    for coords in white_poly
                ]
                pygame.draw.polygon(surface, color=WHEEL_WHITE, points=white_poly)


    def _create_particle(self, point1, point2, grass):
        class Particle:
            pass

        p = Particle()
        p.color = WHEEL_COLOR if not grass else MUD_COLOR
        p.ttl = 1
        p.poly = [(point1[0], point1[1]), (point2[0], point2[1])]
        p.grass = grass
        self.particles.append(p)
        while len(self.particles) > 30:
            self.particles.pop(0)
        return p

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []
