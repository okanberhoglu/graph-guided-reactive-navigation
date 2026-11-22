from dataclasses import dataclass

@dataclass
class RobotConfig:

    v_min = -2.0
    v_max = 8.0
    w_min = -4.0
    w_max = 4.0

    robot_radius = 2.0
    sensor_radius = 20


@dataclass
class CarConfig:

    car_front = 24.4       
    car_rear = 24.5        
    car_half_width = 9.2

    lf = 14.3
    lr = 14.3

    sensor_radius = 90
