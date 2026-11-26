# Distance map the reactive controller used on its last tick, so other nodes can show what it sees.
# Geometry is in meters in header.frame_id (the map frame).
std_msgs/Header header

# cell [i][j] is centred at (origin_x + i * resolution, origin_y + j * resolution)
float64 resolution
float64 origin_x
float64 origin_y
uint32 size_x
uint32 size_y

# distance from each cell centre to the nearest obstacle or unknown cell in meters, 0 = blocked,
# clipped at max_distance (the sensor radius); size_x * size_y values indexed [i * size_y + j]
float64 max_distance
float32[] distances

# robot pose and radius the controller used on this tick
float64 robot_x
float64 robot_y
float64 robot_theta
float64 robot_radius

# rectangular footprint (Ackermann controller), the rectangle the controller checks: meters along the heading from
# (robot_x, robot_y) to its front and to its rear, and its half width; all 0 = the footprint is the circle of robot_radius
float64 footprint_front
float64 footprint_rear
float64 footprint_half_width
# the rectangle's clearance (smallest distance along its outline, meters) and the outline point where it is
float64 footprint_clearance
float64 closest_x
float64 closest_y
