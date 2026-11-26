# Route nodes the controller should keep the robot inside: the node the robot is in,
# then the route nodes after it up to the one holding the current temp goal.
# Empty when the robot is outside every route node.
# Geometry is in meters in header.frame_id.
std_msgs/Header header
RectRegion[] regions
