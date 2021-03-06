#include "ros/ros.h"
#include "pi3_irobot_pkg/rl_vel.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "controller");

  ros::NodeHandle control;
  
  ros::Publisher control_pub = control.advertise<pi3_irobot_pkg::rl_vel>("control", 1);

  ros::Rate loop_rate(20);
  
  int count = 0;
  while (ros::ok())
  {
    pi3_irobot_pkg::rl_vel control_msg;
    control_msg.r_vel = .25;
    control_msg.l_vel = .25;
    control_pub.publish(control_msg);
    ros::spinOnce();
    loop_rate.sleep();
    ++count;
  }
  return 0;
}
