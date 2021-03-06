#include "ros/ros.h"
#include "pi3_irobot_pkg/irobot_state.h"
#include "pi3_irobot_pkg/rl_vel.h"
#include <math.h>

#define STATE_DIM 5
#define CONTROL_DIM 2
#define HZ 20

//==============================
//class definition
class Simulator
{

private:
  //Defines the current state of the robot
  float dt;
  ros::NodeHandle n;
  ros::Subscriber sub;

public:
  Simulator(float* init_state);
  void dynamics();
  void controlCallback(const pi3_irobot_pkg::rl_vel::ConstPtr& control_msg);
  void init_subscriber();
  float s[STATE_DIM];
  float u[CONTROL_DIM];
};


Simulator::Simulator(float* init_state) {
  int i;
  for (i = 0; i < STATE_DIM; i++) {
    s[i] = init_state[i];
  }
  for (i = 0; i < CONTROL_DIM; i++) {
    u[i] = 0;
  }
  dt = 1.0/(1.0*HZ);
}

void Simulator::controlCallback(const pi3_irobot_pkg::rl_vel::ConstPtr& control_msg) {
  u[0] = control_msg->r_vel;
  u[1] = control_msg->l_vel;
  ROS_INFO("I heard: (%f, %f) u = (%f, %f), ", control_msg->r_vel, control_msg->l_vel, u[0], u[1]);
}

void Simulator::init_subscriber() {
  sub = n.subscribe("control", 1, &Simulator::controlCallback, this);
}

void Simulator::dynamics() {
  ROS_INFO("u is (%f, %f)", u[0], u[1]);
  s[0] += dt*(s[3] + s[4])/2.0*cos(s[2]);
  s[1] += dt*(s[3] + s[4])/2.0*sin(s[2]);
  s[2] += dt*(s[3] - s[4])/.258;
  s[3] += dt*(u[0] - s[3]);
  s[4] += dt*(u[1] - s[4]);
  if (s[0] > 10.0) {
    s[0] = 10.0;
  }
  else if (s[0] < -10.0) {
    s[0] = -10.0;
  }
  if (s[1] > 10.0) {
    s[1] = 10.0;
  }
  else if (s[1] < -10.0) {
    s[1] = -10.0;
  }
  if (s[2] > 3.14) {
    s[2] = -3.14;
  }
  else if (s[2] < -3.14) {
    s[2] = 3.14;
  }
  if (s[3] > .5) {
    s[3] = .5;
  }
  else if (s[3] < -.5) {
    s[3] = -.5;
  }
  if (s[4] > .5) {
    s[4] = .5;
  }
  else if (s[4] < -.5) {
    s[4] = -.5;
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "simulator");

  ros::NodeHandle n1;
  ros::Rate loop_rate(HZ);
  ros::Publisher state_pub = n1.advertise<pi3_irobot_pkg::irobot_state>("state", 1);
  float init_state[] = {0,0,0,0,0};
  Simulator simmer(init_state);
  int count = 0;
  simmer.init_subscriber();
  while (ros::ok())
  {
    simmer.dynamics();
    pi3_irobot_pkg::irobot_state state_msg;
    state_msg.x = simmer.s[0];
    state_msg.y = simmer.s[1];
    state_msg.theta = simmer.s[2];
    state_msg.r_vel = simmer.s[3];
    state_msg.l_vel = simmer.s[4];
    state_pub.publish(state_msg);
    ros::spinOnce();
    loop_rate.sleep();
    ++count;
  }
  return 0;
}
