#include <boost/bind.hpp>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

#include <stdio.h>
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>

#include <iostream>
#include <string>
#include <vector>
#include <random>

// These need to be pulled out to parameters...
const float WHEEL_RAD = 0.1016; // meters
const float WHEELBASE = 0.466725; // meters
const float TRACK = 0.2667; // meters

namespace gazebo
{
  class Mecanum : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      // Store the pointer to the model
      this->model = _parent;

      integral_part = {0,0,0};

      mbl = 0;
      mbr = 0;
      mfl = 0;
      mfr = 0;
      
      ignition::math::Pose3d pose = this->model->WorldPose();
      integral_part.at(0) = pose.Pos().X();;
      integral_part.at(1) = pose.Pos().Y();;

      mRosnode.reset(new ros::NodeHandle(""));

      x_prev_vel = 0.001;
      y_prev_vel = 0.001;

      mfl_sub = mRosnode->subscribe("omni_control/front_left_wheel_joint/command",1,&Mecanum::fl_cb, this);
      mfr_sub = mRosnode->subscribe("omni_control/front_right_wheel_joint/command",1,&Mecanum::fr_cb, this);
      mbl_sub = mRosnode->subscribe("omni_control/back_left_wheel_joint/command",1,&Mecanum::bl_cb, this);
      mbr_sub = mRosnode->subscribe("omni_control/back_right_wheel_joint/command",1,&Mecanum::br_cb, this);

      // clock_sub = mRosnode->subscribe("clock", 1, &Mecanum::clock_cb, this);

      // odom_pub = mRosnode->advertise<geometry_msgs::Pose>("odom", 1);
      odom_pub = mRosnode->advertise<nav_msgs::Odometry>("odom", 10);

      this_iteration = ros::Time::now().toSec();

      // Listen to the update event. This event is broadcast every
      // simulation iteration.
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        boost::bind(&Mecanum::UpdateChild, this, _1));
      }

      public: void UpdateChild(const common::UpdateInfo & /*_info*/)
      {
        float r = WHEEL_RAD;
        float l1 = TRACK; // left -> right
        float l2 = WHEELBASE; // front -> back
        float l = 1.0/(2*(l1+l2));
        float x = r*(mfl/4.0 + mfr/4.0 + mbl/4.0 + mbr/4.0);
        float y = r*(-mfl/4.0 + mfr/4.0 + mbl/4.0 - mbr/4.0);
        
        float rot = r*(-l*mfl + l*mfr - l*mbl + l*mbr);
        ignition::math::Pose3d pose = this->model->WorldPose();

        // std::default_random_engine generator();
        // std::normal_distribution<double> distribution(2.0, 0.5);
        prev_iteration = this_iteration;
        this_iteration = ros::Time::now().toSec();
        delta_time =  this_iteration - prev_iteration;


        // odom start

        float yaw = pose.Rot().Yaw();
        float pitch  = 0.0;
        
        float x_velocity = x * cosf(yaw) - y * sinf(yaw);
        float y_velocity = y * cosf(yaw) + x * sinf(yaw);
        
        // float x_accel = (x_velocity - x_prev_vel) / delta_time;
        // float y_accel = (y_velocity - y_prev_vel) / delta_time;

        //noise gen start
        float mean = 0.0;
        float sigma_x = 0.02*x_velocity;
        float sigma_y = 0.02*y_velocity;

        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<> dx{mean,sigma_x};
        std::normal_distribution<> dy{mean,sigma_y};

        double noise_x = dx(gen);
        double noise_y = dy(gen);
        // end noise gen
        x_prev_vel = x_velocity;
        y_prev_vel = y_velocity;

        integral_part[0] = integral_part[0] + delta_time * x_velocity + noise_x;
        integral_part[1] = integral_part[1] + delta_time * y_velocity + noise_y;
        integral_part[2] = integral_part[2] + delta_time * rot;


        odom_quat = tf::createQuaternionMsgFromYaw(integral_part.at(2));
        //publish the odometry message over ROS
        nav_msgs::Odometry odom;
        odom.header.stamp = ros::Time::now();
        odom.header.frame_id = "odom";
        //set the position
        odom.pose.pose.position.x = integral_part.at(0);
        odom.pose.pose.position.y = integral_part.at(1);
        odom.pose.pose.position.z = 0.1;
        odom.pose.pose.orientation = odom_quat;

        //set the velocity
        odom.child_frame_id = "base_link";
        odom.twist.twist.linear.x = x_velocity;
        odom.twist.twist.linear.y = y_velocity;
        odom.twist.twist.angular.z = rot;

        //publish the message
        odom_pub.publish(odom);
        
        //odom end
        this->model->SetLinearVel(ignition::math::Vector3d(x * cosf(yaw) - y * sinf(yaw), y * cosf(yaw) + x * sinf(yaw), pitch));
        this->model->SetAngularVel(ignition::math::Vector3d(pitch, pitch, rot));
      }

      public: void fl_cb(const std_msgs::Float64::ConstPtr & cmd_msg)
      {
        mfl = cmd_msg->data;
      }
      public: void fr_cb(const std_msgs::Float64::ConstPtr & cmd_msg)
      {
        mfr = cmd_msg->data;
      }
      public: void bl_cb(const std_msgs::Float64::ConstPtr & cmd_msg)
      {
        mbl = cmd_msg->data;
      }
      public: void br_cb(const std_msgs::Float64::ConstPtr & cmd_msg)
      {
        mbr = cmd_msg->data;
      }

      private:
        // Pointer to the model
        physics::ModelPtr model;

        // Pointer to the update event connection
        event::ConnectionPtr updateConnection;

        boost::shared_ptr<ros::NodeHandle> mRosnode;

        float mfl;
        float mfr;
        float mbl;
        float mbr;

        std::vector<float> integral_part;

        geometry_msgs::Quaternion odom_quat;

        float noise;
        // time in seconds
        float prev_iteration;
        float this_iteration; 
        float delta_time;
        // velocity previous
        float x_prev_vel;
        float y_prev_vel;

        ros::Subscriber mfl_sub;
        ros::Subscriber mfr_sub;
        ros::Subscriber mbl_sub;
        ros::Subscriber mbr_sub;
        ros::Subscriber clock_sub;

        ros::Publisher odom_pub;
      };

      // Register this plugin with the simulator
      GZ_REGISTER_MODEL_PLUGIN(Mecanum)
    }
