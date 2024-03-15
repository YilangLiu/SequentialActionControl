#ifndef CARTPOLE_DYNAMICS
#define CARTPOLE_DYNAMICS

#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Dense>

class cartpole_dynamics{
public:
    cartpole_dynamics();
    ~cartpole_dynamics()=default;
    void set_init_state(Eigen::Vector4d x_init);
    Eigen::Matrix4d calc_gradient_dynamics(Eigen::Vector4d x, double u);
    Eigen::Vector4d step_forward(Eigen::Vector4d x, double u, double dt);
    Eigen::Vector4d calc_dynamics_u(Eigen::Vector4d x);
    Eigen::Vector4d calc_sys_dynamics(Eigen::Vector4d x, double u);


    double h = 2;
    double gravity = 9.81;

    Eigen::Vector4d curr_state;
    Eigen::Vector4d system_dynamics;
    Eigen::Matrix<double, 4, 4> gradient;
};
#endif