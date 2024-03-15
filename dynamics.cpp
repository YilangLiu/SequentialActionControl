#include "dynamics.h"

cartpole_dynamics::cartpole_dynamics(){
    curr_state.setZero();
    system_dynamics.setZero();
    gradient.setZero();
}

void cartpole_dynamics::set_init_state(Eigen::Vector4d x_init){
    curr_state = x_init;
}

Eigen::Vector4d cartpole_dynamics::calc_dynamics_u(Eigen::Vector4d x){
    Eigen::Vector4d g_x(0, cos(x(0))/h, 0, 1);
    return g_x;
}

Eigen::Vector4d cartpole_dynamics::step_forward(Eigen::Vector4d x, double u, double dt){
    system_dynamics(0) = x(1);
    system_dynamics(1) = gravity/h * sin(x(0)) + u * cos(x(0))/h;
    system_dynamics(2) = x(3);
    system_dynamics(3) = u;
    
    x += system_dynamics * dt;

    // if (x(0) > M_PI){}

    return x; 
}

Eigen::Vector4d cartpole_dynamics::calc_sys_dynamics(Eigen::Vector4d x, double u){
    Eigen::Vector4d syst_dyn;
    syst_dyn(0) = x(1);
    syst_dyn(1) = gravity/h * sin(x(0)) + u * cos(x(0))/h;
    syst_dyn(2) = x(3);
    syst_dyn(3) = u;
    return syst_dyn;
}

Eigen::Matrix4d cartpole_dynamics::calc_gradient_dynamics(Eigen::Vector4d x, double u){
    gradient(0,0) = 0;
    gradient(1,0) = gravity/h*cos(x(0)) - u*sin(x(0))/h;
    gradient(2,0) = 0;
    gradient(3,0) = 0;

    gradient(0,1) = 1;
    gradient(1,1) = 0;
    gradient(2,1) = 0;
    gradient(3,1) = 0;
    
    gradient(0,2) = 0;
    gradient(1,2) = 0;
    gradient(2,2) = 0;
    gradient(3,2) = 0;
    
    gradient(0,3) = 0;
    gradient(1,3) = 0;
    gradient(2,3) = 1;
    gradient(3,3) = 0;

    return gradient;
}

