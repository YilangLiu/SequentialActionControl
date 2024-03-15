#ifndef SAC_CONTROL
#define SAC_CONTROL

#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include "dynamics.h"
#include <algorithm>
#include <numeric>

class SAC{
public:
    SAC(double _dt);
    ~SAC()=default;
    void set_init_state(Eigen::Vector4d x_init);
    void forward_simulate(Eigen::Vector4d x_init, double t_0, double t_f);
    double calc_step_loss(Eigen::Vector4d x, Eigen::Vector4d x_d, double u);
    double calc_terminal_loss(Eigen::Vector4d x, Eigen::Vector4d x_d, double u);
    double calc_opti_u(Eigen::Vector4d x, Eigen::Vector4d rho, double J_init_value);
    double re_simulate_opti_u(Eigen::Vector4d x, double opti_u, int tau_0, int tau_f);
    void calc_ctrl_loss();
    void sac_update_ctrl_plan(Eigen::Vector4d x, double curr_time);
    bool sac_test(Eigen::Vector4d x);
    double get_sac_ctrl(double curr_time);
    double constrainAngle(double x);

    Eigen::Vector4d calc_terminal_grad_loss(Eigen::Vector4d x, Eigen::Vector4d x_d);
    Eigen::Vector4d calc_grad_loss(Eigen::Vector4d x, Eigen::Vector4d x_d);
    Eigen::Vector4d calc_rho_dot(Eigen::Vector4d x, Eigen::Vector4d x_d, double u, Eigen::Vector4d curr_rho);
    void calc_mode_grad();
    void update_plan(Eigen::Vector4d x, double curr_time);
    void line_search();
    int IndexOfMinimumElement(const std::vector<double>& input);
    int IndexOfElement(const std::vector<double>& input, double value);
    void update_time_vec_ctrl_plan(double curr_time);
    void erase_vec_before_idx(std::vector<double>& input, int idx);

    Eigen::Matrix4d Q;
    Eigen::Matrix4d Q_f;
    Eigen::Vector4d x_desired;
    std::vector<Eigen::Vector4d> state_vec;
    std::vector<Eigen::Vector4d> rho_vec;
    std::vector<double> opti_u_2;
    std::vector<double> mode_insert_grad;
    std::vector<double> time_vec;
    std::vector<double> ctrl_loss;
    std::vector<double> J_init;
    std::vector<double> planned_opti_u;
    std::vector<double> planned_time_vec;

    double prediction_horizon;
    double dt;
    double R;
    double R_f;
    double t_0;
    double t_f;

    double alpha_d;
    int calc_time_idx = 10;
    double calc_time = 0;

    double opti_time;
    int opti_time_idx;
    double opti_time_interval;
    double delta_J = -100;
    double default_ctrl=0.0;
    int plan_time_end_idx;
    double t_s;
    

    cartpole_dynamics dynamics;
};
#endif