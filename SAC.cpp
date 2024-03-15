#include "SAC.h"

SAC::SAC(double _dt){
    prediction_horizon = 1.5; // make 0.5s forward prediction
    dt = _dt;
    t_s = 10 * dt;
    state_vec.clear();
    rho_vec.clear();
    opti_u_2.clear();

    
    Q.setZero();

    Q_f.setZero();

    Q_f.diagonal()<< 200, 0, 50, 0;

    R = 0.3;  // step cost of R term 
    R_f = 0.3; // terminal cost of R_f term 
    x_desired <<  0.0, 0.0, 0.0, 0.0;
    alpha_d = -1000.0;
    
    planned_opti_u.clear();
    planned_time_vec.clear();
}

double SAC::constrainAngle(double x){
    // x = fmod(x + 180,360);
    // if (x < 0)
    //     x += 360;
    // return x - 180;
    x = fmod(x + M_PI,2*M_PI);
    if (x < 0){
        x += 2*M_PI;
    }
    return x - M_PI;
}


double SAC::calc_step_loss(Eigen::Vector4d x, Eigen::Vector4d x_d, double u){
    // while (x(0) > M_PI){x(0) += -2*M_PI;}
    // while (x(0) < -M_PI){x(0) += 2*M_PI;}
    x(0) = constrainAngle(x(0));

    Q.setZero();
    Q(0,0) = 200;
    Q(1,1) = 0;
    Q(2,2) = pow(x(2)/2,8);
    Q(3,3) = 50;

    return 0.5*(x-x_d).transpose()*Q*(x-x_d) + 0.5*u*R*u;
}

double SAC::calc_terminal_loss(Eigen::Vector4d x, Eigen::Vector4d x_d, double u){
    // while (x(0) > M_PI){x(0) += -2*M_PI;}
    // while (x(0) < -M_PI){x(0) += 2*M_PI;}
    x(0) = constrainAngle(x(0));
    
    return 0.5*(x-x_d).transpose()*Q_f*(x-x_d) + 0.5*u*R_f*u;
}

Eigen::Vector4d SAC::calc_terminal_grad_loss(Eigen::Vector4d x, Eigen::Vector4d x_d){
    // while (x(0) > M_PI){x(0) += -2*M_PI;}
    // while (x(0) < -M_PI){x(0) += 2*M_PI;}
    x(0) = constrainAngle(x(0));

    return Q_f*(x-x_d);
}

Eigen::Vector4d SAC::calc_grad_loss(Eigen::Vector4d x, Eigen::Vector4d x_d){
    // while (x(0) > M_PI){x(0) += -2*M_PI;}
    // while (x(0) < -M_PI){x(0) += 2*M_PI;}
    x(0) = constrainAngle(x(0));
    Q.setZero();
    Q(0,0) = 200;
    Q(1,1) = 0;
    Q(2,2) = pow(x(2)/2,8);
    // std::cout<<"Q for position is "<<Q(2,2)<<std::endl;
    Q(3,3) = 50;

    Eigen::Vector4d grad_loss;
    grad_loss(0) = Q(0,0) * (x(0) - x_d(0));
    grad_loss(1) = Q(1,1) * (x(1) - x_d(1));
    grad_loss(2) = pow(x(2)/2,8)*(x(2)-x_d(2))*Q(2,2) + 4*pow(x(2)-x_d(2),2)*pow(x(2)/2,7);
    grad_loss(3) = Q(3,3) * (x(3)- x_d(3));

    // std::cout<< "Q diagnoal at third row is "<<Q(2,2) << std::endl;
    // std::cout<< "x is "<<x<<std::endl;
    // std::cout<<"x_d is "<<x_d <<std::endl;
    // std::cout<<"grad loss is "<<grad_loss<<std::endl;
    return grad_loss;
}

Eigen::Vector4d SAC::calc_rho_dot(Eigen::Vector4d x, Eigen::Vector4d x_d, double u, Eigen::Vector4d curr_rho){ 
    return -calc_grad_loss(x,x_d) - dynamics.calc_gradient_dynamics(x,u).transpose()* curr_rho;
}

double SAC::calc_opti_u(Eigen::Vector4d x, Eigen::Vector4d rho, double J_init_value){
    if (J_init_value < 0){std::cout<<"J_INIT HAS TO BE POSITIVE"<<std::endl;}
    Eigen::Matrix<double, 4, 1> h_x = dynamics.calc_dynamics_u(x);
    Eigen::Matrix<double, 4, 1> rho_opti = rho;
    double big_A = h_x.transpose()*rho_opti * rho_opti.transpose() * h_x;
    double right_side = h_x.transpose()*rho_opti;
    alpha_d = -1000; // -1 * J_init_value;

    // std::cout<<"opti u is: "<<1/(big_A+R) * (right_side*alpha_d)<<std::endl;
    return 1/(big_A+R) * (right_side*alpha_d);
}

void SAC::calc_mode_grad(){
    mode_insert_grad.clear();
    if (opti_u_2.empty()){
        std::cout<<"opti u 2 cannot be empty! in calc_mode_grad"<<std::endl;
    }
    if (opti_u_2.size()+1 != state_vec.size()){std::cout<<"SIZE MISMATCH!"<<std::endl;}
    for (int j =0; j < opti_u_2.size(); j++){
        //calculate mode insertion gradient over prediction horizon 
        double mode_grad = rho_vec.at(j).transpose()*(dynamics.calc_sys_dynamics(state_vec.at(j),opti_u_2.at(j))- 
                                                    dynamics.calc_sys_dynamics(state_vec.at(j),default_ctrl));
        mode_insert_grad.push_back(mode_grad);
        // std::cout<<"mode insert grad "<< mode_grad<<std::endl;
    }
}

void SAC::forward_simulate(Eigen::Vector4d x_init, double t_0, double t_f){
    state_vec.clear();
    rho_vec.clear();
    opti_u_2.clear();
    time_vec.clear();
    J_init.clear();

    state_vec.push_back(x_init);
    time_vec.push_back(t_0);
    

    // default_ctrl = 0; // can change this in the future 
     // initialize cost 
    int idx = 0;
    // forward simulate x 
    for (double t = t_0+dt; t < t_f; t = t + dt){
        state_vec.push_back(dynamics.step_forward(state_vec.at(idx), default_ctrl, dt));
        // std::cout<<"The simulate X "<< idx <<" is: "<<state_vec[idx] <<std::endl;

        J_init.push_back(calc_step_loss(state_vec.at(idx), x_desired, default_ctrl));

        // std::cout<<"loss is "<<calc_step_loss(state_vec[idx], x_desired, default_ctrl)<<std::endl;
        // std::cout<<"state is "<<state_vec[idx]<<std::endl;
        idx += 1;
        time_vec.push_back(t);
    }
    J_init.push_back(calc_terminal_loss(state_vec.back(),x_desired, default_ctrl));
    // time_vec [t_0, t_f]

    if (state_vec.empty()){std::cout<<"forward simulate state_vec does not exists!!!!"<<std::endl;}

    // std::cout<<"J_init with nominal control is "<<J_init;
    // backward simulate rho
    rho_vec.push_back(calc_terminal_grad_loss(state_vec.back(), x_desired));
    for (int j = state_vec.size()-2; j>=0; j--){
        Eigen::Vector4d next_rho = rho_vec.back() - calc_rho_dot(state_vec.at(j), x_desired, default_ctrl, rho_vec.back())*dt;
        rho_vec.push_back(next_rho);
        // opti_u_2.push_back(-1/R*next_rho.transpose() * dynamics.calc_dynamics_u(state_vec[j]));
        opti_u_2.push_back(calc_opti_u(state_vec.at(j), rho_vec.back(), J_init.at(j)));
        // std::cout<<"The simulate rho "<< state_vec.size()-2-j <<" is: "<<rho_vec[state_vec.size()-2-j] <<std::endl;
    }
    std::reverse(opti_u_2.begin(), opti_u_2.end());
    std::reverse(rho_vec.begin(), rho_vec.end());
}

double SAC::re_simulate_opti_u(Eigen::Vector4d x, double opti_u, int tau_0_idx, int tau_f_idx){
    // will start 
    if (opti_u_2.empty() || J_init.empty()){
        std::cout<<"Opti control or J_init cannot be empty!"<<std::endl;
        return -1;
    }

    if (tau_0_idx > tau_f_idx){std::cout<<"must be future time idx!!!"<<std::endl;}

    double J_init_1 = 0;
    double J_new = 0;
    std::vector<Eigen::Vector4d> opt_states;
    
    opt_states.clear();
    opt_states.push_back(x);
    
    J_new += calc_step_loss(x, x_desired, opti_u);
    J_init_1 += J_init.at(tau_0_idx);
    
    // std::cout<<"J_init size is "<<J_init.size()<<" re sim 1 with tau_0_idx is "<<tau_0_idx<<" and tau_f_idx is "<<tau_f_idx<<std::endl;
    // std::cout<<"J_init with tau_0_idx is "<<J_init.at(tau_0_idx)<<" at tau_f_idx is"<<J_init.at(tau_f_idx)<<std::endl;
    // double J_init_value = std::accumulate(J_init.begin()+tau_0_idx, J_init.begin()+tau_f_idx, 0);
    // std::cout<<"re sim 2"<<std::endl;
    
    for (int k = tau_0_idx+1; k<= tau_f_idx; k++){    
        opt_states.push_back(dynamics.step_forward(opt_states.back(),opti_u,dt));
        J_new +=calc_step_loss(opt_states.back(), x_desired, opti_u);
        J_init_1 += J_init.at(k);
    }

    // std::cout<<"J_new_value: "<<J_new <<" J_init_value: "<<J_init_1<<std::endl;
    return J_new - J_init_1;
}

void SAC::erase_vec_before_idx(std::vector<double>& input, int idx){
    for (int i =0; i< idx; i++){
        input.erase(input.begin());
    }
}

int SAC::IndexOfMinimumElement(const std::vector<double>& input){
    if (input.empty()){
        std::cout<<"##############################################"<<std::endl;
        std::cout<<"Input cannot be empty!"<<std::endl;
        std::cout<<"##############################################"<<std::endl;
        return -1;
    } 
        
    auto ptrMinElement = std::min_element(input.begin(), input.end()); 
    return std::distance(input.begin(), ptrMinElement);
}

void SAC::calc_ctrl_loss(){
    ctrl_loss.clear();
    if (mode_insert_grad.empty()){
        std::cout<<"mode insertion gradient cannot be empty for calc_ctrl_loss"<<std::endl;
    }
    for (int i = 0; i < opti_u_2.size(); i++){
        double loss = std::abs(opti_u_2.at(i)) + mode_insert_grad.at(i) + pow(time_vec.at(i)-time_vec.front(), 1.6);
        // std::cout<<"mode insert grad at time"<<time_vec[i]<<" is "<<mode_insert_grad[i]<<" ctrl_loss at idx: "<<i<<" is "<<loss<<std::endl;
        ctrl_loss.push_back(loss);
    }
}

void SAC::update_plan(Eigen::Vector4d x, double curr_time){
    // forward_simulate(x, curr_time, curr_time + prediction_horizon);
    calc_mode_grad();
    calc_ctrl_loss();
    opti_time_idx = IndexOfMinimumElement(ctrl_loss);
    opti_time = time_vec[opti_time_idx];
    // std::cout<<"At time: "<<curr_time<<" replan time: "<<curr_time + t_s<<" opti time is "<< opti_time<<std::endl;
}

int SAC::IndexOfElement(const std::vector<double>& input, double value){
    auto const it = std::lower_bound(input.begin(), input.end(), value);

    if (it == input.end()){return input.size()-1;}
    return std::distance(input.begin(), it);
}

void SAC::line_search(){
    double J_diff = std::numeric_limits<double>::infinity();
    int line_search_power = 0;
    int k_max = 20;
    double omega = 0.8;
    double init_ctrl_duration = 1.0;
    int t_end_idx;
    double lambda;
    while ((J_diff>delta_J) && (line_search_power < k_max)){
        lambda = pow(omega, line_search_power) * init_ctrl_duration;
        // std::cout<<"size of the time is "<<time_vec.size()<<" which should be "<< prediction_horizon/dt<<std::endl;
        // std::cout<<"At this opti time: "<<opti_time<<" with idx: "<<opti_time_idx<<std::endl;
        // std::cout<<"in time vec: "<<time_vec[opti_time_idx]<<std::endl;
        // std::cout<<"lambda is: "<<lambda<<std::endl;

        
        t_end_idx = IndexOfElement(time_vec, opti_time + lambda);
        // std::cout<<"t_end_idx = "<<t_end_idx;
        // std::cout<<"time_vec last number is"<< time_vec[time_vec.size()-1]<<std::endl;
        // std::cout<<"opti_time is "<<opti_time <<" with lambda is "<<lambda<<std::endl;
        // std::cout<<"t_end_idx is: "<<t_end_idx<<std::endl;
        // std::cout<<"with lambda: "<<lambda<<" t_end_idx: "<<t_end_idx<<std::endl;
        J_diff = re_simulate_opti_u(state_vec.at(opti_time_idx), opti_u_2.at(opti_time_idx), opti_time_idx, t_end_idx);
        // std::cout<<"with lambda: "<<lambda<<" t_end_idx: "<<t_end_idx<<std::endl;
        // std::cout<<"J_new is: "<<J_new<<std::endl;
        line_search_power++;        
    }
    // std::cout<<"J_diff is "<<J_diff<< " k is "<<line_search_power<<" interval is "<<lambda<<std::endl;
    plan_time_end_idx = t_end_idx;
}

void SAC::update_time_vec_ctrl_plan(double curr_time){
    if (planned_time_vec.empty()){
        std::cout<<"Initialize time vec. It should only appear onece!"<<std::endl;
        planned_time_vec = time_vec;
        
                // assemble u1(t) for future time steps

        planned_time_vec = time_vec;
        planned_opti_u.resize(time_vec.size());

        double optimal_ctrl = opti_u_2.at(opti_time_idx);
        if (optimal_ctrl > 4.8){optimal_ctrl=4.8;}
        if (optimal_ctrl <-4.8){optimal_ctrl=-4.8;}

        int plan_idx = 0;
        // std::cout<<"At t_0: "<< t_0 <<" to t_f: "<<t_f<<" opti_time: "<<opti_time<< " opti u: "<<optimal_ctrl<<" t_0+t_s: "<<t_0 + t_s<<std::endl;
        for (double n = t_0; n < t_f; n= n+dt){
            if (n>= opti_time && n<= time_vec.at(plan_time_end_idx)){ //  && n >= t_0 + calc_time && n <= t_0+t_s+calc_time
                planned_opti_u[plan_idx] = optimal_ctrl;
                // std::cout<<"this part is used "<<std::endl;
            } else{
                // std::cout<<"the size of the time vec is: "<<time_vec.size()<<" here is plan_idx is "<<plan_idx<<std::endl;
                planned_opti_u[plan_idx] = default_ctrl;
            }
            plan_idx++;
        }
    } else{
        // remove time that is before the curr time 
        int erease_idx = IndexOfElement(planned_time_vec, curr_time);
        erase_vec_before_idx(planned_time_vec, erease_idx);
        erase_vec_before_idx(planned_opti_u, erease_idx);

        // extend time from the current simulation
        int add_idx = IndexOfElement(time_vec, planned_time_vec.back());
        for (int j =add_idx; j< time_vec.size(); j++){
            planned_time_vec.push_back(time_vec[j]);
            planned_opti_u.push_back(default_ctrl);
        }

        // update the control strategy 
        int plan_idx = 0;
        double optimal_ctrl = opti_u_2.at(opti_time_idx);
        // std::cout<<"optimal ctrl is "<<optimal_ctrl<<std::endl;
        if (optimal_ctrl > 4.8){optimal_ctrl=4.8;}
        if (optimal_ctrl <-4.8){optimal_ctrl=-4.8;}

        for (double n = t_0; n < t_f; n= n+dt){
            if (n>= opti_time && n<= time_vec.at(plan_time_end_idx)){ //  && n >= t_0 + calc_time && n <= t_0+t_s+calc_time
                planned_opti_u.at(plan_idx) = optimal_ctrl;
            } 
            plan_idx++;
        }
    }

}

void SAC::sac_update_ctrl_plan(Eigen::Vector4d x, double curr_time){
    t_0  = curr_time;
    t_f = curr_time + prediction_horizon;

    forward_simulate(x, t_0, t_f);
    update_plan(x, curr_time); // find for time tau 
    line_search();
    update_time_vec_ctrl_plan(curr_time);
}

bool SAC::sac_test(Eigen::Vector4d x){
    double t_0 = 0.0;
    forward_simulate(x, t_0, t_0+prediction_horizon); // do initialize to find first best plans for actions
    update_plan(x, t_0);
    line_search();

    return true;
}

double SAC::get_sac_ctrl(double curr_time){
    if (planned_time_vec.empty()){return default_ctrl;}
    if(planned_opti_u.empty()){return default_ctrl;}

    int u_idx = IndexOfElement(planned_time_vec, curr_time);
    // std::cout<<"at idx "<<u_idx<<"ctrl is "<<planned_opti_u.at(u_idx)<<std::endl;
    return planned_opti_u.at(u_idx);
}