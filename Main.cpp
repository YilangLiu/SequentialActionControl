#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <thread>
#include "dynamics.h"
#include "SAC.h"
#include "Params.h"

int main(){
    Eigen::Vector4d x;
    Eigen::Vector4d x_desired;
    x << M_PI, 0.0, 0.0, 0.0;
    x_desired << 0.0,0.0,0.0,0.0;
    int iterations = 100;
    double u = 0;
    double dt = 0.001;
    cartpole_dynamics cartpole;
    SAC sac(dt);
    
    bool ctrl_finished = false;
    double curr_time = 0;
    int sac_control_duration = SAC_UPDATE_FREQUENCY;
    int robo_sim_duration = SIM_UPDATE_FREQUENCY;
    const char delimiter = ',';
    double t_s = (10) * dt;

    std::chrono::milliseconds sac_duration(sac_control_duration);
    std::chrono::milliseconds sim_duration(robo_sim_duration);

    std::thread compute_sac_control([&](){
        
        
        // std::cout<<"initial opti time is "<< sac.opti_time<<std::endl;
        // std::cout<<"initial opti time idx is "<< sac.opti_time_idx<<std::endl;
        // std::cout<<"initial opti time duration is "<< sac.opti_time_interval<<std::endl;
        // for (int i =0; i< sac.opti_u_2.size(); i++){
        //     std::cout<<"initial opti contrl in idx "<< i <<" is "<< sac.opti_u_2[i]<<std::endl;
        // }
        

        // first initialize the control 
        auto start = std::chrono::high_resolution_clock::now();
        auto prev = std::chrono::high_resolution_clock::now();
        auto now  = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (prev-start);
        Eigen::Vector4d prev_x;


        while (curr_time <= 20.0){
            now = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds> (now-prev);
            prev = now;
            double t_0 = curr_time;
            
            // std::cout<<"curr time is "<<curr_time<<std::endl;
            sac.sac_update_ctrl_plan(x, t_0);
            prev_x = x;
            // std::cout<<"t_0 is "<<t_0<<std::endl;

            while ((curr_time < t_0 + t_s)){
                continue;
            }
            // std::cout<<"curr time is "<<curr_time<<std::endl;

            // std::this_thread::sleep_for(std::chrono::milliseconds(sac_duration));   
        }

        ctrl_finished = true;
        std::cout<<"reached end "<<std::endl;
        // test = false;
        // std::ofstream modefile;
        // modefile.open("./modes.csv");

        // sac.forward_simulate(x_init, 0, 3.0);
        // std::vector<Eigen::Vector4d> state_list;
        // std::vector<double> loss_list;
        // std::vector<double> mode_insert_grad;
        // state_list.push_back(x_init);
        
        // for(int i = 0; i < sac.opti_u_2.size(); i++){
        //     Eigen::Vector4d next_state = cartpole.step_forward(state_list.back(),sac.opti_u_2[i],dt);
        //     state_list.push_back(next_state);
        // }
        
        // for (int j =0; j < sac.opti_u_2.size(); j++){
        //     //calculate mode insertion gradient over prediction horizon 
        //     double mode_grad = sac.rho_vec[j].transpose()*(cartpole.calc_sys_dynamics(sac.state_vec[j],sac.opti_u_2[j])- 
        //                                                     cartpole.calc_sys_dynamics(sac.state_vec[j],0.0));
        //     // std::cout<<"mode insertion grad: "<<mode_grad<<std::endl;
        //     modefile << mode_grad<<"\n";
        // }
        // modefile.close();
    });

    std::thread robot_simulation([&](){
        auto start = std::chrono::high_resolution_clock::now();
        auto prev = std::chrono::high_resolution_clock::now();
        auto now  = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (prev-start);
        double u;
        std::ofstream myfile, timefile;
        myfile.open("./states.csv");
        timefile.open("./time.csv");
        while (!ctrl_finished){
            now = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds> (now-prev);
            prev = now;
            // std::cout<<"curr time is "<<curr_time<<std::endl;
            u = sac.get_sac_ctrl(curr_time);
            // u = 0.0;
            x = cartpole.step_forward(x, u ,duration.count()/1000.0);

            curr_time = curr_time + duration.count()/1000.0;

            // std::cout<<"At time: "<<curr_time<<", angle is "<<x[0]<<" position is "<<x[2]<<std::endl;
            // std::cout<<"wait three seconds"<<std::endl;
            
            timefile<< curr_time<<"\n";
            myfile << std::fixed << std::setprecision(5) 
                                << x(0) <<delimiter 
                                << x(1) <<delimiter
                                << x(2) <<delimiter
                                << x(3) <<delimiter
                                << u << delimiter <<"\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(sim_duration));
            // std::cout<<"angle is "<<x(0)<<std::endl;
            // std::cout<<"position is "<<x(2)<<std::endl;
            // std::cout<<"control is "<<u<<std::endl;
            // std::cout<< "simulation time is "<<curr_time <<std::endl;
        }
        myfile.close();
        timefile.close();
        std::cout<<"finished this with curr time is "<<curr_time<<std::endl;
    });

    compute_sac_control.join();
    robot_simulation.join();
    // double cost = 0;
    // //calculate cost for the first step SAC 
    // for (int j = 0; j <sac.opti_u_2.size(); j ++){
    //     cost = sac.calc_step_loss(state_list[j], x_desired, sac.opti_u_2[j]);
    //     // std::cout<<"stage cost is : "<<cost <<std::endl;
    // }
    // std::cout<<"state size: "<<state_list.size()<< " control size: "<< sac.opti_u_2.size()<<std::endl;
    // std::cout<<"cost for the next time step is: "<<cost<<std::endl;
}