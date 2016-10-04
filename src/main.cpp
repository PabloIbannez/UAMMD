/*
Raul P. Pealez 2016. Launcher for Driver class.

Checkout Driver class to understand usage of this branch as a module.


NOTES:
The idea is to use either Integrator or Interactor in another project as a module.

Once initialized this classes will perform a single task very fast as black boxes:

Integrator uploads the positions according to the velocities, forces and current positions.
Interactor computes the pair forces using the current positions according to the selected potential

The idea is for Integrator to control the positions and velocities and for Interactor to control the forces. Communicating each variable when needed. So if you need the vel. in the force computing you can pass it when computing the force and modify the force function accordingly.
-------
Current benchmark:
GTX980 CUDA-7.5
N = 2^20
L = 128
dt = 0.001f
1e4 steps
PairForces with rcut = 2.5 and no energy measure
TwoStepVelverlet, no writing to disk, Tini = 0.03
Starting in a cubicLattice
---------------------HIGHSCORE-----------------------
Number of cells: 51 51 51; Total cells: 132651
Initializing...	DONE!!
Initialization time: 0.15172s
Computing step: 10000   
Mean step time: 127.33 FPS

Total time: 78.535s

real	1m19.039s
user	0m53.772s
sys	0m25.212s
---------------------------------------------------
TODO:
100- Read and construct simulation configuration from script
*/
#include<iomanip>
#include"globals/globals.h"
#include"Driver/SimulationConfig.h"


/*Declaration of extern variables in globals.h*/
GlobalConfig gcnf;
Vector4 pos, force;
Vector3 vel;


int main(int argc, char *argv[]){
  Timer tim;
  tim.tic();

  size_t size;
  cudaDeviceGetLimit(&size,cudaLimitPrintfFifoSize);
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size*1000);

  /*The simulation handler*/
  SimulationConfig psystem(argc, argv);

  cerr<<"Initialization time: "<<setprecision(5)<<tim.toc()<<"s"<<endl;

  
  tim.tic();  
  psystem.run();
  
  float total_time = tim.toc();
  
  cerr<<"\nMean step time: "<<setprecision(5)<<(float)gcnf.nsteps/total_time<<" FPS"<<endl;
  cerr<<"\nTotal time: "<<setprecision(5)<<total_time<<"s"<<endl;
  
  if(gcnf.print_steps>0) psystem.write(true);

  cudaDeviceSynchronize();
  
  /*Free the global arrays manually*/
  pos.freeMem();
  force.freeMem();
  vel.freeMem();

  return 0;
}


