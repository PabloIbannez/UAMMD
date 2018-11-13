/*Raul P. Pelaez 2018. Inertial Coupling Method example, ideal blobs diffusing.

See Hydro/ICM.cuh for more info.
 */


#include"uammd.cuh"
#include"Integrator/Hydro/ICM.cuh"
#include"utils/InitialConditions.cuh"
#include<fstream>

using namespace std;
using namespace uammd;



int main(int argc, char *argv[]){

  if(argc==1){
    std::cerr<<"Run with: ./icm 14 32 0.01 200000 50 1 1 1"<<std::endl;
    exit(1);
  }

  int N = pow(2,atoi(argv[1]));
  
  auto sys = make_shared<System>();

  ullint seed = 0xf31337Bada55D00dULL^time(NULL);
  sys->rng().setSeed(seed);

  auto pd = make_shared<ParticleData>(N, sys);

  Box box(std::stod(argv[2]));
  {
    auto pos = pd->getPos(access::location::cpu, access::mode::write);

    real boxfac = std::stod(argv[8]);
    auto initial =  initLattice(box.boxSize*boxfac, N, sc);
    fori(0,N){
      pos.raw()[i] = make_real4(sys->rng().uniform3(-box.boxSize.x*boxfac*0.5, box.boxSize.x*boxfac*0.5), 0) ;//initial[i];
      pos.raw()[i].w = 0;
    }    
  }
  

  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  
  ofstream out("kk");
  
  double hydrodynamicRadius =  std::stod(argv[7]);
  
  Hydro::ICM::Parameters par;
  par.temperature = std::stod(argv[6]);
  par.viscosity = 1.0;
  par.density = 1.0;
  par.hydrodynamicRadius =  hydrodynamicRadius;
  par.dt = std::stod(argv[3]);
  par.box = box;
  auto bdhi = make_shared<Hydro::ICM>(pd, sys, par);
   
  sys->log<System::MESSAGE>("RUNNING!!!");

  Timer tim;
  tim.tic();
  int nsteps = std::atoi(argv[4]);
  int printSteps = std::atoi(argv[5]);
  //Run the simulation
  forj(0,nsteps){
    bdhi->forwardTime();
    //Write results
    if(j%printSteps==0)
    {
      sys->log<System::DEBUG1>("[System] Writing to disk...");

      auto pos = pd->getPos(access::location::cpu, access::mode::read);
      
      out<<"#Lx="<<box.boxSize.x*0.5<<";Ly="<<box.boxSize.x*0.5<<";Lz="<<box.boxSize.x*0.5<<";\n";
      real3 p;
      fori(0,N){
	real4 pc = pos.raw()[i];
	p = make_real3(pc);
	out<<p<<"\n";
      }
      out<<flush;
    }
  }
  
  auto totalTime = tim.toc();
  sys->log<System::MESSAGE>("mean FPS: %.2f", nsteps/totalTime);
  //sys->finish() will ensure a smooth termination of any UAMMD module.
  sys->finish();

  return 0;
}