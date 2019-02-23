/*Pablo Ibáñez Freire. pablo.ibannez@uam.es

Steepest descent algorithm for energy minimization. 
The energy is minimized until one of the requirements is satisfied;
either the maximum force of the system is less than the indicated force
or a certain number of steps are performed.

PARAMETERS:

h:maximum displacement
epsilon:max force
maxSteps:max number of steps

ALGHORITHM:

1. Compute system energy (Un) and the max force (Fmax)
2. New postions are computed:

    ri+1 = ri+(Fi/Fmax)*h
3. Compute new system energy (Un+1)
4. Check new state
    if(Un+1 <  Un) Accept change and h*=1.2
    if(Un+1 >= Un) Reject change and h*=0.2

USAGE:

//Declaration
SteepestDescent::Parameters STpar;
STpar.h = 0.1;
STpar.epsilon = 1;
STpar.maxSteps = 1000;

auto st = make_shared<SteepestDescent>(pd, pg, sys, STpar);
st->addInteractor(...);

//This integrator can be used in two ways. 
//In the first case a minimization step is performed 
//,it is not checked if the conditions of force 
//and maximum number of steps have been reached.

st->forwarTime() 

//The second way is to use the function "minimizationStep"
//that works in a similar way to the previous case
//,but returns "false" if any of the requirements are satisfied 
//or "true" otherwise. 
//Then to perform the minimization process the function can be used in the following way: 

while(st->minimizationStep());

///////////////////////////////////////////////////////////////////////

More information about the different methods can be found in "SteepestDescent.cu".

*/
#ifndef STEEPEST_DESCENT_CUH
#define STEEPEST_DESCENT_CUH

#include "Integrator/Integrator.cuh"
#include <thrust/device_vector.h>
#include "third_party/cub/cub.cuh"
#include <limits>
namespace uammd{
  
  class SteepestDescent: public Integrator{
      
    real h; //maximum displacement
    real epsilon; //force tolerance
    int maxSteps; //max number of steps
    
    int steps; //steps counter
    
    cudaStream_t stream;
    
    //cub reduction variables
    
    //energy sum variables
    real*  totalEnergy;
    void*    cubTempStorageSum = NULL;
    size_t   cubTempStorageSizeSum = 0;
    
    //max force searching variables
    real* maxForce;
    void*    cubTempStorageMax = NULL;
    size_t   cubTempStorageSizeMax = 0;
    
    void setUpCubReduction();
    
  public:
  
    struct Parameters{
      real h = 0.01;
      real epsilon = 0;
      int maxSteps = std::numeric_limits<int>::max();
      bool is2D = false;
    };
    
    SteepestDescent(shared_ptr<ParticleData> pd,
		    shared_ptr<ParticleGroup> pg,
		    shared_ptr<System> sys,
		    Parameters par);
    ~SteepestDescent();

    virtual void forwardTime() override;
    bool minimizationStep();
    
    virtual real sumEnergy() override;
    real computeMaxForce();
  };

}

#include"SteepestDescent.cu"
#endif
