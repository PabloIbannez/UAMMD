/*Pablo Ibáñez Freire. pablo.ibannez@uam.es*/
#include "SteepestDescent.cuh"

namespace uammd{
    
    namespace SteepestDescent_ns{
      
      //Functor used for energy sum
      struct energyFunctor{
	ParticleGroup::IndexIterator groupIterator;
	
	real* energy;
	
	energyFunctor(ParticleGroup::IndexIterator groupIterator,real* energy):groupIterator(groupIterator),energy(energy){}

	__host__ __device__ __forceinline__
	real operator()(const int &index) const {

	    int i = groupIterator[index];
	    return energy[i]; 
	}
      };
      
      //Functor used for finding max force
      struct forceFunctor{
	ParticleGroup::IndexIterator groupIterator;
	
	real4* force;
	
	forceFunctor(ParticleGroup::IndexIterator groupIterator,real4* force):groupIterator(groupIterator),force(force){}

	__host__ __device__ __forceinline__
	real operator()(const int &index) const {

	    int i = groupIterator[index];
	    return sqrt(dot(force[i],force[i])); 
	}
      };
      
      __device__ int sign(real x){ 
          int t = x<0 ? -1 : 0;
          return x > 0 ? 1 : t;
    }
      
      //In this kernel the particle positions are updated. 
      //The parameter "factor" is used with two values 1 or -1.
      //In the second case is to reverse the change.
      __global__ void minimizationStepKernel(real4* pos,
				             real4* force,
				             real maxForce,
				             real h,
				             real factor,
				             int numberParticles,
				             ParticleGroup::IndexIterator groupIterator){
	  const int id = blockIdx.x*blockDim.x + threadIdx.x;
	  if(id >= numberParticles) return;
	  
	  const int i = groupIterator[id];
	  
      real fmod = sqrt(dot(force[i],force[i]));
      
      if(fmod/maxForce > real(1.0)){
          real4 f;
          
          f.x = sign(force[i].x);
          f.y = sign(force[i].y);
          f.z = sign(force[i].z);
          f.w = real(0);
          
          pos[i] += factor*f*h;
          
      } else {
          pos[i] += factor*(force[i]/maxForce)*h;
      }
      
      }
  }
    
  SteepestDescent::SteepestDescent(shared_ptr<ParticleData> pd,
				   shared_ptr<ParticleGroup> pg,
				   shared_ptr<System> sys,		       
				   SteepestDescent::Parameters par):
    Integrator(pd, pg, sys, "SteepestDescent"),
    h(par.h), epsilon(par.epsilon),saturationForce(par.saturationForce),maxSteps(par.maxSteps),steps(0){
    
    sys->log<System::MESSAGE>("[SteepestDescent] Initial maximum displacement: %.3f", h);
    sys->log<System::MESSAGE>("[SteepestDescent] Force tolerance: %.3f", epsilon);
    
    cudaStreamCreate(&stream);
    
    cudaMallocManaged((void**)&totalEnergy,sizeof(real));
    cudaMallocManaged((void**)&maxForce,sizeof(real));
    this->setUpCubReduction();
  }
  
  SteepestDescent::~SteepestDescent(){
    
    sys->log<System::MESSAGE>("[SteepestDescent] Destroyed.");
    
    cudaFree(totalEnergy);
    cudaFree(maxForce);
    cudaFree(cubTempStorageSum);
    cudaFree(cubTempStorageMax);
    cudaStreamDestroy(stream);
  }
  
  //This function initializes all the necessary variables used by "cub".
  void SteepestDescent::setUpCubReduction(){
      //More info:https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html
      
      //common
      cub::CountingInputIterator<int> countingIterator(0);
      auto groupIterator = pg->getIndexIterator(access::location::gpu);
      
      //energy
      auto energy = pd->getEnergy(access::location::gpu, access::mode::read);
      SteepestDescent_ns::energyFunctor eF(groupIterator,energy.raw());

      cub::TransformInputIterator<real, SteepestDescent_ns::energyFunctor, cub::CountingInputIterator<int>> energySumIterator(countingIterator,eF);
      
      cub::DeviceReduce::Sum(cubTempStorageSum, cubTempStorageSizeSum, energySumIterator, totalEnergy, pg->getNumberParticles(), stream);
      cudaMalloc(&cubTempStorageSum, cubTempStorageSizeSum);
      
      //force
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      SteepestDescent_ns::forceFunctor fF(groupIterator,force.raw());
      
      cub::TransformInputIterator<real, SteepestDescent_ns::forceFunctor, cub::CountingInputIterator<int>> forceMaxIterator(countingIterator,fF);
      
      cub::DeviceReduce::Max(cubTempStorageMax, cubTempStorageSizeMax, forceMaxIterator, maxForce, pg->getNumberParticles(), stream);
      cudaMalloc(&cubTempStorageMax, cubTempStorageSizeMax);
  }
  
  
  bool SteepestDescent::minimizationStep(){
      
      real currentMaxForce = this->computeMaxForce();
      if(abs(currentMaxForce) < epsilon or steps > maxSteps) return false;
      
      real currentEnergy = this->sumEnergy();
      real newEnergy;
      
      {
	  
	int numberParticles = pg->getNumberParticles();
	auto groupIterator = pg->getIndexIterator(access::location::gpu);
	  
	int Nthreads=128;
	int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      
	auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
	auto force = pd->getForce(access::location::gpu, access::mode::read);
	
	SteepestDescent_ns::minimizationStepKernel<<<Nblocks,Nthreads>>>(pos.raw(),force.raw(),
									 currentMaxForce,
									 h,
									 real(1),
									 numberParticles,groupIterator);
	
      }
      
      newEnergy = this->sumEnergy();
      
      if(newEnergy < currentEnergy){
	  h*=real(1.2);
	  sys->log<System::DEBUG>("[SteepestDescent] Current energy: %.3f. Current max force: %.3f. Step: %i",
				    newEnergy,currentMaxForce,steps);
      } else {
	  
	  {
	  
	  int numberParticles = pg->getNumberParticles();
	  auto groupIterator = pg->getIndexIterator(access::location::gpu);
	  
	  int Nthreads=128;
	  int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
          
	  auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
	  auto force = pd->getForce(access::location::gpu, access::mode::read);
	  
	  SteepestDescent_ns::minimizationStepKernel<<<Nblocks,Nthreads>>>(pos.raw(),force.raw(),
	  								   currentMaxForce,
	  								   h,
	  								   real(-1),
	  								   numberParticles,groupIterator);
	
	  }
	  
	  h*=real(0.2);
	  sys->log<System::DEBUG>("[SteepestDescent] Current energy: %.3f. Current max force: %.3f. Step: %i",
				    currentEnergy,currentMaxForce,steps);
	  
      }
      
      steps++;
      
      
      
      return true;
      
  }
  
  void SteepestDescent::forwardTime(){
      
      real currentMaxForce = this->computeMaxForce();
      
      real currentEnergy = this->sumEnergy();
      real newEnergy;

      {
	  
	int numberParticles = pg->getNumberParticles();
	auto groupIterator = pg->getIndexIterator(access::location::gpu);
	  
	int Nthreads=128;
	int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      
	auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
	auto force = pd->getForce(access::location::gpu, access::mode::read);
	
	SteepestDescent_ns::minimizationStepKernel<<<Nblocks,Nthreads>>>(pos.raw(),force.raw(),
									 currentMaxForce,
									 h,
									 real(1),
									 numberParticles,groupIterator);
	
      }
      
      newEnergy = this->sumEnergy();
      
      if(newEnergy < currentEnergy){
	  h*=1.2;
	  sys->log<System::DEBUG>("[SteepestDescent] Current energy: %.3f. Current max force: %.3f. Step: %i",
				    currentEnergy,currentMaxForce,steps);
      } else {
	  
	  {
	  
	  int numberParticles = pg->getNumberParticles();
	  auto groupIterator = pg->getIndexIterator(access::location::gpu);
	  
	  int Nthreads=128;
	  int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
          
	  auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
	  auto force = pd->getForce(access::location::gpu, access::mode::read);
	  
	  SteepestDescent_ns::minimizationStepKernel<<<Nblocks,Nthreads>>>(pos.raw(),force.raw(),
	  								   currentMaxForce,
	  								   h,
	  								   real(-1),
	  								   numberParticles,groupIterator);
	
	  }
	  
	  h*=0.2;
	  sys->log<System::DEBUG>("[SteepestDescent] Current energy: %.3f. Current max force: %.3f. Step: %i",
				    currentEnergy,currentMaxForce,steps);
	  
      }
      
      return;
  }
  
  real SteepestDescent::sumEnergy(){
      
      int numberParticles = pg->getNumberParticles();
      auto groupIterator = pg->getIndexIterator(access::location::gpu);
      
      int Nthreads=128;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      
      {
	auto energy = pd->getEnergy(access::location::gpu, access::mode::readwrite);
        fillWithGPU<<<Nblocks, Nthreads>>>(energy.raw(), groupIterator, real(0), numberParticles);
      }
      
      for(auto energyComp: interactors) energyComp->sumEnergy();
      cudaDeviceSynchronize();
      
      cub::CountingInputIterator<int> countingIterator(0);
      
      //energy
      auto energy = pd->getEnergy(access::location::gpu, access::mode::read);
      SteepestDescent_ns::energyFunctor eF(groupIterator,energy.raw());
      
      cub::TransformInputIterator<real, SteepestDescent_ns::energyFunctor, cub::CountingInputIterator<int>> energySumIterator(countingIterator,eF);
      
      cub::DeviceReduce::Sum(cubTempStorageSum, cubTempStorageSizeSum, energySumIterator, totalEnergy, pg->getNumberParticles(), stream);
      cudaStreamSynchronize(stream);
      
      return *totalEnergy/real(2.0);
  }
  
  real SteepestDescent::computeMaxForce(){
      
      int numberParticles = pg->getNumberParticles();
      auto groupIterator = pg->getIndexIterator(access::location::gpu);
      
      int Nthreads=128;
      int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
      
      {
	auto force = pd->getForce(access::location::gpu, access::mode::readwrite);
        fillWithGPU<<<Nblocks, Nthreads>>>(force.raw(), groupIterator, make_real4(0), numberParticles);
      }
      
      for(auto forceComp: interactors) forceComp->sumForce(stream);
      cudaStreamSynchronize(stream);
      
      cub::CountingInputIterator<int> countingIterator(0);
      
      auto force = pd->getForce(access::location::gpu, access::mode::read);
      SteepestDescent_ns::forceFunctor fF(groupIterator,force.raw());
      
      cub::TransformInputIterator<real, SteepestDescent_ns::forceFunctor, cub::CountingInputIterator<int>> forceMaxIterator(countingIterator,fF);
      
      cub::DeviceReduce::Max(cubTempStorageMax, cubTempStorageSizeMax, forceMaxIterator, maxForce, numberParticles, stream);
      cudaStreamSynchronize(stream);
      
      return (*maxForce==INFINITY)?saturationForce:*maxForce;
  }
}
