
#ifndef LBM_CUH
#define LBM_CUH

#include"uammd.cuh"
#include"Integrator/Integrator.cuh"
#include<thrust/device_vector.h>
#include<fstream>
namespace uammd{
  namespace Hydro{
    namespace LBM_ns{

      enum class Boundary{
	None = 0,
	Overdensity,
	BounceBack,
	ZhouHe_in,
	ZhouHe_out,
	Free_out,
	Particle};
      
      class DefaultInitialConditions; //Forward declaration of class in LBM.cu
      
      template<class Scheme>
      class LBM_base: public Integrator{
	int steps;
	thrust::device_vector<real> sourceGrid, destGrid;
	thrust::device_vector<int> cellType;
	Grid grid;
        int numberVelocities;
	real soundSpeed, relaxTime, dt, viscosity;
	std::ofstream out;
      public:
	struct Parameters{
	  Box box;
	  real soundSpeed;
	  real dt;
	  real viscosity;	  
	};
	template<class InitialConditions  = DefaultInitialConditions>
	LBM_base(shared_ptr<ParticleData> pd,
		 shared_ptr<System> sys,
		 Parameters par,
		 InitialConditions init = InitialConditions());    
	virtual void forwardTime() override;

	template<class CellSelector>
	void setBoundaries(CellSelector cs = CellSelector());
	
	void write();
	void writePNG();
	void writeVTK();
      };

      class D3Q19_Scheme;
      class D2Q9_Scheme;
    }

    namespace LBM{
      using Boundary = LBM_ns::Boundary;
      using D2Q9  = LBM_ns::LBM_base<LBM_ns::D2Q9_Scheme>;
      using D3Q19 = LBM_ns::LBM_base<LBM_ns::D3Q19_Scheme>;
    }
    
  }
}


#include"LBM.cu"
#endif
/*

      class D2Q9: public Integrator{
	int steps;
	thrust::device_vector<real> sourceGrid, destGrid;
	thrust::device_vector<int> cellType;
	Grid grid;
	static constexpr int numberVelocities = 9;
	real soundSpeed, relaxTime, dt, viscosity;
	std::ofstream out;
      public:
	struct Parameters{
	  Box box;
	  real soundSpeed;
	  real dt;
	  real viscosity;
	};
	D2Q9(shared_ptr<ParticleData> pd,
	      shared_ptr<System> sys,
	      Parameters par);    
	virtual void forwardTime() override;
	void write();
	void writePNG();
	void writeVTK();
      };

*/