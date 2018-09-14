/* Raul P. Pelaez 2018. Lattice Boltzmann Integrator. 

References:

[1] Optimized implementation of the Lattice Boltzmann Method on a graphics processing unit towards real-time fluid simulation.  N. Delbosc et al. http://dx.doi.org/10.1016/j.camwa.2013.10.002
[2] Accelerating fluid–solid simulations (Lattice-Boltzmann & Immersed-Boundary) on heterogeneous architectures. Pedro Valero-Lara et. al. https://hal.archives-ouvertes.fr/hal-01225734

 */

#include"LBM.cuh"
#include"utils/debugTools.cuh"
#include"misc/RPNG.cpp"
#include"visit_writer.c"
#define FLOWVEL 0.04
namespace uammd{
  namespace Hydro{
    namespace LBM_ns{

      
      struct D3Q19_Scheme{
	static constexpr int numberVelocities = 19;
	const int3 velocities[19] = {
	  { 0,-1,-1},                           //0
	  {-1, 0,-1}, {0, 0,-1}, {1, 0,-1},     //1 2 3
	  { 0, 1,-1},                           //4
	  {-1, 0, 0}, {0, 0, 0}, {1, 0, 0},     //5 6 7
	  {-1, 1, 0}, {0, 1, 0}, {1, 1, 0},     //8 9 10
	  {-1,-1, 0}, {0,-1, 0}, {1,-1, 0},     //11 12 13
	  { 0, 1, 1},                           //14
	  {-1, 0, 1}, {0, 0, 1}, {1, 0, 1},     //15 16 17
	  {0, -1, 1}};                          //18

	const int opposite[19] = {
	  14,
	  17, 16, 15,
	  18,
	  7, 6, 5,
	  13, 12, 11,
	  10, 9, 8,
	  0,
	  3, 2, 1,
	  4};
	  	
	const real wi[19] = {
	  1.0/36.0,
	  1.0/36.0, 1.0/18.0, 1.0/36.0,
	  1.0/36.0,
	  1.0/18.0, 1.0/3.0, 1.0/18.0,
	  1.0/36.0, 1.0/18.0, 1.0/36.0,
	  1.0/36.0, 1.0/18.0, 1.0/36.0,
	  1.0/36.0,
	  1.0/36.0, 1.0/18.0, 1.0/36.0,
	  1.0/36.0};

	inline __host__ __device__ real equilibriumDistribution(int i, real3 velocity, real density){
	  const real eu = dot(make_real3(velocities[i]), velocity);
	  const real si = (real(3.0)*eu +
			   real(4.5)*eu*eu -
			   real(1.5)*dot(velocity, velocity));
	  real feq = density*wi[i]*(real(1.0) + si);

	  return feq;
	}
	static inline __host__ const char * name(){return "D3Q19";}
      };      
      struct D2Q9_Scheme{
	static constexpr int numberVelocities = 9;
	const int3 velocities[9] = {
	  {0, 0, 0},                            //0
	  {1, 0, 0}, {0, 1, 0},                 //1 2
	  {-1, 0, 0}, {0, -1, 0},               //3 4
	  {1, 1, 0}, {-1, 1, 0},                //5 6
	  {-1, -1, 0}, {1, -1, 0},              //7 8	    
	};

	const int opposite[9] = {
	  0,
	  3, 4,
	  1, 2,
	  7, 8,
	  5, 6
	};
	  	
	const real wi[9] = {
	  4.0/9.0,
	  1.0/9.0, 1.0/9.0,
	  1.0/9.0, 1.0/9.0,
	  1.0/36.0, 1.0/36.0,
	  1.0/36.0, 1.0/36.0	  
	};

	inline __host__ __device__ real equilibriumDistribution(int i, real3 velocity, real density){
	  const real eu = dot(make_real3(velocities[i]), velocity);
	  const real si = (real(3.0)*eu +
			   real(4.5)*eu*eu -
			   real(1.5)*dot(velocity, velocity));
	  real feq = density*wi[i]*(real(1.0) + si);

	  return feq;
	}

	static inline __host__ const char * name(){return "D2Q9";}
      };

      template<class Scheme>
      __global__ void lbm_kernel(real * sourceGrid, real *destGrid,
				 int *cellType,
				 real relaxTime,
				 Grid grid,
				 int ncells,
				 Scheme sc){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>=ncells) return;
	int3 celli = make_int3(id%grid.cellDim.x,
			       (id/grid.cellDim.x)%grid.cellDim.y,
			       id/(grid.cellDim.x*grid.cellDim.y));
	//int icell = grid.getCellIndex(celli);
	const int icell = id;
	  
	
	real fi[sc.numberVelocities];
	const Boundary ct = static_cast<Boundary>(cellType[icell]);
	//Pull
	for(int i = 0; i<sc.numberVelocities; i++){
	  int3 cellj = celli - sc.velocities[i];
	  cellj = grid.pbc_cell(cellj);
	  int icellj = grid.getCellIndex(cellj);
	  fi[i] = sourceGrid[icellj+i*ncells];
	  if(ct == Boundary::Overdensity) fi[i] = sc.equilibriumDistribution(i, make_real3(0), 2.0f);
	  if(ct == Boundary::Free_out){
	    int3 cellj = celli-make_int3(1,0,0);
	    int icellj = grid.getCellIndex(cellj);	      
	    fi[i] = sourceGrid[icellj+i*ncells];
	  }
	}

	//Full way bounce-back
	if(ct == Boundary::Particle || ct == Boundary::BounceBack){
	  for(int i = 0; i<sc.numberVelocities; i++){
	    destGrid[icell + i*ncells] = fi[sc.opposite[i]];
	  }
	  return;
	}
	else if (ct == Boundary::ZhouHe_in){	  
	  real inflow = FLOWVEL;
	  real3 velocity = make_real3(inflow, 0, 0);
	  real density = (fi[0] + fi[2] + fi[4] +2*(fi[3] + fi[6] + fi[7]))/(real(1.0)-inflow);

	  fi[1] = sc.equilibriumDistribution(1, velocity, density) +
	    (fi[3] - sc.equilibriumDistribution(3, velocity, density));
	  fi[5] = sc.equilibriumDistribution(5, velocity, density) +
	    (fi[7] - sc.equilibriumDistribution(7, velocity, density));
	  fi[8] = sc.equilibriumDistribution(8, velocity, density) +
	    (fi[6] - sc.equilibriumDistribution(6, velocity, density));
	  
	}
	else if (ct == Boundary::ZhouHe_out){
	  int3 cellj = celli - make_int3(1,0,0);
	  cellj = grid.pbc_cell(cellj);
	  int icellj = grid.getCellIndex(cellj);
	  
	  fi[3] = sourceGrid[icellj+3*ncells];
	  fi[6] = sourceGrid[icellj+6*ncells];
	  fi[7] = sourceGrid[icellj+7*ncells];
	}

	real density = 0;
	real3 velocity = make_real3(0, 0, 0);
	for(int i = 0; i<sc.numberVelocities; i++){
	  density += fi[i];
	  velocity += fi[i]*make_real3(sc.velocities[i]);
	}

	velocity /= density;

	//Streaming+Collision
	for(int i = 0; i<sc.numberVelocities; i++){
	  real feq = sc.equilibriumDistribution(i, velocity, density);
	  destGrid[icell + i*ncells] = fi[i] - (real(1.0)/relaxTime)*(fi[i] - feq);
	}
      }
	  
      __global__ void particles2Grid(int *cellType,
				     real4  *pos,
				     real  *radius,
				     int N,
				     Grid grid){
	//int3 P, real radius2){
	const int id = blockIdx.x;
	const int tid = threadIdx.x;
	if(id>=N) return;

	__shared__ real3 pi;
	__shared__ real radius_i;
	__shared__ int3 celli;
	if(tid==0){
	  pi = make_real3(pos[id]);
	  /*Get my cell*/
	  celli = grid.getCell(pi);
	  radius_i= real(1.0);
	  if(radius) radius_i = radius[id];
	}
	/*Conversion between cell number and cell center position*/
	const real3 cellPosOffset = real(0.5)*(grid.cellSize - grid.box.boxSize);

	int3 P = make_int3(radius_i/grid.cellSize+0.5)+1;
	if(grid.cellDim.z==1) P.z = 0; //2D case
	const int3 supportCells = 2*P + 1;
	const int numberNeighbourCells = supportCells.x*supportCells.y*supportCells.z;

	__syncthreads();
	for(int i = tid; i<numberNeighbourCells; i+=blockDim.x){
	  /*Compute neighbouring cell*/
	  int3 cellj = make_int3(celli.x + i%supportCells.x - P.x,
				 celli.y + (i/supportCells.x)%supportCells.y - P.y,
				 celli.z + i/(supportCells.x*supportCells.y) - P.z );
	  cellj = grid.pbc_cell(cellj);	  	  
	  const int jcell = grid.getCellIndex(cellj);
	  real3 rij = grid.box.apply_pbc(pi-make_real3(cellj)*grid.cellSize - cellPosOffset);
	  real r2 = dot(rij, rij);
	  if(r2<=(radius_i*radius_i))
	    cellType[jcell] = static_cast<int>(Boundary::Particle);
	}
      }



      struct DefaultInitialConditions{

	inline __device__ real density(int3 celli, Grid grid){
	  //if(celli.x == grid.cellDim.x/2) return real(2.0);
	  return real(1.0);
	}

	inline __device__ real3 velocity(int3 celli, Grid grid){
	  real u0 = 0.0;
	  real3 vel = make_real3(real(0.0));
	  // real y = celli.y/(float)grid.cellDim.y;
	  // real z = celli.z/(float)grid.cellDim.z;	
	  real x = celli.x/(float)grid.cellDim.x;	
	  vel.y = u0*sin(2*M_PI*x*4);	 
	  return vel;
	}
	
	
      };
      template<class Scheme, class InitialConditions>
      __global__ void lbm_initial(real * sourceGrid, real *destGrid,
				  real relaxTime,
				  Grid grid,
				  int ncells,
				  Scheme sc,
				  InitialConditions init){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>=ncells) return;
	int3 celli = make_int3(id%grid.cellDim.x,
			       (id/grid.cellDim.x)%grid.cellDim.y,
			       id/(grid.cellDim.x*grid.cellDim.y));
	
	real density = init.density(celli, grid);
	real3 velocity = init.velocity(celli, grid);
		
	//int icell = grid.getCellIndex(celli);
	int icell = id;
	
	for(int i = 0; i<sc.numberVelocities; i++){
	  real feq = sc.equilibriumDistribution(i, velocity, density);	    
	  sourceGrid[icell + i*ncells] = feq;	 
	}
      }
	
      
      template<class Scheme>
      template<class InitialConditions>
      LBM_base<Scheme>::LBM_base(shared_ptr<ParticleData> pd,
				 shared_ptr<System> sys,
				 Parameters par,
				 InitialConditions init):
	Integrator(pd, sys, "LBM"),
	soundSpeed(par.soundSpeed),
	viscosity(par.viscosity),
	dt(par.dt){
	sys->log<System::MESSAGE>("[LBM] Created");

	this->numberVelocities = Scheme::numberVelocities;
      
	real dx = sqrt(3)*soundSpeed/dt; //Simulation units c = dx_lbm/dt, dt_lbm=1
	int3 cellDim = make_int3(par.box.boxSize/dx);
	grid = Grid(par.box, cellDim);
	int ncells = grid.getNumberCells();
	
	relaxTime = 0.5 + pow(soundSpeed/dt,2)*viscosity/(dt*dt);

	real minR = 10000000000;
	if(pd->getNumParticles()>0){	  
	  auto radius = pd->getRadius(access::location::cpu, access::mode::read);
	  fori(0, pd->getNumParticles()){
	    minR = std::min(minR, radius.raw()[i]);
	  }
	}
	
	sys->log<System::MESSAGE>("[LBM] Cells: %d %d %d", grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
	sys->log<System::MESSAGE>("[LBM] ncells: %d", ncells);
	sys->log<System::MESSAGE>("[LBM] soundSpeed: %e", this->soundSpeed);
	sys->log<System::MESSAGE>("[LBM] relaxTime: %e", this->relaxTime);
	sys->log<System::MESSAGE>("[LBM] viscosity: %e", viscosity);
	sys->log<System::MESSAGE>("[LBM] Re: %e", minR*FLOWVEL/viscosity);
	sys->log<System::MESSAGE>("[LBM] Mode: %s", Scheme::name());

	sourceGrid.resize(ncells*numberVelocities, 0.0);
	
	cellType.resize(ncells, static_cast<int>(Boundary::None));
      
	destGrid = sourceGrid;
	int Nthreads = 64;
	int Nblocks = ncells/Nthreads+1;
	real *sourceGrid_ptr = thrust::raw_pointer_cast(sourceGrid.data());
	real *destGrid_ptr = thrust::raw_pointer_cast(destGrid.data());
	lbm_initial<<<Nblocks, Nthreads>>>(sourceGrid_ptr,
					   destGrid_ptr,
					   relaxTime,
					   grid,
					   ncells,
					   Scheme(),
					   init);
}


      //Set to zero the cells that are inside a particles support
      __global__ void resetParticleCells(int *cellType,
				    int ncells){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id>=ncells) return;
	if(cellType[id] == static_cast<int>(Boundary::Particle)) cellType = static_cast<int>(Boundary::None);
      }
      
      template<class Scheme>
      void LBM_base<Scheme>::forwardTime(){
	static int steps = 0;
	steps++;
	sys->log<System::DEBUG1>("[LBM] Performing step %d", steps);
	int ncells = grid.getNumberCells();
	int Nthreads = 64;
	int Nblocks = ncells/Nthreads+1;
      
	real *sourceGrid_ptr = thrust::raw_pointer_cast(sourceGrid.data());
	real *destGrid_ptr = thrust::raw_pointer_cast(destGrid.data());      
	int *cellType_ptr = thrust::raw_pointer_cast(cellType.data());

	int numberParticles = pd->getNumParticles();
      
	if(numberParticles > 0){
	  sys->log<System::DEBUG2>("[LBM] Spreading particles to grid");
	  resetParticleCells<<<Nblocks, Nthreads>>>(cellType_ptr, ncells);
	  auto pos = pd->getPos(access::location::gpu, access::mode::read);
	  auto radius = pd->getRadiusIfAllocated(access::location::gpu, access::mode::read);	 
	 
	 
	  particles2Grid<<<numberParticles, 32>>>(cellType_ptr,
						  pos.raw(),
						  radius.raw(),
						  numberParticles,
						  grid);
	  CudaCheckError();
	}
	sys->log<System::DEBUG2>("[LBM] Launching main kernel");
	lbm_kernel<<<Nblocks, Nthreads>>>(sourceGrid_ptr, destGrid_ptr,
					  cellType_ptr,
					  relaxTime,
					  grid,
					  ncells,
					  Scheme());
	CudaCheckError();     
	try{
	  destGrid.swap(sourceGrid);
	}
	catch(thrust::system_error &e){
	  sys->log<System::CRITICAL>("[LBM] Thrust could not swap grid buffers with error: %s", e.what());
	}
	CudaCheckError();
}


      template<class CellTypeSelector>
      __global__ void setBoundariesGPU(int *cellType,
				       Grid grid,
				       int ncells,
				       CellTypeSelector cs){
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id >= ncells) return;
	
	int3 celli = make_int3(id%grid.cellDim.x,
			       (id/grid.cellDim.x)%grid.cellDim.y,
			       id/(grid.cellDim.x*grid.cellDim.y));


	cellType[id] = static_cast<int>(cs(celli, grid));
	
     } 
      template<class Scheme>
      template<class CellSelector>
      void LBM_base<Scheme>::setBoundaries(CellSelector cs){

	int ncells = grid.getNumberCells();
	int Nthreads = 64;
	int Nblocks = ncells/Nthreads+1;
      
	int *cellType_ptr = thrust::raw_pointer_cast(cellType.data());

	setBoundariesGPU<<<Nblocks, Nthreads>>>(cellType_ptr, grid, ncells, cs);

      }

      
      template<class Scheme>
      void LBM_base<Scheme>::write(){}
      
      template<class Scheme>
      void LBM_base<Scheme>::writePNG(){
	thrust::host_vector<real> h_data = sourceGrid;
	
	int ncells = grid.getNumberCells();
	std::vector<unsigned char> image(4*grid.cellDim.x*grid.cellDim.y,0);
	real max = 0;
	real min = 100000;
	fori(0, grid.cellDim.x){
	  forj(0,grid.cellDim.y){
	    //for(int kz = 0; kz<grid.cellDim.z; kz++){
	    {
	      real density = 0.0;
	      real velocity = 0.0;
	      real3 vel = make_real3(0);
	      int icell = grid.getCellIndex(make_int3(i, j, 0));
	      for(int k = 0; k<numberVelocities; k++){
		real f = h_data[icell+ncells*k];
		density += f;
		vel += f*make_real3(Scheme().velocities[k]);
	      }
	      velocity = dot(vel, vel)/density;
	      density = velocity;
	      max = std::max(max, density);
	      min = std::min(min, density);
	    }
	  }
	}
	
	fori(0, grid.cellDim.x){
	  forj(0,grid.cellDim.y){
	    {
	      real density = 0.0;
	      real velocity = 0;
	      real3 vel = make_real3(0);
	      int icell = grid.getCellIndex(make_int3(i, j, 0));
	      for(int k = 0; k<numberVelocities; k++){
		real f = h_data[icell+ncells*k];
		density += f;
		vel += f*make_real3(Scheme().velocities[k]);
	      }
	      velocity = dot(vel, vel)/density;
	      density = velocity;
	      unsigned char R = std::min((unsigned char)255, (unsigned char)(((density-min)/(max-min))*255) );
	      unsigned char B = 255-R;
	      image[4*(i+grid.cellDim.x*j)] = R;
	      image[4*(i+grid.cellDim.x*j)+1] = 0;
	      image[4*(i+grid.cellDim.x*j)+2] = B;
	      image[4*(i+grid.cellDim.x*j)+3] = 255;
	    }
	  }
	}
	static int counter = 0;
	savePNG((std::to_string(counter)+".png").c_str(), image.data(), grid.cellDim.x, grid.cellDim.y);
	counter++;}
     
      template<class Scheme>
      void LBM_base<Scheme>::writeVTK(){

	thrust::host_vector<real> h_data = sourceGrid;

	
	int ncells = grid.getNumberCells();
	thrust::host_vector<double3> velMesh(ncells);
	
	fori(0, grid.cellDim.x){
	  forj(0,grid.cellDim.y){
	    real density = 0.0;
	    real3 vel = make_real3(0);
	    int icell = grid.getCellIndex(make_int3(i, j, 0));
	    for(int k = 0; k<numberVelocities; k++){
	      real f = h_data[icell+ncells*k];
	      density += f;
	      vel += f*make_real3(Scheme().velocities[k]);
	    }
	    vel/=density;
	    velMesh[i+grid.cellDim.x*j] = make_double3(vel);
	  }
	}
	
	static int counter = 0;

	int dims[] = {grid.cellDim.x+1,grid.cellDim.y+1,1};
	int vardims[] = {3};
	int centering[]={0};
	const char *varnames[]={"vel"};
	
	double *data[] = {((double*)thrust::raw_pointer_cast(velMesh.data()))};
	
	write_regular_mesh(("vel."+std::to_string(counter)+".vtk").c_str(),
			   0, dims,
			   1,
			   vardims, centering,
			   varnames, data);

	
	counter++;
}
    }      
  }
}






/*
      D3Q19::D3Q19(shared_ptr<ParticleData> pd,
		   shared_ptr<System> sys,
		   Parameters par):
	Integrator(pd, sys, "LBM::D3Q19"),
	soundSpeed(par.soundSpeed),
        viscosity(par.viscosity),
        dt(par.dt), out("fluid.dat"){
	sys->log<System::MESSAGE>("[LBM::D3Q19] Created");

	

	real dx = sqrt(3)*soundSpeed/dt; //Simulation units c = dx_lbm/dt, dt_lbm=1
	int3 cellDim = make_int3(par.box.boxSize/dx);
	grid = Grid(par.box, cellDim);
	int ncells = grid.getNumberCells();
	
	relaxTime = 0.5 + pow(soundSpeed/dt,2)*viscosity/(dt*dt);
	
	sys->log<System::MESSAGE>("[LBM::D3Q19] Cells: %d %d %d", grid.cellDim.x, grid.cellDim.y, grid.cellDim.z);
	sys->log<System::MESSAGE>("[LBM::D3Q19] ncells: %d", ncells);
	sys->log<System::MESSAGE>("[LBM::D3Q19] soundSpeed: %e", this->soundSpeed);
	sys->log<System::MESSAGE>("[LBM::D3Q19] relaxTime: %e", this->relaxTime);
	sys->log<System::MESSAGE>("[LBM::D3Q19] viscosity: %e", viscosity);

	sourceGrid.resize(ncells*numberVelocities, 0.0);
	
        cellType.resize(ncells, 0);
	
	destGrid = sourceGrid;
	int Nthreads = 64;
	int Nblocks = ncells/Nthreads+1;
	real *sourceGrid_ptr = thrust::raw_pointer_cast(sourceGrid.data());
	real *destGrid_ptr = thrust::raw_pointer_cast(destGrid.data());
	lbm_initial<<<Nblocks, Nthreads>>>(sourceGrid_ptr,
					   destGrid_ptr,
					   relaxTime,
					   grid,
					   ncells,
					   D3Q19_ns::Scheme());
      }

      void D3Q19::forwardTime(){
	static int steps = 0;
	steps++;
	sys->log<System::DEBUG>("[Hydro::LBM::D3Q19] Performing step %d", steps);
	int ncells = grid.getNumberCells();
	sys->log<System::DEBUG>("[Hydro::LBM::D3Q19] Cells %d", ncells);
	int Nthreads = 64;
	int Nblocks = ncells/Nthreads+1;
	real *sourceGrid_ptr = thrust::raw_pointer_cast(sourceGrid.data());
	real *destGrid_ptr = thrust::raw_pointer_cast(destGrid.data());

	int numberParticles = pd->getNumParticles();	
	int *cellType_ptr = thrust::raw_pointer_cast(cellType.data());
	fillWithGPU<<<Nblocks, Nthreads>>>(cellType_ptr, 0, ncells);
	if(numberParticles > 0){
	  auto pos = pd->getPos(access::location::gpu, access::mode::read);
	  auto radius = pd->getRadiusIfAllocated(access::location::gpu, access::mode::read);	 

	 
	 
	  particles2Grid<<<numberParticles, 32>>>(cellType_ptr,
						  pos.raw(),
						  radius.raw(),
						  numberParticles,
						  grid);
	}
	lbm_kernel<<<Nblocks, Nthreads>>>(sourceGrid_ptr, destGrid_ptr,
					  cellType_ptr,
					  relaxTime,
					  grid,
					  ncells,
					  D3Q19_ns::Scheme());
	
	destGrid.swap(sourceGrid);

      }
      void D3Q19::write(){}
      
      void D3Q19::writePNG(){
	
	thrust::host_vector<real> h_data = sourceGrid;
	
	int ncells = grid.getNumberCells();
	std::vector<unsigned char> image(4*grid.cellDim.x*grid.cellDim.y,0);
	real max = 0;
	real min = 100000;
	fori(0, grid.cellDim.x){
	  forj(0,grid.cellDim.y){
	    //for(int kz = 0; kz<grid.cellDim.z; kz++){
	    {
	    int kz = grid.cellDim.z/2;
	    //if(kz!=20) break;
	      real density = 0.0;
	      int icell = grid.getCellIndex(make_int3(i, j, kz));
	      for(int k = 0; k<19; k++){
		density += h_data[icell+ncells*k];
	      }
	      max = std::max(max, density);
	      min = std::min(min, density);
	    }
	  }
	}
	
	fori(0, grid.cellDim.x){
	  forj(0,grid.cellDim.y){
	    //for(int kz = 0; kz<grid.cellDim.z; kz++){
	    {
	    int kz = grid.cellDim.z/2;
	      real density = 0.0;
	      //real3 vel = make_real3(0);
	      int icell = grid.getCellIndex(make_int3(i, j, kz));
	      for(int k = 0; k<19; k++){
		density += h_data[icell+ncells*k];
	
	      }       
	      unsigned char R = std::min((unsigned char)255, (unsigned char)(((density-min)/(max-min))*255) );
	      unsigned char B = 255-R;
	      image[4*(i+grid.cellDim.x*j)] = R;
	      image[4*(i+grid.cellDim.x*j)+1] = 0;
	      image[4*(i+grid.cellDim.x*j)+2] = B;
	      image[4*(i+grid.cellDim.x*j)+3] = 255;
	    }
	  }
	}
	//out<<std::flush;
	static int counter = 0;
	savePNG((std::to_string(counter)+".png").c_str(), image.data(), grid.cellDim.x, grid.cellDim.y);
	counter++;
      }

*/