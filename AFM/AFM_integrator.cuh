#include "Integrator/Integrator.cuh"

#ifndef SINGLE_PRECISION
#define curandGenerateNormal curandGenerateNormalDouble
#endif

#include "third_party/saruprng.cuh"

#include "Interactor/ExternalForces.cuh"
#include "third_party/cub/cub.cuh"

namespace uammd{
  namespace AFM{
    namespace AFM_integrator_ns{
        
        
        __global__ void initialVelocities(real3* vel, const real* mass,
                                          ParticleGroup::IndexIterator indexIterator, //global index of particles in my group
                                          real vamp, bool is2D, int N, uint seed){
                    
            int id = blockIdx.x*blockDim.x + threadIdx.x;
            
            if(id>=N) return;
            
            Saru rng(id, seed);
            
            int i = indexIterator[id];
            
            real mass_i = real(1.0);
            if(mass) mass_i = mass[i];
            
            double3 noisei = make_double3(rng.gd(0, vamp/mass_i), rng.gd(0, vamp/mass_i).x); //noise[id];
            
            int index = indexIterator[i];
            vel[index].x = noisei.x;
            vel[index].y = noisei.y;
            if(!is2D){
                vel[index].z = noisei.z;
            }
        }

        //Integrate the movement 1 dt and reset the forces in the first step
        //Uses the Gronbech Jensen scheme[1]
        //  r[t+dt] = r[t] + b·dt·v[t] + b·dt^2/(2·m) + b·dt/(2·m) · noise[t+dt]     
        //  v[t+dt] = a·v[t] + dt/(2·m)·(a·f[t] + f[t+dt]) + b/m·noise[t+dt]
        // b = 1/( 1 + \gamma·dt/(2·m))
        // a = (1 - \gamma·dt/(2·m) ) ·b
        // \gamma = 6*pi*viscosity*radius
        template<int step>
        __global__ void integrateGPU(real4 __restrict__  *pos,
                                     real3 __restrict__  *vel,
                                     real4 __restrict__  *force,
                                     const real __restrict__ *mass,
                                     const real __restrict__ *radius,
                                     ParticleGroup::IndexIterator indexIterator,
                                     int N,
                                     real dt, real viscosity, bool is2D,
                                     real noiseAmplitude,
                                     uint stepNum, uint seed){
                         
              const int id = blockIdx.x*blockDim.x+threadIdx.x;
              if(id>=N) return;
              
              //Index of current particle in group
              const int i = indexIterator[id];       
          
              Saru rng(id, stepNum, seed);
          
              real invMass = real(1.0);
              if(mass){
                  invMass = real(1.0)/mass[i];
              }
              real radius_i = real(1.0);
              if(radius){
                  radius_i = radius[i];
              }
          
              noiseAmplitude *= sqrtf(radius_i)*invMass;
              real3 noisei = make_real3(rng.gf(0, noiseAmplitude), rng.gf(0, noiseAmplitude).x); //noise[id];
              
              const real damping = real(6.0)*real(M_PI)*viscosity*radius_i;
          
              if(step==1){
                  const real gdthalfinvMass = damping*dt*invMass*real(0.5);
                  const real b = real(1.0)/(real(1.0) + gdthalfinvMass);
              
                  const real a = (real(1.0) - gdthalfinvMass)*b;
                  
              
                  real3 p = make_real3(pos[i]);
                  p = p + b*dt*(
                      vel[i] +
                      real(0.5)*(
                              (dt*invMass)*make_real3(force[i]) +
                              noisei
                              )
                      );
              
                  pos[i] = make_real4(p, pos[i].w);
          
                  vel[i] = a*vel[i] +
                  dt*real(0.5)*invMass*a*make_real3(force[i]) +
                  b*noisei;
                  
                  if(is2D) vel[i].z = real(0.0);
                  
                  force[i] = make_real4(0);
              }      
              else{
                  vel[i] += dt*real(0.5)*invMass*make_real3(force[i]);
              }
    
        }

    }
    
    struct TipForce: public ParameterUpdatable{
    
    
        real3 tipPosition = {0,0,0};
        real  tipVel = 0;
        
        real  harmonicWallTip;
        
        real  tipRadius;
        real  tipMass;
        
        TipForce(real harmonicWallTip,real tipRadius,real tipMass):harmonicWallTip(harmonicWallTip),tipRadius(tipRadius),tipMass(tipMass){}
        
        __device__ __forceinline__ real3 force(const real4 &pos,const real &radius){
            
            real3 rtp = tipPosition-make_real3(pos);
            real  r2 = dot(rtp,rtp);
            real  r  = sqrt(r2);
            
            real rMin = tipRadius+radius;
            
            if(r < rMin){
                
                real fmod = harmonicWallTip*(r-rMin);
                
                return fmod*(rtp/r);
                
            } else {
                return make_real3(0);
            }
        }
        
        std::tuple<const real4 *,const real *> getArrays(ParticleData *pd){
            auto pos    = pd->getPos(access::location::gpu, access::mode::read);
            auto radius = pd->getRadius(access::location::gpu, access::mode::read);
            return std::make_tuple(pos.raw(),radius.raw());
        }
        
        void setTipPosition(real3 newTipPosition){
            tipPosition = newTipPosition;
        }
        
        void setTipHeight(real newHeight){
            tipPosition.z = newHeight;
        }
        
        real getTipHeight(){
            return tipPosition.z;
        }
        
        real3 getTipPosition(){
            return tipPosition;
        }
        
        void setTipRadius(real newTipRadius){
            tipRadius = newTipRadius;
        }
        
        real getTipRadius(){
            return tipRadius;
        }
        
        void setTipMass(real newMass){
            tipMass = newMass;
        }
        
        real getTipMass(){
            return tipMass;
        }
        
        void setTipVel(real newVel){
            tipVel = newVel;
        }
        
        real getTipVel(){
            return tipVel;
        }
    
    };
    
    struct forceFunctor{
        ParticleGroup::IndexIterator groupIterator;
        
        real4* force;
        
        forceFunctor(ParticleGroup::IndexIterator groupIterator,real4* force):groupIterator(groupIterator),force(force){}
        
        __host__ __device__ __forceinline__
        real3 operator()(const int &index) const {
            
            int i = groupIterator[index];
            return make_real3(force[i]); 
        }
    };
    
    class AFM_integrator: public Integrator{
        
        public:
            
            struct Parameters{
                real temperature = 0;
                real dt = 0;
                real viscosity = 1.0;
                bool is2D = false;
                
                real kTip;
                real harmonicWallTip;
                
                real tipMass;
                real tipRadius;
            };
            
        protected:
        
            real noiseAmplitude;
            uint seed;
            real dt, temperature, viscosity;    
            bool is2D;
    
            cudaStream_t stream;
            int steps;
            
            ////////////////////////////////////////////////////////////
            
            std::shared_ptr<TipForce> tipPot;
            std::shared_ptr<ExternalForces<TipForce>> tip;
            
            real tipPosEq;
            
            real kTip;
            real invTipMass;
            
            real dampingTip;
            real aTip;
            real bTip;
            
            real noiseAmplitudeTip;
            
            real3 integrationTipForce;
            
            ////////////////////////////////////////////////////////////
            
            //cub reduction variables
            real3*   tipForce;
            void*    cubTempStorageSum = NULL;
            size_t   cubTempStorageSizeSum = 0;
            
            void setUpCubReduction(){
                
                //common
                cub::CountingInputIterator<int> countingIterator(0);
                auto groupIterator = pg->getIndexIterator(access::location::gpu);
                
                //force
                auto force = pd->getForce(access::location::gpu, access::mode::read);
                forceFunctor fF(groupIterator,force.raw());
            
                cub::TransformInputIterator<real3, forceFunctor, cub::CountingInputIterator<int>> forceSumIterator(countingIterator,fF);
                
                cub::DeviceReduce::Sum(cubTempStorageSum, cubTempStorageSizeSum, forceSumIterator, tipForce, pg->getNumberParticles(), stream);
                cudaMalloc(&cubTempStorageSum, cubTempStorageSizeSum);

            }
            
            real3 sumTipForce(bool resetForces = true){
          
                int numberParticles = pg->getNumberParticles();
                auto groupIterator = pg->getIndexIterator(access::location::gpu);
                
                int Nthreads=128;
                int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
                
                if(resetForces){
	                auto force = pd->getForce(access::location::gpu, access::mode::readwrite);
                    fillWithGPU<<<Nblocks, Nthreads>>>(force.raw(), groupIterator, make_real4(0), numberParticles);
                }
                
                this->tip->sumForce(stream);
                cudaDeviceSynchronize();
                
                cub::CountingInputIterator<int> countingIterator(0);
                
                //force
                auto force = pd->getForce(access::location::gpu, access::mode::read);
                forceFunctor fF(groupIterator,force.raw());
                
                cub::TransformInputIterator<real3, forceFunctor, cub::CountingInputIterator<int>> forceSumIterator(countingIterator,fF);
                
                cub::DeviceReduce::Sum(cubTempStorageSum, cubTempStorageSizeSum, forceSumIterator, tipForce, pg->getNumberParticles(), stream);
                cudaStreamSynchronize(stream);
                
                return *tipForce;
            }
    
        public:
        
            AFM_integrator(shared_ptr<ParticleData> pd,
                           shared_ptr<ParticleGroup> pg,
                           shared_ptr<System> sys,
                           Parameters par):Integrator(pd, pg, sys, "AFM_INTEGRATOR"),
                                           dt(par.dt), temperature(par.temperature), viscosity(par.viscosity), is2D(par.is2D),
                                           kTip(par.kTip),
                                           steps(0){
                sys->rng().next32();
                sys->rng().next32();
                seed = sys->rng().next32();
                sys->log<System::MESSAGE>("[%s] Temperature: %f", name.c_str(), temperature);
                sys->log<System::MESSAGE>("[%s] Time step: %f", name.c_str(), dt);
                sys->log<System::MESSAGE>("[%s] Viscosity: %f", name.c_str(), viscosity);
              
                if(is2D){
	                sys->log<System::MESSAGE>("[%s] Working in 2D mode.", name.c_str());
                }
        
                this->noiseAmplitude = sqrt(2*dt*6*M_PI*viscosity*temperature);
            
                int numberParticles = pg->getNumberParticles();
            
                int Nthreads=128;
                int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
        
                if(pd->isVelAllocated()){
	                sys->log<System::WARNING>("[%s] Velocity will be overwritten to ensure temperature conservation!", name.c_str());
                }
                
                ////////////////////////////////////////////////////////
                
                {
	                auto vel_handle = pd->getVel(access::location::gpu, access::mode::write);
	                auto groupIterator = pg->getIndexIterator(access::location::gpu);
                      
	                real velAmplitude = sqrt(3.0*temperature);
                      	
	                auto mass = pd->getMassIfAllocated(access::location::gpu, access::mode::read);
                
                      
	                AFM_integrator_ns::initialVelocities<<<Nblocks, Nthreads>>>(vel_handle.raw(),
                                                                                mass.raw(),
                                                                                groupIterator,
                                                                                velAmplitude, is2D, numberParticles,
                                                                                sys->rng().next32());
                }
        
                cudaStreamCreate(&stream);
                
                ////////////////////////////////////////////////////////
                
                tipPot = std::make_shared<TipForce>(par.harmonicWallTip,par.tipRadius,par.tipMass);
                tip = std::make_shared<ExternalForces<TipForce>>(pd, pg, sys,tipPot);
                
                invTipMass = real(1.0)/this->tipPot->getTipMass();
                
                dampingTip = real(6.0)*real(M_PI)*viscosity*this->tipPot->getTipRadius();
                
                bTip = real(1.0)/(1.0+dampingTip*dt*invTipMass*real(0.5));
                aTip = (1.0-dampingTip*dt*invTipMass*real(0.5))*bTip;
                
                noiseAmplitudeTip = noiseAmplitude*sqrtf(this->tipPot->getTipRadius())*invTipMass;
                
                ////////////////////////////////////////////////////////
                
                cudaMallocManaged((void**)&tipForce,sizeof(real3));
                this->setUpCubReduction();
            }
            
            ~AFM_integrator(){
                
                cudaFree(tipForce);
                cudaFree(cubTempStorageSum);
                cudaStreamDestroy(stream);
            }
    
            virtual void forwardTime() override;
            virtual real sumEnergy()   override{ return 0;};
            
            real3 sumTipForce();
            
            void setTipPosition(real3 newTipPosition){this->tipPot->setTipPosition(newTipPosition);}
            void setTipPositionEq(real newTipPosEq){tipPosEq = newTipPosEq;}
            void setTipHeight(real newHeight){this->tipPot->setTipHeight(newHeight);}
            void setTipRadius(real newTipRadius){this->tipPot->setTipRadius(newTipRadius);}
            
            real3 getTipPosition(){return this->tipPot->getTipPosition();}
            real  getTipPositionEq(){return tipPosEq;}
            real  getTipRadius(){return this->tipPot->getTipRadius();}
            
            real3 getTipForce(){return integrationTipForce;}
    };
    
    //Move the particles in my group 1 dt in time.
    void AFM_integrator::forwardTime(){
        
        CudaCheckError();
        
        for(auto forceComp: interactors) {
            forceComp->updateSimulationTime(steps*dt);
            forceComp->updateStep(steps);
        }
    
        steps++;
        sys->log<System::DEBUG1>("[%s] Performing integration step %d", name.c_str(), steps);
    
        int numberParticles = pg->getNumberParticles();
    
        int Nthreads=128;
        int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
    
        //First simulation step is special
        if(steps==1){
            
            tipPosEq = this->tipPot->getTipPosition().z;
            
            {
                auto groupIterator = pg->getIndexIterator(access::location::gpu);
                auto force = pd->getForce(access::location::gpu, access::mode::write);     
                fillWithGPU<<<Nblocks, Nthreads>>>(force.raw(), groupIterator, make_real4(0), numberParticles);
            }
            
            integrationTipForce = this->sumTipForce(false);
            
            for(auto forceComp: interactors){
                forceComp->updateTemperature(temperature);
                forceComp->updateTimeStep(dt);
                forceComp->sumForce(stream);
            }
            
            CudaSafeCall(cudaDeviceSynchronize());
            
        }
        
        ////////////////////////////////////////////////////////////////
        
        real tipNoise;
        real currentTipForce;
        
        real tipPos;
        real tipVel;
        
        ////////////////////////////////////////////////////////////////
        
        //First integration step
        {

            //An iterator with the global indices of my groups particles
            auto groupIterator = pg->getIndexIterator(access::location::gpu);
            //Get all necessary properties
            auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
            auto vel = pd->getVel(access::location::gpu, access::mode::readwrite);
            auto force = pd->getForce(access::location::gpu, access::mode::read);     
            //Mass is assumed 1 for all particles if it has not been set.
            auto mass = pd->getMassIfAllocated(access::location::gpu, access::mode::read);
            auto radius = pd->getRadiusIfAllocated(access::location::gpu, access::mode::read);
                
            //First step integration and reset forces
            AFM_integrator_ns::integrateGPU<1><<<Nblocks, Nthreads, 0, stream>>>(pos.raw(),
                                                                                 vel.raw(),
                                                                                 force.raw(),
                                                                                 mass.raw(),
                                                                                 radius.raw(),
                                                                                 groupIterator,
                                                                                 numberParticles, dt, viscosity, is2D,
                                                                                 noiseAmplitude,
                                                                                 steps, seed);
            CudaCheckError();
            
            //Tip integration
            
            tipPos = this->tipPot->getTipHeight();
            tipVel = this->tipPot->getTipVel();
            
            tipNoise = sys->rng().gaussian(0,noiseAmplitudeTip);
            currentTipForce = -integrationTipForce.z-kTip*(tipPos-tipPosEq);
            
            tipPos = tipPos + bTip*dt*(tipVel+real(0.5)*(dt*invTipMass*currentTipForce)+tipNoise);
            tipVel = aTip*tipVel + dt*real(0.5)*invTipMass*aTip*currentTipForce+bTip*tipNoise;
            
            this->tipPot->setTipHeight(tipPos);           
            this->tipPot->setTipVel(tipVel);
        }      
        
        integrationTipForce = this->sumTipForce(false);
        for(auto forceComp: interactors) forceComp->sumForce(stream);
        
        CudaCheckError();
        
        //Second integration step, does not need noise
        {
            auto groupIterator = pg->getIndexIterator(access::location::gpu);
                
            auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
            auto vel = pd->getVel(access::location::gpu, access::mode::readwrite);
            auto force = pd->getForce(access::location::gpu, access::mode::read);      
                
            auto mass = pd->getMassIfAllocated(access::location::gpu, access::mode::read);
            auto radius = pd->getRadiusIfAllocated(access::location::gpu, access::mode::read);
        
            AFM_integrator_ns::integrateGPU<2><<<Nblocks, Nthreads, 0 , stream>>>(pos.raw(),
                                                                                  vel.raw(),
                                                                                  force.raw(),
                                                                                  mass.raw(),
                                                                                  radius.raw(),
                                                                                  groupIterator,
                                                                                  numberParticles, dt, viscosity, is2D,
                                                                                  noiseAmplitude,
                                                                                  steps, seed);
            CudaCheckError();
            
            //Tip integration
            
            tipPos = this->tipPot->getTipHeight();
            tipVel = this->tipPot->getTipVel();
            
            currentTipForce = -integrationTipForce.z-kTip*(tipPos-tipPosEq);
            tipVel = tipVel + dt*real(0.5)*invTipMass*currentTipForce;
            
            this->tipPot->setTipVel(tipVel);
            
          }

    }
  }
}
