#ifndef REACTION_DIFFUSION_FHD_CUH
#define REACTION_DIFFUSION_FHD_CUH
#include "global/defines.h"
#include "../Integrator.cuh"

namespace uammd{
  namespace reactionDiffusin_FHD{    
      
    class reactionDiffusin_FHD: public Integrator{
        
        public:
            
            struct Parameters{
                
                int3 gridSize;
                real cellSize;
            };
        
        private:
            
            reactionDiffusin_FHD(shared_ptr<ParticleData> pd,
                                 shared_ptr<ParticleGroup> pg,
                                 shared_ptr<System> sys,
                                 Parameters par):
                                 Integrator(pd, pg, sys, "[reactionDiffusion::FHD]"){
                
                
                
            }
            
        public:
            
            thrust::device_vector<real3> GWN_staggedGrid;
            
            thrust::device_vector<real> gridValues;
            thrust::device_vector<real> gridValuesAlt;
            
    };
}}

#endif
