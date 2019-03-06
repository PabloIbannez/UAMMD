#ifndef AFM_POT_CUH
#define AFM_POT_CUH

#include<thrust/device_vector.h>

#include"ParticleData/ParticleData.cuh"
#include"utils/Box.cuh"
#include"third_party/type_names.h"


namespace uammd {
namespace Potential {
    
    class AFMPotential{
        
        protected:
        
            std::shared_ptr<System> sys;

            std::string name = "AFMPotential";
            
            real cutOff_;
            real epsilon_;
        
        public:
            AFMPotential(std::shared_ptr<System> sys,real cutOff,real epsilon):
                             sys(sys),
                             cutOff_(cutOff),
                             epsilon_(epsilon)
            {
                sys->log<System::MESSAGE>("[AFMPotential] Initialized");
            }
        
            ~AFMPotential() {
                sys->log<System::MESSAGE>("[AFMPotential] Destroyed");
            }
        
            real getCutOff() {
                return cutOff_;
            }
        
            class ForceTransverser {
                
                protected:
            
                    Box box;
                    real cutOff2;
                    real epsilon;
            
                    real4* force;
                    
                    real* radius;
                    int* molId;
            
                public:
                    ForceTransverser(Box box,real cutOff,
                                     real epsilon,
                                     real4* force,
                                     real* radius,
                                     int* molId):
                    box(box),cutOff2(cutOff*cutOff),
                    epsilon(epsilon),
                    force(force), 
                    radius(radius),
                    molId(molId) {}
            
                    inline __device__ real4 zero() {return make_real4(0);}
                    inline __device__ void accumulate(real4& total, const real4& current) {total += current;}
                    inline __device__ void set(int pi, const real4 &total) {force[pi] += total;}
            
                    struct Info {
                        real partRadius;
                        int  partMolId;
                    };
            
                    inline __device__ Info getInfo(int pi) {
                        return {radius[pi],molId[pi]};
                    }
            
                    inline __device__ real4 compute(const real4 &pi, const real4 &pj,
                                                    const Info &infoi, const Info &infoj) {
            
            
                        if(infoi.partMolId == infoj.partMolId) {
                            return make_real4(0);
                        }
            
                        real3 r21 = this->box.apply_pbc(make_real3(pj)-make_real3(pi));
                        const real r2 = dot(r21, r21);
            
                        if(r2 > cutOff2 or r2 == real(0.0)) return make_real4(0);
                        
                        const real effDiam = (infoi.partRadius+infoj.partRadius);
                        
                        const real r       = sqrt(r2);
                        const real invr2   = real(1.0)/r2;
                        const real Dinvr2  = effDiam*effDiam*invr2;
                        const real Dinvr6  = Dinvr2*Dinvr2*Dinvr2;
                        const real Dinvr12 = Dinvr6*Dinvr6;
                        
			            real fmod = 0;
			
                        //wca
                        
                        fmod = fmod + ((r<real(1.122462)*effDiam)?real(4*6)*epsilon*(real(2)*Dinvr12-Dinvr6):real(0));
                        
                        //printf("%f %f %f %f %f\n",fmod,r,real(1.122462)*effRadius,infoi.partRadius,infoj.partRadius);
                        
                        return make_real4(-fmod*(r21*invr2),0);
                    }
            
            };
            
            //Create and return a transverser
            ForceTransverser getForceTransverser(Box box, shared_ptr<ParticleData> pd) {
                sys->log<System::DEBUG2>("[proteinPotential] ForceTransverser requested");
                
                auto force  = pd->getForce(access::location::gpu, access::mode::readwrite);
                auto radius = pd->getRadius(access::location::gpu, access::mode::read);
                auto molId  = pd->getMolId(access::location::gpu, access::mode::read);
            
                return ForceTransverser(box,cutOff_,
                                        epsilon_,
                                        force.raw(),
                                        radius.raw(),
                                        molId.raw());
            
            }
            
            class EnergyTransverser {
                
                protected:
            
                    Box box;
                    real cutOff2;
                    real epsilon;
            
                    real* energy;
                    
                    real* radius;
                    int* molId;
            
                public:
                    EnergyTransverser(Box box,real cutOff,
                                     real epsilon,
                                     real* energy,
                                     real* radius,
                                     int* molId):
                    box(box),cutOff2(cutOff*cutOff),
                    epsilon(epsilon),
                    energy(energy), 
                    radius(radius),
                    molId(molId) {}
            
                    inline __device__ real zero() {return real(0);}
                    inline __device__ void accumulate(real& total, const real& current) {total += current;}
                    inline __device__ void set(int pi, const real &total) {energy[pi] += total;}
            
                    struct Info {
                        real partRadius;
                        int  partMolId;
                    };
            
                    inline __device__ Info getInfo(int pi) {
                        return {radius[pi],molId[pi]};
                    }
            
                    inline __device__ real compute(const real4 &pi, const real4 &pj,
                                                   const Info &infoi, const Info &infoj) {
            
            
                        if(infoi.partMolId == infoj.partMolId) {
                            return real(0);
                        }
            
                        real3 r21 = this->box.apply_pbc(make_real3(pj)-make_real3(pi));
                        const real r2 = dot(r21, r21);
            
                        if(r2 > cutOff2 or r2 == real(0.0)) return real(0);
                        
                        const real effDiam = (infoi.partRadius+infoi.partRadius);
                        
                       const real invr2   = real(1.0)/r2;
                       const real Dinvr2  = effDiam*effDiam*invr2;
                       const real Dinvr6  = Dinvr2*Dinvr2*Dinvr2;
                       const real Dinvr12 = Dinvr6*Dinvr6;
                        
			            real energy = 0;
			
                        //wca
                        
                        energy = energy + epsilon*Dinvr12;
                        
                        return energy;
                    }
            
                };
            
                //Create and return a transverser
                EnergyTransverser getEnergyTransverser(Box box, shared_ptr<ParticleData> pd) {
                    sys->log<System::DEBUG2>("[proteinPotential] ForceTransverser requested");
                    
                    auto energy = pd->getEnergy(access::location::gpu, access::mode::readwrite);
                    auto radius = pd->getRadius(access::location::gpu, access::mode::read);
                    auto molId  = pd->getMolId(access::location::gpu, access::mode::read);
            
                    return EnergyTransverser(box,cutOff_,
                                             epsilon_,
                                             energy.raw(),
                                             radius.raw(),
                                             molId.raw());
            
                }
                
            
    };

}
}
#endif
