#ifndef PROTEIN_POT_CUH
#define PROTEIN_POT_CUH

#include"ParticleData/ParticleData.cuh"
#include"utils/Box.cuh"
#include"Interactor/Potential/ParameterHandler.cuh"
#include"third_party/type_names.h"

namespace uammd {
namespace Potential {

    struct sasaParameters {
    
        struct InputPairParameters {
            real cutOff;
            real sasa;
        };
    
        struct PairParameters {
            real sasa;
        };
    
        static inline __host__ PairParameters processPairParameters(InputPairParameters in_par) {
    
            return {in_par.sasa};
        }
    
    
    };
    
    class proteinPotential{
        
        public:
        
            using InputPairParameters = typename sasaParameters::InputPairParameters;
        
        protected:
        
            std::shared_ptr<System> sys;
        
            shared_ptr<BasicParameterHandler<sasaParameters>> pairParameters;
            std::string name = "proteinPotential";
            
            real cutOff_;
            
            real debyeLength_;
            real epsilon_;
        
        public:
            proteinPotential(std::shared_ptr<System> sys,real cutOff,real debyeLength,real epsilon):
                             sys(sys),
                             cutOff_(cutOff),
                             debyeLength_(debyeLength),
                             epsilon_(epsilon)
            {
                pairParameters = std::make_shared<BasicParameterHandler<sasaParameters>>();
                sys->log<System::MESSAGE>("[proteinPotential] Initialized");
            }
        
            ~proteinPotential() {
                sys->log<System::MESSAGE>("[proteinPotential] Destroyed");
            }
        
            void setPotParameters(int ti, int tj, InputPairParameters p) {
                sys->log<System::MESSAGE>("[proteinPotential] Type pair %d %d parameters added", ti, tj);
                pairParameters->add(ti, tj, p);
            }
            real getCutOff() {
                return std::max(pairParameters->getCutOff(),cutOff_);
            }
        
            class ForceTransverser {
                public:
                    using pairParameterIterator = typename BasicParameterHandler<sasaParameters>::Iterator;
                protected:
                    pairParameterIterator typeParameters;
            
                    Box box;
                    real cutOff2;
            
                    real4* force;
                    
                    real* C6;
                    real* C12;
                    
                    real* charge;
                    real  invDebyeLength;
                    real  invEpsilon;
                    
                    real* solvE;
                    
                    int* molId;
            
                public:
                    ForceTransverser(pairParameterIterator tp,
                                     Box box,real cutOff,
                                     real4* force,
                                     real* C6,real* C12,
                                     real* charge, real debyeLength, real epsilon,
                                     real* solvE,
                                     int* molId):
                    typeParameters(tp),  
                    box(box),cutOff2(cutOff*cutOff),
                    force(force), 
                    C6(C6),C12(C12),
                    charge(charge),invDebyeLength(real(1)/debyeLength),invEpsilon(real(1)/epsilon),
                    solvE(solvE),
                    molId(molId) {}
            
                    size_t getSharedMemorySize() {
                        return typeParameters.getSharedMemorySize();
                    }
            
                    inline __device__ real4 zero() {return make_real4(0);}
                    inline __device__ void accumulate(real4& total, const real4& current) {total += current;}
                    inline __device__ void set(int pi, const real4 &total) {force[pi] += total;}
            
                    struct Info {
                        real partC6;
                        real partC12;
                        real partChg;
                        real partSolventE;
                        int  partMolId;
                    };
            
                    inline __device__ Info getInfo(int pi) {
                        return {C6[pi],C12[pi],charge[pi],solvE[pi],molId[pi]};
                    }
            
                    inline __device__ real4 compute(const real4 &pi, const real4 &pj,
                                                    const Info &infoi, const Info &infoj) {
            
            
                        if(infoi.partMolId == infoj.partMolId) {
                            return make_real4(0);
                        }
            
                        real3 r21 = this->box.apply_pbc(make_real3(pj)-make_real3(pi));
                        const real r2 = dot(r21, r21);
            
                        if(r2 > cutOff2 or r2 == real(0.0)) return make_real4(0);
                        
                        const real r      = sqrt(r2);
                        const real invr   = real(1)/r;
                        const real invr2  = invr*invr;
                        const real invr6  = invr2*invr2*invr2;
                        const real invr12 = invr6*invr6;
                        
                        //vdW
                        
                        real C6  = infoi.partC6*infoj.partC6;
                        real C12 = infoi.partC12*infoj.partC12;
                        
                        real fmod = (real(2)*C12*invr12-C6*invr6)*real(6)*invr;
                        
                        //ele
                        
                        real chgProduct = infoi.partChg*infoj.partChg;
                        
                        fmod = fmod + chgProduct*exp(-invDebyeLength*r)*(invDebyeLength*r+real(1))*(invr2*invEpsilon);
                        
                        //solvent, gamma=-1
                        
                        real solvE = infoi.partSolventE;
                        auto sasaP = this->typeParameters((int) pi.w, (int) pj.w);
                        
                        fmod = fmod - solvE*sasaP.sasa*invr2;
                        
                        return make_real4(-fmod*(r21/r),0);
                    }
            
                };
            
                //Create and return a transverser
                ForceTransverser getForceTransverser(Box box, shared_ptr<ParticleData> pd) {
                    sys->log<System::DEBUG2>("[proteinPotential] ForceTransverser requested");
                    
                    auto force  = pd->getForce(access::location::gpu, access::mode::readwrite);
                    auto C6     = pd->getC6(access::location::gpu, access::mode::read);
                    auto C12    = pd->getC12(access::location::gpu, access::mode::read);
                    auto charge = pd->getCharge(access::location::gpu, access::mode::read);
                    auto solvE  = pd->getSolvE(access::location::gpu, access::mode::read);
                    auto molId  = pd->getMolId(access::location::gpu, access::mode::read);
            
                    return ForceTransverser(pairParameters->getIterator(),
                                            box,cutOff_,
                                            force.raw(),
                                            C6.raw(),C12.raw(),
                                            charge.raw(),debyeLength_,epsilon_,
                                            solvE.raw(),
                                            molId.raw());
            
                }
            
                class EnergyTransverser {
                public:
                    using pairParameterIterator = typename BasicParameterHandler<sasaParameters>::Iterator;
                protected:
                    pairParameterIterator typeParameters;
            
                    Box box;
                    real cutOff2;
            
                    real* energy;
                    
                    real* C6;
                    real* C12;
                    
                    real* charge;
                    real  invDebyeLength;
                    real  invEpsilon;
                    
                    real* solvE;
                    
                    int* molId;
            
                public:
                    EnergyTransverser(pairParameterIterator tp,
                                      Box box,real cutOff,
                                      real* energy,
                                      real* C6,real* C12,
                                      real* charge, real debyeLength, real epsilon,
                                      real* solvE,
                                      int* molId):
                    typeParameters(tp),  
                    box(box),cutOff2(cutOff*cutOff),
                    energy(energy), 
                    C6(C6),C12(C12),
                    charge(charge),invDebyeLength(real(1)/debyeLength),invEpsilon(real(1)/epsilon),
                    solvE(solvE),
                    molId(molId) {}
            
                    size_t getSharedMemorySize() {
                        return typeParameters.getSharedMemorySize();
                    }
            
                    inline __device__ real zero(){return real(0);}
                    inline __device__ void accumulate(real& total, const real& current){total += current;}
                    inline __device__ void set(int pi, const real &total){energy[pi] += total;}
            
                    struct Info {
                        real partC6;
                        real partC12;
                        real partChg;
                        real partSolventE;
                        int  partMolId;
                    };
            
                    inline __device__ Info getInfo(int pi) {
                        return {C6[pi],C12[pi],charge[pi],solvE[pi],molId[pi]};
                    }
            
                    inline __device__ real compute(const real4 &pi, const real4 &pj,
                                                   const Info &infoi, const Info &infoj) {
            
            
                        if(infoi.partMolId == infoj.partMolId) {
                            return real(0);
                        }
            
                        real3 r21 = this->box.apply_pbc(make_real3(pj)-make_real3(pi));
                        const real r2 = dot(r21, r21);
            
                        if(r2 > cutOff2 or r2 == real(0.0)) return real(0);
                        
                        const real r      = sqrt(r2);
                        const real invr   = real(1)/r;
                        const real invr2  = invr*invr;
                        const real invr6  = invr2*invr2*invr2;
                        const real invr12 = invr6*invr6;
                        
                        //vdW
                        
                        real C6  = infoi.partC6*infoj.partC6;
                        real C12 = infoi.partC12*infoj.partC12;
                        
                        real energy = C12*invr12-C6*invr6;
                        
                        //ele
                        
                        real chgProduct = infoi.partChg*infoj.partChg;
                        
                        energy = energy + chgProduct*exp(-invDebyeLength*r)*invDebyeLength*invr;
                        
                        //solvent, gamma=-1
                        
                        real solvE = infoi.partSolventE;
                        auto sasaP = this->typeParameters((int) pi.w, (int) pj.w);
                        
                        energy = energy + solvE*sasaP.sasa*invr; //Falta termino sasa free
                        
                        return energy;
                    }
            
                };
            
                //Create and return a transverser
                EnergyTransverser getEnergyTransverser(Box box, shared_ptr<ParticleData> pd) {
                    sys->log<System::DEBUG2>("[proteinPotential] EnergyTransverser requested");
                    
                    auto energy = pd->getEnergy(access::location::gpu, access::mode::readwrite);
                    auto C6     = pd->getC6(access::location::gpu, access::mode::read);
                    auto C12    = pd->getC12(access::location::gpu, access::mode::read);
                    auto charge = pd->getCharge(access::location::gpu, access::mode::read);
                    auto solvE  = pd->getSolvE(access::location::gpu, access::mode::read);
                    auto molId  = pd->getMolId(access::location::gpu, access::mode::read);
            
                    return EnergyTransverser(pairParameters->getIterator(),
                                             box,cutOff_,
                                             energy.raw(),
                                             C6.raw(),C12.raw(),
                                             charge.raw(),debyeLength_,epsilon_,
                                             solvE.raw(),
                                             molId.raw());
            
                }
    
    };

}
}
#endif
