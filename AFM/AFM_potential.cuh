#ifndef AFM_POT_CUH
#define AFM_POT_CUH

#include<thrust/device_vector.h>

#include"ParticleData/ParticleData.cuh"
#include"utils/Box.cuh"
#include"third_party/type_names.h"

#include"Interactor/Potential/ParameterHandler.cuh"

namespace uammd {
namespace Potential {
    
    namespace SASAmodel{
        //SASA functions
        struct A{
            __device__ static real SASAweight(real SASAratio){
                return real(1.0);
            }
        };
        
        struct B{
            __device__ static real SASAweight(real SASAratio){
                real w = tanhf(real(10)*tanf(SASAratio*M_PI_2));
                return w<real(0.0)?real(1.0):w;
            }
        };
        
        struct C{
            __device__ static real SASAweight(real SASAratio){
                real w = tanhf(real(5)*tanf(SASAratio*M_PI_2));
                return w<real(0.0)?real(1.0):w;
            }
        };
        
        struct D{
            __device__ static real SASAweight(real SASAratio){
                real w = tanhf(real(2)*tanf(SASAratio*M_PI_2));
                return w<real(0.0)?real(1.0):w;
            }
        };
        
        struct E{
            __device__ static real SASAweight(real SASAratio){
                real w = (real(1.0)+tanhf(real(2)*tanf(SASAratio*M_PI_2)))/real(2.0);
                return w<real(0.0)?real(1.0):w;
            }
        };
        
        struct F{
            __device__ static real SASAweight(real SASAratio){
                real w = (real(1.0)+tanhf(tanf(SASAratio*M_PI_2)))/real(2.0);
                return w<real(0.0)?real(1.0):w;
            }
        };
    }
    
    //2^(1/6) effective radius factor
    constexpr real _EFFR_F = 1.122462;
    constexpr real _EFFR_F2 = 1.259921;
    
    struct ProteinProteinParameters{
    
        struct InputPairParameters{
            real cutOff, epsilon;
        };
            
        struct PairParameters{
            real cutOff2;
            real epsilon;
        };
        
        static inline __host__ PairParameters processPairParameters(InputPairParameters in_par){
    
            PairParameters params;
            
            params.cutOff2 = in_par.cutOff*in_par.cutOff;
            params.epsilon = in_par.epsilon;
            
            return params;
        }
    };
    
    template<class SASAmodel>
    class AFMPotential{
        
        using ParameterHandle = BasicParameterHandler<ProteinProteinParameters>;
        using pairParameterIterator = typename ParameterHandle::Iterator;
        
        using InputPairParameters = typename ProteinProteinParameters::InputPairParameters;
        
        protected:
        
            std::shared_ptr<System> sys;
            
            shared_ptr<ParameterHandle> pairParameters;
            
            std::string name = "AFMPotential";
            
            ////////////////////////////////////////////////////////////
            
            real kBT;
            
            real epsilon_wca;
            
            std::string partInteractionData;
            
            real epsilon_0;
            real lambda;
            real gamma;
            
            real f;
            real debyeLenght;
            real dielectricConstant;
            
            real finvD; //f/(dielectricConstant)
            real invXi; //1/debyeLenght
            
            real cutOff;
            real cutOff2;
            
            ////////////////////////////////////////////////////////////
            
            bool allSteric = false;
        
        public:
        
            struct Parameters{
                
                real kBT;
            
                real epsilon_wca;
                
                std::string partInteractionData;
                
                real epsilon_0;
                real lambda;
                real gamma;
                
                real f;
                real debyeLenght;
                real dielectricConstant;
                
                real cutOff;
            };
        
            //f: electric coversion factor
            AFMPotential(std::shared_ptr<System> sys,Parameters par): sys(sys),
                                                                      kBT(par.kBT),
                                                                      epsilon_wca(par.epsilon_wca),
                                                                      partInteractionData(par.partInteractionData),
                                                                      epsilon_0(par.epsilon_0),lambda(par.lambda),gamma(par.gamma),
                                                                      f(par.f),debyeLenght(par.debyeLenght),dielectricConstant(par.dielectricConstant),
                                                                      finvD(f/dielectricConstant),invXi(real(1.0)/debyeLenght),
                                                                      cutOff(par.cutOff),cutOff2(par.cutOff*par.cutOff)
            {
                sys->log<System::MESSAGE>("[AFMPotential] Initialized");
                
                pairParameters = std::make_shared<ParameterHandle>();
                
                {
                    std::ifstream partInt(partInteractionData);
                    
                    std::stringstream ss;
                    std::string line;
                    
                    InputPairParameters inputParBuffer;
                    
                    int  i,j;
                    real epsilonBuffer;
                    
                    while(std::getline(partInt,line)) {
                        
                        ss.str(line);
                        
                        ss >> i >> j >> epsilonBuffer;
                        
                        epsilonBuffer = lambda*kBT*(epsilonBuffer - epsilon_0);
                        
                        inputParBuffer = {real(0),epsilonBuffer};
                        
                        this->setPotParameters(i,j,inputParBuffer);
                    }
                    
                }
            }
        
            ~AFMPotential() {
                sys->log<System::MESSAGE>("[AFMPotential] Destroyed");
            }
            
            void setPotParameters(int ti, int tj, InputPairParameters p){
                sys->log<System::MESSAGE>("[AFMPotential] Type pair %d %d parameters added", ti, tj);
                pairParameters->add(ti, tj, p);
            }
        
            real getCutOff() {
                return cutOff;
            }
            
            void setAllSteric(bool allS){
                allSteric = allS;
            }
            
            template<class SASA>
            class ForceTransverser {
                
                protected:
                
                    pairParameterIterator typeParameters;
            
                    Box box;
                    real epsilon_wca;
                    
                    real gamma;
                    
                    real finvD;
                    real invXi;
                    
                    real cutOff2;
            
                    real4* force;
                    
                    int*  molId;
                    real* radius;
                    real* charge;
                    real* SASAratio;
                    
                    bool allSteric;
                    
                public:
                    
                    ForceTransverser(pairParameterIterator tp,
                                     Box box,
                                     real epsilon_wca,
                                     real gamma,
                                     real finvD,real invXi,
                                     real cutOff2,
                                     real4* force,
                                     int* molId,
                                     real* radius,
                                     real* charge,
                                     real* SASAratio,
                                     bool allSteric):
                    typeParameters(tp),
                    box(box),
                    epsilon_wca(epsilon_wca),
                    gamma(gamma),
                    finvD(finvD),invXi(invXi),
                    cutOff2(cutOff2),
                    force(force), 
                    molId(molId),
                    radius(radius),
                    charge(charge),
                    SASAratio(SASAratio),
                    allSteric(allSteric) {}
                    
                    
                    size_t getSharedMemorySize(){
                        return typeParameters.getSharedMemorySize();
                    }   
            
                    inline __device__ real4 zero() {return make_real4(0);}
                    inline __device__ void accumulate(real4& total, const real4& current) {total += current;}
                    inline __device__ void set(int pi, const real4 &total) {force[pi] += total;}
            
                    struct Info {
                        int  partMolId;
                        real radius;
                        real charge;
                        real SASAratio;
                    };
            
                    inline __device__ Info getInfo(int pi) {
                        return {molId[pi],radius[pi],charge[pi],SASAratio[pi]};
                    }
            
                    inline __device__ real4 compute(const real4 &pi, const real4 &pj,
                                                    const Info &infoi, const Info &infoj) {
                        
                        if(infoi.partMolId == infoj.partMolId) {
                            return make_real4(0);
                        }
            
                        real3 r21 = this->box.apply_pbc(make_real3(pj)-make_real3(pi));
                        const real r2 = dot(r21, r21);
                        
                        auto params = this->typeParameters((int) pi.w, (int) pj.w);
                        
                        if(r2 > cutOff2 or r2 == real(0.0)) return make_real4(0);
                        
                        real effDiam  = gamma*(infoi.radius + infoj.radius);
                        real effDiam2 = effDiam*effDiam;
                        
                        const real invr2   = real(1.0)/r2;
                        const real Dinvr2  = effDiam2*invr2;
                        const real Dinvr6  = Dinvr2*Dinvr2*Dinvr2;
                        const real Dinvr12 = Dinvr6*Dinvr6;
                        
                        
                        if((infoi.partMolId < int(0) and infoj.partMolId < int(0)) or allSteric){
                            
                            if(r2<_EFFR_F2*effDiam2){
                                
                                real fmod = real(4*6)*epsilon_wca*(real(2)*Dinvr12-Dinvr6);
                                
                                return make_real4(-fmod*(r21*invr2),0);
                                
                            } else {
                                
                                return make_real4(0);
                                
                            }
                            
                        } else {
                            
                            
                            real SASAfactor = SASA::SASAweight(infoi.SASAratio)*SASA::SASAweight(infoj.SASAratio);
                            if(SASAfactor == real(0.0)){
                                return make_real4(0);
                            }
                            
                            real fmod = real(0);
                            
                            if(params.epsilon < real(0)){
        
                                fmod += real(4*6)*fabs(params.epsilon)*(real(2)*Dinvr12-Dinvr6);
                        
                            } else {
                                
                                if(r2<_EFFR_F2*effDiam2){
                                    fmod += real(4*6)*params.epsilon*(real(2)*Dinvr12-Dinvr6);
                                } else {
                                    fmod -= real(4*6)*params.epsilon*(real(2)*Dinvr12-Dinvr6);
                                }
                            }
                            
                            real chgProduct = infoi.charge*infoj.charge;
                            if(chgProduct != real(0)){
                                
                                    const real A = chgProduct*finvD;
                                    
                                    real r    = sqrt(r2);
                                    real invr = real(1.0)/r;
                                
                                    fmod += A*exp(-r*invXi)*(invXi+invr);
                            }
                            
                            fmod = fmod*SASAfactor;
                            
                            return make_real4(-fmod*(r21*invr2),0);
                            
                        }
                        
                    }
            
            };
            
            //Create and return a transverser
            ForceTransverser<SASAmodel> getForceTransverser(Box box, shared_ptr<ParticleData> pd) {
                sys->log<System::DEBUG2>("[proteinPotential] ForceTransverser requested");
                
                auto force     = pd->getForce(access::location::gpu, access::mode::readwrite);
                auto molId     = pd->getMolId(access::location::gpu, access::mode::read);
                auto radius    = pd->getRadius(access::location::gpu, access::mode::read);
                auto charge    = pd->getCharge(access::location::gpu, access::mode::read);
                auto SASAratio = pd->getSASAratio(access::location::gpu, access::mode::read);
            
                return ForceTransverser<SASAmodel>(pairParameters->getIterator(),
                                                   box,
                                                   epsilon_wca,
                                                   gamma,
                                                   finvD,invXi,
                                                   cutOff2,
                                                   force.raw(),
                                                   molId.raw(),
                                                   radius.raw(),
                                                   charge.raw(),
                                                   SASAratio.raw(),
                                                   allSteric);
            }
            
            /*
            class EnergyTransverser {
                
                protected:
                
                    pairParameterIterator typeParameters;
            
                    Box box;
                    real epsilon_wca;
            
                    real* energy;

                    int* molId;
                    real* radius;
                    real* charge;
            
                public:
                    EnergyTransverser(pairParameterIterator tp,
                                      Box box,
                                      real epsilon_wca,
                                      real* energy,
                                      int* molId,
                                      real* radius,
                                      real* charge):
                    typeParameters(tp),
                    box(box),
                    epsilon_wca(epsilon_wca),
                    energy(energy),
                    molId(molId),
                    radius(radius),
                    charge(charge){}
                    
                    size_t getSharedMemorySize(){
                        return typeParameters.getSharedMemorySize();
                    }
            
                    inline __device__ real zero() {return real(0);}
                    inline __device__ void accumulate(real& total, const real& current) {total += current;}
                    inline __device__ void set(int pi, const real &total) {energy[pi] += total;}
            
                    struct Info {
                        int  partMolId;
                        real radius;
                        real charge;
                    };
            
                    inline __device__ Info getInfo(int pi) {
                        return {molId[pi],radius[pi],charge[pi]};
                    }
            
                    inline __device__ real compute(const real4 &pi, const real4 &pj,
                                                   const Info &infoi, const Info &infoj) {
            
            
                        if(infoi.partMolId == infoj.partMolId) {
                            return real(0);
                        }
                        
                        return real(0);
                    }
            
                };
            
                //Create and return a transverser
                EnergyTransverser getEnergyTransverser(Box box, shared_ptr<ParticleData> pd) {
                    sys->log<System::DEBUG2>("[proteinPotential] ForceTransverser requested");
                    
                    auto energy = pd->getEnergy(access::location::gpu, access::mode::readwrite);
                    auto molId  = pd->getMolId(access::location::gpu, access::mode::read);
                    auto radius = pd->getRadius(access::location::gpu, access::mode::read);
                    auto charge = pd->getCharge(access::location::gpu, access::mode::read);
            
                    return EnergyTransverser(pairParameters->getIterator(),
                                             box,
                                             epsilon_wca,
                                             energy.raw(),
                                             molId.raw(),
                                             radius.raw(),
                                             charge.raw());
            
                }
            */
                
            
    };

}
}
#endif
