#include "uammd.cuh"

#include "Integrator/BrownianDynamics.cuh"
#include "Integrator/VerletNVT.cuh"
#include "Integrator/SteepestDescent.cuh"

#include "Interactor/Potential/RadialPotential.cuh"
#include "Interactor/PairForces.cuh"
#include "Interactor/ExternalForces.cuh"
#include "Interactor/NBodyForces.cuh"

#include <fstream>

using namespace uammd;
using namespace std;

struct Capsid: public ParameterUpdatable{
    
    real epsilon;
    real alphaCut;
    
    real radius2_Z;
    real radius2_XY;
    
    Capsid(real epsilon, real alphaCut ,real radius):epsilon(epsilon),alphaCut(alphaCut),
                                                     radius2_Z(radius*radius),radius2_XY(radius*radius){}

    __device__ __forceinline__ real3 force(const real4 &pos){
        
        real alpha = (pos.x*pos.x+pos.y*pos.y)/(radius2_XY)+
                      pos.z*pos.z/radius2_Z;
        
        if(alpha > alphaCut){
            
            real fmod = (real(1)-alpha)*(real(1)-alpha);
                 fmod = -epsilon/fmod;
                 
            return make_real3(fmod*real(2.0*pos.x/radius2_XY),
                              fmod*real(2.0*pos.y/radius2_XY),
                              fmod*real(2.0*pos.z/radius2_Z));
            
        } else {
            return make_real3(0.0f);
        }
    }
    
    __device__ __forceinline__ real energy(const real4 &pos){
        
        real alpha = (pos.x*pos.x+pos.y*pos.y)/(radius2_XY)+
                      pos.z*pos.z/radius2_Z;
        
        if(alpha > alphaCut){
            
            real energy = real(1)/(real(1)-alpha)-real(1)/(real(1)-alphaCut);
                 energy = epsilon*energy;
                 
            return energy;
            
        } else {
            return real(0);
        }
    }
    
    std::tuple<const real4 *> getArrays(ParticleData *pd){
        auto pos = pd->getPos(access::location::gpu, access::mode::read);
        return std::make_tuple(pos.raw());
    }
  
};

struct potentialFunctor{
    struct InputPairParameters{
        real cutOff;
        real diam;
        real KbT;
        real kappa;
        real B1,B2;
    };
      
    struct PairParameters{
        real cutOff;
        real diam;
        real KbT;
        real kappa;
        real B1,B2;
    };

    static inline __host__ __device__ real force(const real &r2, const PairParameters &params){
        
        real r = sqrt(r2);
        
        if(r >= params.cutOff) return 0;
        
        //DLVO
        
        real fmod  = (params.KbT*params.B1*exp(-params.kappa*(r-params.diam))/r)*(params.kappa*r+real(1))/r2;
        
        //VdW
             
        real D2divr2   = params.diam*params.diam/r2;
        real D8divr8   = D2divr2*D2divr2*D2divr2*D2divr2;
        real D18divr18 = D8divr8*D8divr8*D2divr2;
        real D36divr36 = D18divr18*D18divr18;
             
             fmod += (params.KbT*params.B2/r2)*(real(36)*D36divr36-real(18)*D18divr18);
             
        return -fmod;      
    }
      
    static inline __host__ __device__ real energy(const real &r2, const PairParameters &params){
        
        real r = sqrt(r2);
        
        if(r >= params.cutOff) return 0;
        
        //DLVO
        
        real energy  = params.KbT*params.B1*exp(-params.kappa*(r-params.diam))/r;
        
        //VdW
             
        real D2divr2   = params.diam*params.diam/r2;
        real D8divr8   = D2divr2*D2divr2*D2divr2*D2divr2;
        real D18divr18 = D8divr8*D8divr8*D2divr2;
        real D36divr36 = D18divr18*D18divr18;
             
             energy += params.KbT*params.B2*(D36divr36-D18divr18);
        
        return energy;
    }




    static inline __host__ PairParameters processPairParameters(InputPairParameters in_par){

        PairParameters params;
        
        params.cutOff = in_par.cutOff;
        params.diam   = in_par.diam;
        params.KbT    = in_par.KbT;
        params.kappa  = in_par.kappa;
        params.B1     = in_par.B1;
        params.B2     = in_par.B2;
        
        return params;
        
    }
};

using potential    = Potential::Radial<potentialFunctor>;
using pairforces   = PairForces<potential>;
using nbodyforces  = NBodyForces<potential>;

void outputState(shared_ptr<System> sys,shared_ptr<ParticleData> pd,std::ostream& out){
    sys->log<System::DEBUG1>("[System] Writing to disk...");
    
    auto pos = pd->getPos(access::location::cpu, access::mode::read).raw();
    auto rad = pd->getRadius(access::location::cpu, access::mode::read).raw();
    
    const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
    
    out<<"#"<<std::endl;
    
    fori(0,pd->getNumParticles()){
        real4 p = pos[sortedIndex[i]];
        real3 pPos = {p.x,p.y,p.z};
        int   type = p.w;
        
        real radius = rad[sortedIndex[i]];
        
        out<<pPos<<" "<<radius<<" "<<type<<endl;
    }
}

int main(int argc, char *argv[]){

    int  N = 200;
    Box  box({100,100,100});
    
    real partDiam   =  2;
    real kappa      =  3/partDiam;
    real B1         =  20;
    real B2         =  60;
    
    real cutOff     = partDiam*2.5;
    
    real radius = 15;
    real epsilon  = 1;
    real alphaCut = 0.9;
    
    real h = 0.1;
    real epsilonST = 10;
    real saturationForce     = 100;
    real maxSteps = 100000;
    
    real temperature         = 1;
    real viscosity           = 1;
    real hydrodynamicRadius  = 1;
    real dt                  = 0.01;
    
    int nSteps     = 10000000;
    int printSteps = 10000;
    
    ofstream out("state.sp");
    
    //////////////////////////////////////////////////////////////////////
    
    auto sys = make_shared<System>();
    
    ullint seed = 0xf31337Bada55D00dULL;
    sys->rng().setSeed(seed);
    
    auto pd = make_shared<ParticleData>(N, sys);
    
    ////////////////////////////////////////////////////////////////////
    
    //Initial conditions
    {
        auto pos = pd->getPos(access::location::cpu, access::mode::write).raw();
        auto rad = pd->getRadius(access::location::cpu, access::mode::write).raw();
        
        bool accp = true;
      
        fori(0,N){
            real3 spherePoint  = {real(sys->rng().uniform(-radius,radius)),
                                  real(sys->rng().uniform(-radius,radius)),
                                  real(sys->rng().uniform(-radius,radius))};
            
            real d = spherePoint.x*spherePoint.x+
                     spherePoint.y*spherePoint.y+
                     spherePoint.z*spherePoint.z;
                 
                 d = sqrt(d);
            
            if(d < radius){
                forj(0,i){
                    real3 dst = spherePoint-make_real3(pos[j]);
                    if(sqrt(dot(dst,dst)) < partDiam){
                        accp = false;
                        break;
                    }
                }
                
                if(accp == true){
                    pos[i] = make_real4(spherePoint,1);
                    rad[i] = partDiam*real(0.5);
                } else {
                    i--;
                    accp = true;
                }
            } else {
                i--;
            }
        }
    }
    
    auto pg = make_shared<ParticleGroup>(pd, sys, "All");
    
    ////////////////////////////////////////////////////////////////////
    
    auto capsidPot   = make_shared<Capsid>(epsilon,alphaCut,radius);
    auto capsidForce = make_shared<ExternalForces<Capsid>>(pd, pg, sys,capsidPot);

    ////////////////////////////////////////////////////////////////////
    
    auto pot = make_shared<potential>(sys);
    
    {
        potential::InputPairParameters params;
        
        params.cutOff = cutOff;
        params.diam   = partDiam;
        params.KbT    = temperature;
        params.kappa  = kappa;
        params.B1     = B1;
        params.B2     = B2;
        
        pot->setPotParameters(0, 0, params);
    }
    
    //nbodyforces::Parameters params;
    //params.box = box; 
    //auto partForces = make_shared<nbodyforces>(pd, pg, sys, params, pot);
    
    pairforces::Parameters params;
    params.box = box; 
    auto partForces = make_shared<pairforces>(pd, pg, sys, params, pot);
    
    ////////////////////////////////////////////////////////////////////
    
    //SteepestDescent::Parameters STpar;
    //STpar.h = h;
    //STpar.epsilon = epsilonST;
    //STpar.saturationForce = saturationForce;
    //STpar.maxSteps = maxSteps;
    //
    //auto st = make_shared<SteepestDescent>(pd, pg, sys, STpar);
    //
    //st->addInteractor(capsidForce);
    //st->addInteractor(partForces);
    //
    //outputState(sys,pd,out);
    //
    //while(st->minimizationStep());
    
    ////////////////////////////////////////////////////////////////////
    
    //BD::EulerMaruyama::Parameters parBD;
    //parBD.temperature        = temperature;
    //parBD.viscosity          = viscosity;
    //parBD.hydrodynamicRadius = hydrodynamicRadius;
    //parBD.dt                 = dt;
    //
    //auto bd = make_shared<BD::EulerMaruyama>(pd, pg, sys, parBD); 
    //
    //bd->addInteractor(capsidForce);
    //bd->addInteractor(partForces);
    
    
    VerletNVT::GronbechJensen::Parameters parGJ;
    parGJ.temperature = temperature;
    parGJ.dt = dt;
    parGJ.viscosity = viscosity;
    
    auto gj = make_shared<VerletNVT::GronbechJensen>(pd, pg, sys, parGJ);
    
    gj->addInteractor(capsidForce);
    gj->addInteractor(partForces);
    
    outputState(sys,pd,out);
    
    ////////////////////////////////////////////////////////////////////
    
    
    Timer tim;
    tim.tic();
    
    sys->log<System::MESSAGE>("SIMULATION STARTS!!!!");
    
    //Run the simulation
    forj(0,nSteps){
        
        //bd->forwardTime();
        gj->forwardTime();
    
        if(j%printSteps ==0) {
            sys->log<System::MESSAGE>("Progress: %.3f%%",100.0*(real(j)/nSteps));
            outputState(sys,pd,out);
        }
        //if(j%sotSteps   == 0){ pd->sortParticles(); }
    }
    
    auto totalTime = tim.toc();
    sys->log<System::MESSAGE>("mean FPS: %.2f", nSteps/totalTime);
    sys->finish();
  
}
