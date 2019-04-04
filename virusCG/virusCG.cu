#include "uammd.cuh"

#include "Integrator/BrownianDynamics.cuh"
#include "Integrator/VerletNVT.cuh"
#include "Integrator/SteepestDescent.cuh"

#include "Interactor/Potential/RadialPotential.cuh"
#include "Interactor/PairForces.cuh"
#include "Interactor/ExternalForces.cuh"
#include "Interactor/NBodyForces.cuh"

#include "third_party/cub/cub.cuh"

#include <fstream>

using namespace uammd;
using namespace std;

#define GEL_POT

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

struct virialFunctor{
    int2* pairIterator;
    
    real4* pos;
    real4* force;
    
    virialFunctor(int2* pairIterator,real4* pos,real4* force):pairIterator(pairIterator),pos(pos),force(force){}
    
    __host__ __device__ __forceinline__
    real operator()(const int &index) const {
    
        int i = pairIterator[index].x;
        int j = pairIterator[index].y;
        //return dot(make_real3(force[i]),make_real3(pos[j])); 
        return make_real3(force[i]).z*make_real3(pos[j]).z; 
    }
};

class pressureMeasuring{
    
    private:
        
        shared_ptr<System> sys;
        
        shared_ptr<ParticleData> pd;
        shared_ptr<ParticleGroup> pg;
        
        std::vector<shared_ptr<Interactor>> interactors;
        
        cudaStream_t stream;
        
        //
        thrust::device_vector<int2> pairIterator;
        
        //cub reduction variables force
        real3*   totalForce;
        real*    totalVirial;
        void*    cubTempStorageSum = NULL;
        size_t   cubTempStorageSizeSum = 0;
        
        void setUpCubReduction(){
            
            size_t   cubTempStorageSizeForce  = 0;
            size_t   cubTempStorageSizeVirial = 0;
            
            //common
            cub::CountingInputIterator<int> countingIterator(0);
            auto groupIterator = pg->getIndexIterator(access::location::gpu);
            
            //force
            auto force = pd->getForce(access::location::gpu, access::mode::read);
            forceFunctor fF(groupIterator,force.raw());
        
            cub::TransformInputIterator<real3, forceFunctor, cub::CountingInputIterator<int>> forceSumIterator(countingIterator,fF);
            
            cub::DeviceReduce::Sum(cubTempStorageSum, cubTempStorageSizeForce, forceSumIterator, totalForce, pg->getNumberParticles(), stream);
            
            //virial
            auto groupIteratorCPU = pg->getIndexIterator(access::location::cpu);
            int i,j;
            for(i = 0  ;i<pg->getNumberParticles();i++){
            for(j = i+1;j<pg->getNumberParticles();j++){
                pairIterator.push_back({groupIteratorCPU[i],groupIteratorCPU[j]});
            }}
            
            int pairNum = pairIterator.size();
            
            auto pos = pd->getPos(access::location::gpu, access::mode::read);
            virialFunctor vF(thrust::raw_pointer_cast(pairIterator.data()),pos.raw(),force.raw());
            
            cub::TransformInputIterator<real, virialFunctor, cub::CountingInputIterator<int>> viralSumIterator(countingIterator,vF);
            
            cub::DeviceReduce::Sum(cubTempStorageSum, cubTempStorageSizeVirial, viralSumIterator, totalVirial, pairNum, stream);
            
            //
            cubTempStorageSizeSum = std::max(cubTempStorageSizeForce,cubTempStorageSizeVirial);
            cudaMalloc(&cubTempStorageSum, cubTempStorageSizeSum);
        }
    
    public:
    
        pressureMeasuring(shared_ptr<System> sys,
                          shared_ptr<ParticleData> pd,
                          shared_ptr<ParticleGroup> pg):sys(sys),pd(pd),pg(pg){
          
            sys->log<System::MESSAGE>("[pressureMeasuring] Created.");
            
            cudaStreamCreate(&stream);
            
            cudaMallocManaged((void**)&totalForce,sizeof(real3));
            cudaMallocManaged((void**)&totalVirial,sizeof(real));
            this->setUpCubReduction();
        }
        
        ~pressureMeasuring(){
          
            sys->log<System::MESSAGE>("[pressureMeasuring] Destroyed.");
            
            cudaFree(totalForce);
            cudaFree(cubTempStorageSum);
            cudaStreamDestroy(stream);
        }
        
        void addInteractor(shared_ptr<Interactor> an_interactor){
            interactors.emplace_back(an_interactor);      
        }
    
        real3 sumForce(){
          
            int numberParticles = pg->getNumberParticles();
            auto groupIterator = pg->getIndexIterator(access::location::gpu);
            
            int Nthreads=128;
            int Nblocks=numberParticles/Nthreads + ((numberParticles%Nthreads)?1:0);
            
            {
                auto force = pd->getForce(access::location::gpu, access::mode::readwrite);
                fillWithGPU<<<Nblocks, Nthreads>>>(force.raw(), groupIterator, make_real4(0), numberParticles);
            }
            
            for(auto forceComp: interactors) forceComp->sumForce(stream);
            cudaDeviceSynchronize();
            
            cub::CountingInputIterator<int> countingIterator(0);
            
            //force
            auto force = pd->getForce(access::location::gpu, access::mode::read);
            forceFunctor fF(groupIterator,force.raw());
            
            cub::TransformInputIterator<real3, forceFunctor, cub::CountingInputIterator<int>> forceSumIterator(countingIterator,fF);
            
            cub::DeviceReduce::Sum(cubTempStorageSum, cubTempStorageSizeSum, forceSumIterator, totalForce, pg->getNumberParticles(), stream);
            cudaStreamSynchronize(stream);
            
            return *totalForce;
        }
        
        real sumVirial(){
          
            int pairNum = pairIterator.size();
            int numberParticles = pg->getNumberParticles();
            auto groupIterator = pg->getIndexIterator(access::location::gpu);
            
            int Nthreads=128;
            int Nblocks=pairNum/Nthreads + ((pairNum%Nthreads)?1:0);
            
            {
                auto force = pd->getForce(access::location::gpu, access::mode::readwrite);
                fillWithGPU<<<Nblocks, Nthreads>>>(force.raw(), groupIterator, make_real4(0), numberParticles);
            }
            
            for(auto forceComp: interactors) forceComp->sumForce(stream);
            cudaDeviceSynchronize();
            
            cub::CountingInputIterator<int> countingIterator(0);
            
            //force
            auto force = pd->getForce(access::location::gpu, access::mode::read);
            auto pos = pd->getPos(access::location::gpu, access::mode::read);
            virialFunctor vF(thrust::raw_pointer_cast(pairIterator.data()),pos.raw(),force.raw());
            
            cub::TransformInputIterator<real, virialFunctor, cub::CountingInputIterator<int>> viralSumIterator(countingIterator,vF);
            
            cub::DeviceReduce::Sum(cubTempStorageSum, cubTempStorageSizeSum, viralSumIterator, totalVirial, pairNum, stream);
            cudaStreamSynchronize(stream);
            
            return *totalVirial;
        }
};

double ndOrderEquationPositive(double a,double b,double c){
    return (-b+std::sqrt(b*b-double(4*a*c)))/double(2*a);
}

template<int niter>
double computeRadiusXY(real area,real radiusZ){
    
    double p = 1.6075;
    
    double radiusXY = ndOrderEquationPositive(1.0,2.0*std::pow(radiusZ,p),-3.0*std::pow(area/(4.0*M_PI),p));
           radiusXY = std::pow(radiusXY,1.0/p);
    double epsilon;
    
    //for(int i=0;i<niter;i++){
    //    epsilon  = 1.0-std::pow(radiusXY/radiusZ,int(2));
    //    radiusXY = std::sqrt((1.0/(2.0*M_PI))*(area-M_PI*(radiusZ*radiusZ/epsilon)*std::log((1.0+epsilon)/(1.0-epsilon))));
    //}
    
    return radiusXY;
}

struct Capsid: public ParameterUpdatable{
    
    real epsilon;
    real alphaCut;
    
    real radius2_Z;
    real radius2_XY;
    
    real area;
    
    Capsid(real epsilon, real alphaCut ,real radius):epsilon(epsilon),alphaCut(alphaCut),
                                                     radius2_Z(radius*radius),radius2_XY(radius*radius),area(4.0*M_PI*radius*radius){}

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
    
    real getRadiusZ(){
        return std::sqrt(radius2_Z);
    }
    
    real getRadiusXY(){
        return std::sqrt(radius2_XY);
    }
    
    void setRadiusZ(real newHeight){
        radius2_Z  = newHeight*newHeight;
        radius2_XY = computeRadiusXY<100>(area,newHeight);
        radius2_XY = radius2_XY*radius2_XY;
    }
  
};

struct GEL{
    
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
        
        if(r >= params.cutOff) return real(0);
        
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
        
        if(r >= params.cutOff) return real(0);
        
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

struct WCA{
    
    struct InputPairParameters{
        real cutOff;
        real diamEff;
        real epsilon;
    };
      
    struct PairParameters{
        real diamEff;
        real epsilon;
    };

    static inline __host__ __device__ real force(const real &r2, const PairParameters &params){
        
        real r = sqrt(r2);
        
        if(r < real(1.122462)*params.diamEff){
            
            real inv2  = (params.diamEff*params.diamEff)/r2;
            real inv6  = inv2*inv2*inv2;
            real inv12 = inv6*inv6;
            
            real fmod = real(4)*real(6)*params.epsilon*(real(2)*inv12-inv6)/r;
            
            return -fmod;     
        } else {
            return real(0);
        }
    }
      
    static inline __host__ __device__ real energy(const real &r2, const PairParameters &params){
        
        real r = sqrt(r2);
        
        if(r < real(1.122462)*params.diamEff){
            
            real inv2  = (params.diamEff*params.diamEff)/r2;
            real inv6  = inv2*inv2*inv2;
            real inv12 = inv6*inv6;
            
            return real(4)*params.epsilon*(inv12-inv6)+params.epsilon;
        } else {
            return real(0);
        }
    }




    static inline __host__ PairParameters processPairParameters(InputPairParameters in_par){

        PairParameters params;
        
        params.diamEff  = in_par.diamEff;
        params.epsilon  = in_par.epsilon;
        
        return params;
        
    }
};

#ifdef GEL_POT
using potential    = Potential::Radial<GEL>;
#else
using potential    = Potential::Radial<WCA>;
#endif
using pairforces   = PairForces<potential>;
using nbodyforces  = NBodyForces<potential>;

void outputState(shared_ptr<System> sys,shared_ptr<ParticleData> pd,std::ostream& out,real radiusZ, real radiusXY){
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
    
    out<<real3({0,0,  radiusZ})<<" "<<1<<" "<<3<<endl;
    out<<real3({0,0, -radiusZ})<<" "<<1<<" "<<3<<endl;
    out<<real3({ radiusXY,0,0})<<" "<<1<<" "<<5<<endl;
    out<<real3({-radiusXY,0,0})<<" "<<1<<" "<<5<<endl;
    out<<real3({0, radiusXY,0})<<" "<<1<<" "<<5<<endl;
    out<<real3({0,-radiusXY,0})<<" "<<1<<" "<<5<<endl;
}

int main(int argc, char *argv[]){
    
    int  N = 200;
    Box  box({100,100,100});
    real V = box.boxSize.x*box.boxSize.y*box.boxSize.z;
    
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
    
    int sortSteps  = 1000;
    
    int decreaseSteps  = 10000;
    
    int measuringSteps = 1000;
    
    ofstream outState("state.sp");
    ofstream outMeasured("measured.dat");
    
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
    
    auto pM = make_shared<pressureMeasuring>(sys,pd,pg);
    
    ////////////////////////////////////////////////////////////////////
    
    auto capsidPot   = make_shared<Capsid>(epsilon,alphaCut,radius);
    auto capsidForce = make_shared<ExternalForces<Capsid>>(pd, pg, sys,capsidPot);

    ////////////////////////////////////////////////////////////////////
    
    auto pot = make_shared<potential>(sys);
    
    #ifdef GEL_POT
    
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
    
    #else
    
    {
        potential::InputPairParameters params;
        
        params.diamEff   = partDiam;
        params.epsilon = 1;
        
        params.cutOff   = real(1.122462)*partDiam;
        
        pot->setPotParameters(0, 0, params);
    }
    
    #endif
    
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
    
    pM->addInteractor(capsidForce);
    pM->addInteractor(partForces);
    
    outputState(sys,pd,outState,capsidPot->getRadiusZ(),capsidPot->getRadiusXY());
    
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
            outputState(sys,pd,outState,capsidPot->getRadiusZ(),capsidPot->getRadiusXY());
        }
        
        if(j%measuringSteps ==0) {
            sys->log<System::MESSAGE>("Measuting ...");
            //outMeasured << capsidPot->getRadiusZ() << " " << pM->sumVirial() << std::endl;
            outMeasured << capsidPot->getRadiusZ() << " " << pM->sumForce().z << std::endl;
        }
        
        if(j%decreaseSteps ==0) {
            sys->log<System::MESSAGE>("Decreasing radius z ...");
            capsidPot->setRadiusZ(capsidPot->getRadiusZ()-0.005);
            sys->log<System::MESSAGE>("New radii: %f (XY) %f (Z)",capsidPot->getRadiusXY(),capsidPot->getRadiusZ());
        }
        
        if(j%sortSteps   == 0){ pd->sortParticles(); }
    }
    
    auto totalTime = tim.toc();
    sys->log<System::MESSAGE>("mean FPS: %.2f", nSteps/totalTime);
    sys->finish();
  
}
