#include "uammd.cuh"

#include "Integrator/SteepestDescent.cuh"
#include "Integrator/VerletNVT.cuh"
#include "Interactor/BondedForces.cuh"
#include "Interactor/PairForces.cuh"
#include "Interactor/ExternalForces.cuh"
#include "Interactor/NeighbourList/VerletList.cuh"
#include "utils/InputFile.h"

#include "AFM_potential.cuh"
#include "iogpf/iogpf.hpp"

#include "third_party/cub/cub.cuh"

using namespace uammd;

#include <map>
#include <fstream>

class aminoAcidMap{
    
    private:
        
        std::map<std::string,int> aminoAcid2Int;
    
    public:
        
        void loadAminoAcids(std::string inputFilePath){
            
            std::stringstream ss;
    
            std::ifstream inputFile(inputFilePath);
            if (inputFile.fail()) {
                ss.clear();
                ss << "Error loading file \"" << inputFilePath << "\".";
                throw std::runtime_error(ss.str());
            } else {
                std::cerr << "File \"" << inputFilePath << "\" opened." << std::endl;
            }
            
            std::string line;
            
            std::string stringBuffer;
            
            int aminoAcidCount = 0;
            
            while(std::getline(inputFile,line)) {
                
                ss.str(line);
                
                ss >> stringBuffer;
                
                aminoAcid2Int[stringBuffer] = aminoAcidCount;
                aminoAcidCount++;
            }
            
            #ifdef DEBUG
            for(auto pair : aminoAcid2Int){
                std::cout << pair.first << " " << pair.second << std::endl;
            }
            #endif
            
        }
        
        int aminoAcidKey(std::string aminoAcid){
            
            std::stringstream ss;
            
            if(aminoAcid2Int.count(aminoAcid) == 0){
                ss.clear();
                ss << "Amino acid \"" << aminoAcid << "\" has not been introduced";
                throw std::runtime_error(ss.str());
            } else {
                return aminoAcid2Int[aminoAcid];
            }
        }
        
        void applyMap2File(std::string inputFilePath,std::string outputFilePath){
            
            //input file
            std::stringstream ss;
    
            std::ifstream inputFile(inputFilePath);
            if (inputFile.fail()) {
                ss.clear();
                ss << "Error loading file \"" << inputFilePath << "\".";
                throw std::runtime_error(ss.str());
            } else {
                std::cerr << "File \"" << inputFilePath << "\" opened." << std::endl;
            }
            //output file
            
            std::ofstream outputFile(outputFilePath);
            
            ////////////////////////////////////////////////////////////
            
            std::string line;
            
            while(std::getline(inputFile,line)) {
                for(auto pair : aminoAcid2Int){
                    size_t pos = line.find(pair.first);
                    while( pos != std::string::npos)
                    {
                        line.replace(pos, pair.first.size(), std::to_string(pair.second));
                        pos =line.find(pair.first, pos + std::to_string(pair.second).size());
                    }
                }
                
                outputFile << line << " " << std::endl;
            }
        }
        
    
};


namespace uammd{

    namespace BondedType{
      
    struct ENM{
        
        real k;
    
        struct BondInfo{
            real r0;
        };
        
        ENM(real k):k(k){}
    
        inline __device__ real3 force(int i, int j, const real3 &r12, const BondInfo &bi){
            real r2 = dot(r12, r12);
            if(r2==real(0.0)) return make_real3(0.0);
            
            real invr = rsqrt(r2);
            real f = -k*(real(1.0)-bi.r0*invr); //F = -k·(r-r0)·rvec/r
            return f*r12;
        }
    
        static __host__ BondInfo readBond(std::istream &in){
            BondInfo bi;
            in>>bi.r0;
            return bi;
        }
    
        inline __device__ real energy(int i, int j, const real3 &r12, const BondInfo &bi){
            real r2 = dot(r12, r12);
            if(r2==real(0.0)) return real(0.0);
    
            real invr = rsqrt(r2);
            const real dr = real(1.0)-bi.r0*invr;
            
            return real(0.5)*k*dr*dr;
        }
    
    };

    struct ENM_PBC: public ENM{
        Box box;
        
        ENM_PBC(Box box,real k): box(box),ENM(k){}
        
        inline __device__ real3 force(int i, int j, const real3 &r12, const BondInfo &bi){
            return ENM::force(i, j, box.apply_pbc(r12), bi);
        }
    
        inline __device__ real energy(int i, int j, const real3 &r12, const BondInfo &bi){
          return ENM::energy(i, j, box.apply_pbc(r12), bi);
        }
    };
      
    struct Gaussian{
        
        real wellDepth;
        real invSigma;
        
        Gaussian(real wellDepth,real sigma):wellDepth(wellDepth),invSigma(1.0/sigma){}  
      
        struct BondInfo{
			real r0;
        };
    
        inline __device__ real3 force(int i, int j, const real3 &r12, const BondInfo &bi){
          real r2 = dot(r12, r12);
          if(r2==real(0.0)) return make_real3(0.0);
          
          real r = sqrt(r2);
          real dst = r-bi.r0;
          real fmod = -real(2.0)*wellDepth*exp(-dst*dst*invSigma)*dst*invSigma;
          return fmod*(r12/r);
        }
        
        static __host__ BondInfo readBond(std::istream &in){
          
          BondInfo bi;
          in>>bi.r0;
          
          return bi;
        }
    
        inline __device__ real energy(int i, int j, const real3 &r12, const BondInfo &bi){
          real r2 = dot(r12, r12);
          if(r2==real(0.0)) return -(wellDepth/bi.r0);
    
          real r = sqrt(r2);
          real dst = r-bi.r0;
          real energy = -wellDepth*exp(-dst*dst*invSigma);
          
          return energy;
        }
    
    };
    
    struct GaussianPBC: public Gaussian{
        
        Box box;
        
        GaussianPBC(Box box,real E,real sigma): box(box),Gaussian(E,sigma){}
        
        inline __device__ real3 force(int i, int j, const real3 &r12, const BondInfo &bi){
          return Gaussian::force(i, j, box.apply_pbc(r12), bi);
        }
        
        inline __device__ real energy(int i, int j, const real3 &r12, const BondInfo &bi){
          return Gaussian::energy(i, j, box.apply_pbc(r12), bi);
        }
        };
}}

struct WCA_Wall: public ParameterUpdatable{
	
	real zwall;
	
	real wallRadius;
	real epsilon = 1;
	
	WCA_Wall(real zwall,real wallRadius):zwall(zwall),wallRadius(wallRadius){}
	
	__device__ __forceinline__ real3 force(const real4 &pos,const real &radius){
	
		const real z = abs(pos.z-zwall);
		const real effDiam = radius+wallRadius;
            
        //if(z > real(1.122462)*effDiam){
        if(z > real(2.5)*effDiam){
			return make_real3(0);
		} else {
			const real invz2   = real(1.0)/(z*z);
			const real Dinvz2  = effDiam*effDiam*invz2;
			const real Dinvz6  = Dinvz2*Dinvz2*Dinvz2;
			const real Dinvz12 = Dinvz6*Dinvz6;
			
			return {real(0),
					real(0),
					real(4*6)*epsilon*(real(2)*Dinvz12-Dinvz6)*((pos.z-zwall)/z)};
		}
	
	}
	
	std::tuple<const real4 *,const real *> getArrays(ParticleData *pd){
		auto pos    = pd->getPos(access::location::gpu, access::mode::read);
		auto radius = pd->getRadius(access::location::gpu, access::mode::read);
		return std::make_tuple(pos.raw(),radius.raw());
	}
};

struct ConstantForce: public ParameterUpdatable{
	
	real fmod;
	
	ConstantForce(real fmod):fmod(fmod){}
	
	__device__ __forceinline__ real3 force(const real4 &pos){
	
		return make_real3(0.0f, 0.0f, fmod);
	
	}
	
	std::tuple<const real4 *> getArrays(ParticleData *pd){
		auto pos = pd->getPos(access::location::gpu, access::mode::read);
		return std::make_tuple(pos.raw());
	}
}; 

struct TipForce: public ParameterUpdatable{
    
	
    real3 tipPosition = {0,0,0};
    
    real  epsilon = 1;
    real  tipRadius = 20;
    
	TipForce(real epsilon,real tipRadius):epsilon(epsilon),tipRadius(tipRadius){}
	
	__device__ __forceinline__ real3 force(const real4 &pos,const real &radius){
	    
        real3 rtp = tipPosition-make_real3(pos);
        real  r2 = dot(rtp,rtp);
        real  r  = sqrt(r2);
        
        real effDiam = tipRadius+radius;
        
        if(r < real(1.122462)*effDiam){
            
            const real invr2   = real(1.0)/r2;
            const real Dinvr2  = effDiam*effDiam*invr2;
            const real Dinvr6  = Dinvr2*Dinvr2*Dinvr2;
            const real Dinvr12 = Dinvr6*Dinvr6;
            
            real fmod = real(4*6)*epsilon*(real(2)*Dinvr12-Dinvr6);
            
            return -fmod*rtp;
            
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
    
    real3 getTipPosition(){
        return tipPosition;
    }
    
    void setTipRadius(real newTipRadius){
        tipRadius = newTipRadius;
    }
    
    real getTipRadius(){
        return tipRadius;
    }
    
};

void outputState(std::ofstream& os,
			     std::shared_ptr<System> sys,
			     std::shared_ptr<ParticleData> pd,
			     Box box){
	
	sys->log<System::DEBUG>("[System] Writing to disk...");
        
    auto pos      = pd->getPos(access::location::cpu, access::mode::read);
	auto mId      = pd->getMolId(access::location::cpu, access::mode::read);
	auto radius   = pd->getRadius(access::location::cpu, access::mode::read);
	
    const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
    
    os<<"#Lx="<<0.5*box.boxSize.x<<";Ly="<<0.5*box.boxSize.y<<";Lz="<<0.5*box.boxSize.z<<";"<<std::endl;
    
    fori(0,pd->getNumParticles()){
        os  << box.apply_pbc(make_real3(pos.raw()[sortedIndex[i]]))  <<  " "
		    << real(1.122462)*radius.raw()[sortedIndex[i]]           <<  " "
		    << mId.raw()[sortedIndex[i]]                             <<  std::endl;
    }
	
}

void outputState_TipWall(std::ofstream& os,
                           std::shared_ptr<System> sys,
                           std::shared_ptr<ParticleData> pd,
                           Box box,
                           real3 tipPosition,
                           real tipRadius,
                           real wallZ,
                           real wallRadius){
							
	outputState(os,sys,pd,box);
    
    os << tipPosition                << " " 
       << real(1.122462)*tipRadius   << " "
       << -1 << std::endl;
	
	int  nx = std::ceil(box.boxSize.x/(real(2.0)*wallRadius));
	int  ny = std::ceil(box.boxSize.x/(real(2.0)*wallRadius));
	
	fori(0,nx){
	forj(0,ny){
		
		real3 wallAtom = {real(2.0)*wallRadius*i-box.boxSize.x*real(0.5),real(2.0)*wallRadius*j-box.boxSize.y*real(0.5),wallZ};
		
		os  << wallAtom    <<  " "
		    << wallRadius  <<  " "
		    << -2          <<  std::endl;
        
    }}
	
}

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

template <class tipForce>
class tipMeasuring{
    
    private:
        
        shared_ptr<System> sys;
        
        shared_ptr<ParticleData> pd;
        shared_ptr<ParticleGroup> pg;
        
        shared_ptr<tipForce> tipF;
        
        cudaStream_t stream;
        
        //cub reduction variables
        real3*   totalForce;
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
            
            cub::DeviceReduce::Sum(cubTempStorageSum, cubTempStorageSizeSum, forceSumIterator, totalForce, pg->getNumberParticles(), stream);
            cudaMalloc(&cubTempStorageSum, cubTempStorageSizeSum);
        }
    
    public:
    
        tipMeasuring(shared_ptr<System> sys,
                       shared_ptr<ParticleData> pd,
                       shared_ptr<ParticleGroup> pg,
                       shared_ptr<tipForce> tipF):sys(sys),pd(pd),pg(pg),tipF(tipF){
          
          sys->log<System::MESSAGE>("[tipMeasuring] Created.");
          
          cudaStreamCreate(&stream);
          
          cudaMallocManaged((void**)&totalForce,sizeof(real3));
          this->setUpCubReduction();
        }
        
        ~tipMeasuring(){
          
            sys->log<System::MESSAGE>("[tipMeasuring] Destroyed.");
            
            cudaFree(totalForce);
            cudaFree(cubTempStorageSum);
            cudaStreamDestroy(stream);
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
            
            tipF->sumForce(stream);
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
};

using NVT = VerletNVT::GronbechJensen;

int main(int argc, char *argv[]){
    
    auto sys = std::make_shared<System>();
    
    ////////////////////////////////////////////////////////////////////
    
    //System
    
    Box box;
    
    //Particles
    real epsilonParticles;
    real cutOffParticles;
    
    //ENM
    real k;
    
    //Gaussian
	real epsilonGaussian;
    real sigmaGaussian;
    
    //Wall
	real wallRadius;
    
    //Tip
    real tipRadius;
    real tipEpsilon;
    
    real tipInitHeight;
    real initialVirusTipSep;
    
    //Downward force
    real downwardForce;
    
    //Downward
    int nstepsDownward;
    int printStepsDownward;

    //Thermalization
    int nstepsTerm;
	int printStepsTerm;
    
    //Simulation
    int nsteps;
    int printSteps;
    
    //Common
    int sortSteps;
    
    //Measuring
    int  descentSteps;
    real descentDistace;
    int  measureSteps;
    real maxIndentation;
	
    //Integrator
	real temperature;
	real dt;
	real viscosity;
    
    {//Input
        InputFile inputFile("options.in", sys);
        
        if(!(inputFile.getOption("box")>>box.boxSize.x
                                      >>box.boxSize.y
                                      >>box.boxSize.z))
        {sys->log<System::CRITICAL>("Box option has not been introduced properly.");}
                                  
        //Particles
        if(!(inputFile.getOption("epsilonParticles")>>epsilonParticles))
        {sys->log<System::CRITICAL>("epsilonParticles option has not been introduced properly.");}
        if(!(inputFile.getOption("cutOffParticles")>>cutOffParticles))
        {sys->log<System::CRITICAL>("cutOffParticles option has not been introduced properly.");}
        
        //ENM
        if(!(inputFile.getOption("k")>>k))
        {sys->log<System::CRITICAL>("k option has not been introduced properly.");}
        
        //Gaussian                                          
        if(!(inputFile.getOption("epsilonGaussian")>>epsilonGaussian))
        {sys->log<System::CRITICAL>("epsilonGaussian option has not been introduced properly.");}
        if(!(inputFile.getOption("sigmaGaussian")>>sigmaGaussian))
        {sys->log<System::CRITICAL>("sigmaGaussian option has not been introduced properly.");}
                                                            
        //Wall                                              
        if(!(inputFile.getOption("wallRadius")>>wallRadius))
        {sys->log<System::CRITICAL>("wallRadius option has not been introduced properly.");}
                                               
        //Tip                                
        if(!(inputFile.getOption("tipRadius")>>tipRadius))
        {sys->log<System::CRITICAL>("tipRadius option has not been introduced properly.");}
        if(!(inputFile.getOption("tipEpsilon")>>tipEpsilon))
        {sys->log<System::CRITICAL>("tipEpsilon option has not been introduced properly.");}
                                                            
        if(!(inputFile.getOption("tipInitHeight")>>tipInitHeight))
        {sys->log<System::CRITICAL>("tipInitHeight option has not been introduced properly.");}
        if(!(inputFile.getOption("initialVirusTipSep")>>initialVirusTipSep))
        {sys->log<System::CRITICAL>("initialVirusTipSep option has not been introduced properly.");}
        
        //Downward force
        if(!(inputFile.getOption("downwardForce")>>downwardForce))
        {sys->log<System::CRITICAL>("downwardForce option has not been introduced properly.");}

        //Downward
        if(!(inputFile.getOption("nstepsDownward")>>nstepsDownward))
        {sys->log<System::CRITICAL>("nstepsDownward option has not been introduced properly.");}
        if(!(inputFile.getOption("printStepsDownward")>>printStepsDownward))
        {sys->log<System::CRITICAL>("printStepsDownward option has not been introduced properly.");}
        
        
        //Thermalization
        if(!(inputFile.getOption("nstepsTerm")>>nstepsTerm))
        {sys->log<System::CRITICAL>("nstepsTerm option has not been introduced properly.");}
        if(!(inputFile.getOption("printStepsTerm")>>printStepsTerm))
        {sys->log<System::CRITICAL>("printStepsTerm option has not been introduced properly.");}
        
        //Simulation
        if(!(inputFile.getOption("nsteps")>>nsteps))
        {sys->log<System::CRITICAL>("nsteps option has not been introduced properly.");}
        if(!(inputFile.getOption("printSteps")>>printSteps))
        {sys->log<System::CRITICAL>("printSteps option has not been introduced properly.");}
        
        //Common
        if(!(inputFile.getOption("sortSteps")>>sortSteps))
        {sys->log<System::CRITICAL>("sortSteps option has not been introduced properly.");}
        
        //Measuring
        if(!(inputFile.getOption("measureSteps")>>measureSteps))
        {sys->log<System::CRITICAL>("measureSteps option has not been introduced properly.");}
        if(!(inputFile.getOption("descentSteps")>>descentSteps))
        {sys->log<System::CRITICAL>("descentSteps option has not been introduced properly.");}
        if(!(inputFile.getOption("descentDistace")>>descentDistace))
        {sys->log<System::CRITICAL>("descentDistace option has not been introduced properly.");}
        if(!(inputFile.getOption("maxIndentation")>>maxIndentation))
        {sys->log<System::CRITICAL>("maxIndentation option has not been introduced properly.");}
	    
        //Integrator
        if(!(inputFile.getOption("temperature")>>temperature))
        {sys->log<System::CRITICAL>("temperature option has not been introduced properly.");}
        if(!(inputFile.getOption("dt")>>dt))
        {sys->log<System::CRITICAL>("dt option has not been introduced properly.");}
        if(!(inputFile.getOption("viscosity")>>viscosity))
        {sys->log<System::CRITICAL>("viscosity option has not been introduced properly.");}
        
    }
    
    /////////////////////////OUTPUT FILES///////////////////////////////
	
    std::stringstream ss;
    
    ss << "p22_k_" << k << "_e_" << epsilonGaussian << "_s_" << sigmaGaussian << ".sp";
    std::ofstream outState(ss.str());
    
    ss.str("");
    ss.clear();
    ss << "tip_k_" << k << "_e_" << epsilonGaussian << "_s_" << sigmaGaussian << ".dat";
    std::ofstream outTip(ss.str());
	
	////////////////////////////////////////////////////////////////////
    
    aminoAcidMap aminoMap;
    
    aminoMap.loadAminoAcids("aminoAcidsList.dat");
    aminoMap.applyMap2File("p22.top","p22_P.top");
    
    ullint seed = 0xf31337Bada55D00dULL;
    sys->rng().setSeed(seed);
    
    ///////////////////////////INPUT DATA///////////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    auto pd = iogpf::read<iogpf::id,
                          iogpf::type,
                          iogpf::molId,
                          iogpf::mass,
                          iogpf::radius,
                          iogpf::pos>(sys,"p22_P.top");
                          
    int N = pd->getNumParticles();
    
    ////////////////////////PARTICLE GROUPS/////////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    auto pgAll = std::make_shared<ParticleGroup>(pd, sys, "All");
    
    /////////////////////INTERNAL INTERACTORS///////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    using ENM = BondedForces<BondedType::ENM_PBC>;

    ENM::Parameters paramsENM;
    
    paramsENM.file = "p22.enm"; 
    BondedType::ENM_PBC enm(box,k);
    auto bondedforces = std::make_shared<ENM>(pd, sys, paramsENM, enm);
    
    ////////////////////////////////////////////////////////////////////
    
    using GaussianForces = BondedForces<BondedType::GaussianPBC>;

    GaussianForces::Parameters paramsGF;
    
    paramsGF.file = "p22.gaussian";  
    BondedType::GaussianPBC gf(box,epsilonGaussian,sigmaGaussian);
    auto gaussianforces = std::make_shared<GaussianForces>(pd, sys, paramsGF, gf);
    
    ////////////////////////////////////////////////////////////////////
    
    using PairForces = PairForces<Potential::AFMPotential, CellList>;
    
    auto pot = std::make_shared<Potential::AFMPotential>(sys,cutOffParticles,epsilonParticles);
    
    PairForces::Parameters params;
    params.box = box;  //Box to work on
    auto pairforces = std::make_shared<PairForces>(pd, pgAll, sys, params, pot);
    
    /////////////////////EXTERNAL INTERACTORS///////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    
    real wallZ;
    real tipZ;
    
    real minZ =  INFINITY;
    real maxZ = -INFINITY;
    real radiusClosestMin = 0;
    real radiusClosestMax = 0;
    
    {
		auto pos      = pd->getPos(access::location::cpu, access::mode::read);
		auto radius   = pd->getRadius(access::location::cpu, access::mode::read);
		
		fori(0,N){
			real z = pos.raw()[i].z;
			if(z < minZ){
				minZ = z;
				radiusClosestMin = radius.raw()[i];
			}

			if(z > maxZ){
				maxZ = z;
				radiusClosestMax = radius.raw()[i];
			}
		}
	}
    
    //WALL
	
	wallZ = minZ-real(1.122462)*(wallRadius+radiusClosestMin);
    
    auto extWall = std::make_shared<ExternalForces<WCA_Wall>>(pd, pgAll, sys,std::make_shared<WCA_Wall>(wallZ,wallRadius));
    
    //DOWNWARD FORCE 
	auto downwardForceVirus = std::make_shared<ExternalForces<ConstantForce>>(pd, pgAll, sys,std::make_shared<ConstantForce>(downwardForce));
	
	////PROBE FORCE
    
    tipZ = maxZ + real(1.122462)*(tipRadius+radiusClosestMax) + tipInitHeight;
    
	std::shared_ptr<TipForce> tipPot = std::make_shared<TipForce>(tipEpsilon,tipRadius);
    tipPot->setTipPosition({0,0,tipZ});
    
    auto forcesTip = std::make_shared<ExternalForces<TipForce>>(pd, pgAll, sys,tipPot);
    
    //////////////////////////MEASURING/////////////////////////////////
    
    auto measuring = std::make_shared<tipMeasuring<ExternalForces<TipForce>>>(sys,pd,pgAll,forcesTip);
    
    ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    ////////////////////////SIMULATION STARTS///////////////////////////
    
    Timer tim;
    tim.tic();
    
    ////////////////////////THERMALIZATION//////////////////////////////
    
    {
		NVT::Parameters parTherm;
		parTherm.temperature = temperature; 
		parTherm.dt = dt;
		parTherm.viscosity = viscosity;  
		
		auto verletTherm = std::make_shared<NVT>(pd, pgAll, sys,parTherm);
		
		verletTherm->addInteractor(bondedforces);
		verletTherm->addInteractor(gaussianforces);
		verletTherm->addInteractor(pairforces);
		verletTherm->addInteractor(extWall);
		verletTherm->addInteractor(downwardForceVirus);
		
		outputState_TipWall(outState,sys,pd,box,
                              tipPot->getTipPosition(),
                              tipPot->getTipRadius(),
                              wallZ,wallRadius);
		
		////////////////////////////////////////////////////////////////
		
		forj(0,nstepsDownward){
			verletTherm->forwardTime();
		
			//Write results
			if(j%printStepsDownward==1){
				outputState_TipWall(outState,sys,pd,box,
                                      tipPot->getTipPosition(),
                                      tipPot->getTipRadius(),
                                      wallZ,wallRadius);
			}    
			
			if(j%sortSteps == 0){
				pd->sortParticles();
			}
		}
	}
	
    ////////////////////////////RUN/////////////////////////////////////
    
    //Set tip position
    
    real3 tipPosition = {0,0,0};
    
    {
        maxZ = -INFINITY;
        
		auto pos    = pd->getPos(access::location::cpu, access::mode::readwrite);
		auto radius = pd->getRadius(access::location::cpu, access::mode::readwrite);
        
        real3 virusCentroid = {0,0,0};
		
		fori(0,N){
			virusCentroid += make_real3(pos.raw()[i]);
            real z = make_real3(pos.raw()[i]).z;
            
            if(z > maxZ){
				maxZ = z;
				radiusClosestMax = radius.raw()[i];
			}
		}
		
		virusCentroid /= N;
		
		//Move tip to {vC.x,vC.y,maxZ+(tipRadius+radiusClosest)+initialVirusTipSep}
		
        tipPosition = {virusCentroid.x,virusCentroid.y,maxZ + (tipRadius+radiusClosestMax)+initialVirusTipSep};
        tipPot->setTipPosition(tipPosition);
		
	}
	
	
    {
		NVT::Parameters par;
		par.temperature = temperature; 
		par.dt = dt;
		par.viscosity = viscosity;  
		
		auto verlet = std::make_shared<NVT>(pd, pgAll, sys, par);
		
		verlet->addInteractor(bondedforces);
		verlet->addInteractor(gaussianforces);
		verlet->addInteractor(pairforces);
		verlet->addInteractor(extWall);
		verlet->addInteractor(forcesTip);
		
		outputState_TipWall(outState,sys,pd,box,
                              tipPot->getTipPosition(),
                              tipPot->getTipRadius(),
                              wallZ,wallRadius);
		
		////////////////////////////////////////////////////////////////
        
        //Thermalization
        forj(0,nstepsTerm){
			verlet->forwardTime();
		
			//Write results
			if(j%printStepsTerm==1){
				outputState_TipWall(outState,sys,pd,box,
                                    tipPot->getTipPosition(),
                                    tipPot->getTipRadius(),
                                    wallZ,wallRadius);
			}    
			
			if(j%sortSteps == 0){
				pd->sortParticles();
			}
		}
		
		//Run the simulation
		forj(0,nsteps){
			verlet->forwardTime();
            
			if(j%printSteps==1){
				outputState_TipWall(outState,sys,pd,box,
                                      tipPot->getTipPosition(),
                                      tipPot->getTipRadius(),
                                      wallZ,wallRadius);
			}
			
			if(j%measureSteps == 0){
                // 1 kj/(mol·nm) = 0.0016605391 nN
                outTip << tipPot->getTipPosition().z << " " << -measuring->sumForce().z*real(0.0016605391) << std::endl;
            }
            
            if(j%descentSteps == 0){
                real3 currentTipPosition = tipPot->getTipPosition();
                tipPot->setTipHeight(currentTipPosition.z-descentDistace);
                
                if(currentTipPosition.z < maxIndentation){break;}
            }
            
			
			if(j%sortSteps == 0){
				pd->sortParticles();
			}
		}
    
	}
	
    auto totalTime = tim.toc();
    sys->log<System::MESSAGE>("mean FPS: %.2f", nsteps/totalTime);
    sys->finish();
    
    return EXIT_SUCCESS;
}
