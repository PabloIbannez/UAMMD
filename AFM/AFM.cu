#include "uammd.cuh"

#include "Integrator/VerletNVT.cuh"
#include "Interactor/BondedForces.cuh"
#include "Interactor/PairForces.cuh"
#include "Interactor/ExternalForces.cuh"
#include "Interactor/NeighbourList/VerletList.cuh"
#include "utils/InputFile.h"

#include "AFM_potential.cuh"
#include "iogpf/iogpf.hpp"

#include "third_party/cub/cub.cuh"

#include "AFM_integrator.cuh"

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

struct wall: public ParameterUpdatable{
	
	real zwall;
	
	real wallRadius;
	real wallEpsilon;
	
	wall(real zwall,real wallRadius, real wallEpsilon):zwall(zwall),wallRadius(wallRadius),wallEpsilon(wallEpsilon){}
	
	__device__ __forceinline__ real3 force(const real4 &pos,const real &radius){
	
		const real z = abs(pos.z-zwall);
		const real effDiam = radius+wallRadius;
            
        const real invz2   = real(1.0)/(z*z);
        const real Dinvz2  = effDiam*effDiam*invz2;
        const real Dinvz6  = Dinvz2*Dinvz2*Dinvz2;
        const real Dinvz12 = Dinvz6*Dinvz6;
        
        return {real(0),
                real(0),
                real(4*6)*wallEpsilon*(real(2)*Dinvz12-Dinvz6)*((pos.z-zwall)/z)};
	
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
        
        if(mId.raw()[sortedIndex[i]] < 0){
            /*
            if(box.apply_pbc(make_real3(pos.raw()[sortedIndex[i]])).x < 0){
                os  << box.apply_pbc(make_real3(pos.raw()[sortedIndex[i]]))  <<  " "
		            << radius.raw()[sortedIndex[i]]                          <<  " "
		            << mId.raw()[sortedIndex[i]]                             <<  std::endl;
            }
            */
            os  << box.apply_pbc(make_real3(pos.raw()[sortedIndex[i]]))  <<  " "
                << radius.raw()[sortedIndex[i]]                          <<  " "
                << mId.raw()[sortedIndex[i]]                             <<  std::endl;
                
        } else {
            os  << box.apply_pbc(make_real3(pos.raw()[sortedIndex[i]]))  <<  " "
		        << radius.raw()[sortedIndex[i]]                          <<  " "
		        << mId.raw()[sortedIndex[i]]                             <<  std::endl;
        }
    }
	
}

void outputState_Wall(std::ofstream& os,
                           std::shared_ptr<System> sys,
                           std::shared_ptr<ParticleData> pd,
                           Box box,
                           real wallZ,
                           real wallRadius){
							
	outputState(os,sys,pd,box);
	
	int  nx = std::ceil(box.boxSize.x/(real(2.0)*wallRadius));
	int  ny = std::ceil(box.boxSize.x/(real(2.0)*wallRadius));
	
	fori(0,nx){
	forj(0,ny){
		
		real3 wallAtom = {real(2.0)*wallRadius*i-box.boxSize.x*real(0.5),real(2.0)*wallRadius*j-box.boxSize.y*real(0.5),wallZ};
		
		os  << wallAtom    <<  " "
		    << wallRadius  <<  " "
		    << 0           <<  std::endl;
        
    }}
	
}

void outputState_TipWall(std::ofstream& os,
                           std::shared_ptr<System> sys,
                           std::shared_ptr<ParticleData> pd,
                           Box box,
                           real3 tipPosition,
                           real tipRadius,
                           real wallZ,
                           real wallRadius){
							
	outputState_Wall(os,sys,pd,box,wallZ,wallRadius);
    
    os << tipPosition                << " " 
       << tipRadius                  << " "
       << 0                          << std::endl;
}

class virusSelector{
    
    public:    
    
        virusSelector(){}

        bool isSelected(int particleIndex, shared_ptr<ParticleData> &pd){
            int molId = (pd->getMolId(access::cpu, access::read).raw())[particleIndex];
            return molId < 0;
        }
};

using NVT     = VerletNVT::GronbechJensen;
using NVT_AFM = AFM::AFM_integrator::AFM_integrator;

int main(int argc, char *argv[]){
    
    auto sys = std::make_shared<System>();
    
    ////////////////////////////////////////////////////////////////////
    
    //Input
    
    if(argc != 6){
        sys->log<System::CRITICAL>("Input error, expected arguments are: *.top *.enm *.gaussian *.in outputName");
    }
    
    std::string inputTop      = argv[1];
    std::string inputENM      = argv[2];
    std::string inputGaussian = argv[3];
    std::string inputOptions  = argv[4];
    
    std::string outputName    = argv[5];
    
    //System
    
    real kB;
    
    real temperature;
    real viscosity; //gamma = 6*pi*viscosity*radius
    
    real kBT;
    
    Box box;
    
    //Particles
    real epsilon_wca;
    
    real epsilon_0;
    real lambda;
    real gamma;
    
    std::string partInteractionData;
    
    real f;
    real debyeLenght;
    real dielectricConstant;
    
    real cutOffFactorLJ;
    real cutOffFactorDebye;
    
    bool allSteric;
    
    //ENM
    real k;
    
    //Gaussian
	real epsilonGaussian;
    real sigmaGaussian;
    
    //Wall
	real wallRadius;
    real wallEpsilon;
    
    //Tip
    real tipMass;
    real tipRadius;
    
    real kTip;
    real harmonicWallTip;
    
    real initialVirusTipSep;
    
    real tipHorizontalDisplacement;
    
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
    real maxIndentationForce;
	
    //Integrator
	real dt;
    
    {//Input
        InputFile inputFile(inputOptions, sys);
        
        //System
        
        if(!(inputFile.getOption("kB")>>kB))
        {sys->log<System::CRITICAL>("kB option has not been introduced properly.");}
        
        if(!(inputFile.getOption("temperature")>>temperature))
        {sys->log<System::CRITICAL>("temperature option has not been introduced properly.");}
        if(!(inputFile.getOption("viscosity")>>viscosity))
        {sys->log<System::CRITICAL>("viscosity option has not been introduced properly.");}
        
        if(!(inputFile.getOption("box")>>box.boxSize.x
                                       >>box.boxSize.y
                                       >>box.boxSize.z))
        {sys->log<System::CRITICAL>("Box option has not been introduced properly.");}
        
                                  
        //Particles
        if(!(inputFile.getOption("epsilon_wca")>>epsilon_wca))
        {sys->log<System::CRITICAL>("epsilon_wca option has not been introduced properly.");}
        
        
        if(!(inputFile.getOption("epsilon_0")>>epsilon_0))
        {sys->log<System::CRITICAL>("epsilon_0 option has not been introduced properly.");}
        if(!(inputFile.getOption("lambda")>>lambda))
        {sys->log<System::CRITICAL>("lambda option has not been introduced properly.");}
        if(!(inputFile.getOption("gamma")>>gamma))
        {sys->log<System::CRITICAL>("gamma option has not been introduced properly.");}
        
        if(!(inputFile.getOption("partInteractionData")>>partInteractionData))
        {sys->log<System::CRITICAL>("partInteractionData option has not been introduced properly.");}
        
        if(!(inputFile.getOption("f")>>f))
        {sys->log<System::CRITICAL>("f option has not been introduced properly.");}
        if(!(inputFile.getOption("debyeLenght")>>debyeLenght))
        {sys->log<System::CRITICAL>("debyeLenght option has not been introduced properly.");}
        if(!(inputFile.getOption("dielectricConstant")>>dielectricConstant))
        {sys->log<System::CRITICAL>("dielectricConstant option has not been introduced properly.");}
        
        if(!(inputFile.getOption("cutOffFactorLJ")>>cutOffFactorLJ))
        {sys->log<System::CRITICAL>("cutOffFactorLJ option has not been introduced properly.");}
        if(!(inputFile.getOption("cutOffFactorDebye")>>cutOffFactorDebye))
        {sys->log<System::CRITICAL>("cutOffFactorDebye option has not been introduced properly.");}
        
        std::string allStericOption;
        
        if(!(inputFile.getOption("allSteric")>>allStericOption))
        {sys->log<System::CRITICAL>("allStericOption option has not been introduced properly.");}
        
        if(allStericOption == "true"){
            allSteric = true;
        } else if (allStericOption == "false"){
            allSteric = false;
        } else {
            {sys->log<System::CRITICAL>("allStericOption option has not been introduced properly, options are \"true\" or \"false\".");}
        }
        
        
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
        if(!(inputFile.getOption("wallEpsilon")>>wallEpsilon))
        {sys->log<System::CRITICAL>("wallEpsilon option has not been introduced properly.");}
                                               
        //Tip
        if(!(inputFile.getOption("tipMass")>>tipMass))
        {sys->log<System::CRITICAL>("tipMass option has not been introduced properly.");}
        if(!(inputFile.getOption("tipRadius")>>tipRadius))
        {sys->log<System::CRITICAL>("tipRadius option has not been introduced properly.");}
        
        if(!(inputFile.getOption("kTip")>>kTip))
        {sys->log<System::CRITICAL>("kTip option has not been introduced properly.");}
        if(!(inputFile.getOption("harmonicWallTip")>>harmonicWallTip))
        {sys->log<System::CRITICAL>("harmonicWallTip option has not been introduced properly.");}
                                                            
        if(!(inputFile.getOption("initialVirusTipSep")>>initialVirusTipSep))
        {sys->log<System::CRITICAL>("initialVirusTipSep option has not been introduced properly.");}
        
        if(!(inputFile.getOption("tipHorizontalDisplacement")>>tipHorizontalDisplacement))
        {sys->log<System::CRITICAL>("tipHorizontalDisplacement option has not been introduced properly.");}
        
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
        if(!(inputFile.getOption("maxIndentationForce")>>maxIndentationForce))
        {sys->log<System::CRITICAL>("maxIndentationForce option has not been introduced properly.");}
	    
        //Integrator

        if(!(inputFile.getOption("dt")>>dt))
        {sys->log<System::CRITICAL>("dt option has not been introduced properly.");}
        
        ////////////////////////////////////////////////////////////////
        
        kBT = kB*temperature;
        
    }
    
    /////////////////////////OUTPUT FILES///////////////////////////////
	
    std::stringstream ss;
    
    ss << outputName << "_state_k_" << k << "_e_" << epsilonGaussian << "_s_" << sigmaGaussian << ".sp";
    std::ofstream outState(ss.str());
    
    ss.str("");
    ss.clear();
    ss << outputName << "_tip_k_" << k << "_e_" << epsilonGaussian << "_s_" << sigmaGaussian << ".dat";
    std::ofstream outTip(ss.str());
    
    ss.str("");
    ss.clear();
    ss << outputName << "_k_" << k << "_e_" << epsilonGaussian << "_s_" << sigmaGaussian << ".log";
    std::ofstream outLog(ss.str());
	
	////////////////////////////////////////////////////////////////////
    
    aminoAcidMap aminoMap;
    
    aminoMap.loadAminoAcids("aminoAcidsList.dat");
    aminoMap.applyMap2File(inputTop,inputTop + std::string("P"));
    aminoMap.applyMap2File(partInteractionData,partInteractionData + std::string("P"));
    
    ullint seed = 0xf31337Bada55D00dULL;
    //ullint seed = time(NULL);
    sys->rng().setSeed(seed);
    
    ///////////////////////////INPUT DATA///////////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    auto pd = iogpf::read<iogpf::id,
                          iogpf::type,
                          iogpf::molId,
                          iogpf::mass,
                          iogpf::radius,
                          iogpf::charge,
                          iogpf::SASAratio,
                          iogpf::pos>(sys,inputTop + std::string("P"));
                          
    int N = pd->getNumParticles();
    
    ////////////////////////PARTICLE GROUPS/////////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    auto pgAll = std::make_shared<ParticleGroup>(pd, sys, "All");
    
    virusSelector vSel;
    auto pgVirus = std::make_shared<ParticleGroup>(vSel,pd, sys, "Virus");
    
    /////////////////////INTERNAL INTERACTORS///////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    using ENM = BondedForces<BondedType::ENM_PBC>;

    ENM::Parameters paramsENM;
    
    paramsENM.file = inputENM; 
    BondedType::ENM_PBC enm(box,k);
    auto bondedforces = std::make_shared<ENM>(pd, sys, paramsENM, enm);
    
    ////////////////////////////////////////////////////////////////////
    
    using GaussianForces = BondedForces<BondedType::GaussianPBC>;

    GaussianForces::Parameters paramsGF;
    
    paramsGF.file = inputGaussian;  
    BondedType::GaussianPBC gf(box,epsilonGaussian,sigmaGaussian);
    auto gaussianforces = std::make_shared<GaussianForces>(pd, sys, paramsGF, gf);
    
    ////////////////////////////////////////////////////////////////////
    
    real cutOff;
    
    {
        real maxRadius = 0;
        
        auto radius   = pd->getRadius(access::location::cpu, access::mode::read);
        
        fori(0,N){
            if(radius.raw()[i] > maxRadius){
                maxRadius = radius.raw()[i];
            }
        }
        
        if(!allSteric){
            cutOff = std::max(gamma*real(2)*maxRadius*cutOffFactorLJ,debyeLenght*cutOffFactorDebye);
        } else {
            cutOff = real(1.122462)*real(2)*maxRadius*real(1.05);
        }
    
    }
    
    using SASAmodel = Potential::SASAmodel::C;
    
    Potential::AFMPotential<SASAmodel>::Parameters paramsPF;
    
    paramsPF.kBT = kBT;

    paramsPF.epsilon_wca = epsilon_wca;

    paramsPF.partInteractionData = partInteractionData + std::string("P");

    paramsPF.epsilon_0 = epsilon_0;
    paramsPF.lambda = lambda;
    paramsPF.gamma = gamma;

    paramsPF.f = f;
    paramsPF.debyeLenght = debyeLenght;
    paramsPF.dielectricConstant = dielectricConstant;

    paramsPF.cutOff = cutOff;
    
    auto potPairForces = std::make_shared<Potential::AFMPotential<SASAmodel>>(sys,paramsPF);
    
    PairForces<Potential::AFMPotential<SASAmodel>, CellList>::Parameters params;
    params.box = box;  //Box to work on
    auto pairforces = std::make_shared<PairForces<Potential::AFMPotential<SASAmodel>, CellList>>(pd, pgAll, sys, params, potPairForces);
    
    /////////////////////EXTERNAL INTERACTORS///////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    
    real wallZ;
    {
        
        real minZ =  INFINITY;
        real radiusClosestMin = 0;
    
		auto pos      = pd->getPos(access::location::cpu, access::mode::read);
		auto radius   = pd->getRadius(access::location::cpu, access::mode::read);
		
		fori(0,N){
			real z = pos.raw()[i].z;
			if(z < minZ){
				minZ = z;
				radiusClosestMin = radius.raw()[i];
			}
		}
        
        wallZ = minZ-real(1.122462)*(wallRadius+radiusClosestMin);
	}
    
    //WALL
	
    auto extWall = std::make_shared<ExternalForces<wall>>(pd, pgAll, sys,std::make_shared<wall>(wallZ,wallRadius,wallEpsilon));
    
    //DOWNWARD FORCE 
	auto downwardForceVirus = std::make_shared<ExternalForces<ConstantForce>>(pd, pgVirus, sys,std::make_shared<ConstantForce>(downwardForce));
    
    ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    outLog << "kBT: "       << kBT       << std::endl;
    outLog << "cutOff: "    << cutOff    << std::endl;
    outLog << "allSteric: " << allSteric << std::endl;
    
    ////////////////////////SIMULATION STARTS///////////////////////////
    
    Timer tim;
    tim.tic();
    
    ////////////////////////DOWNWARD////////////////////////////////////
    
    {
		NVT::Parameters parDown;
		parDown.temperature = kBT; 
		parDown.dt = dt;
		parDown.viscosity = viscosity;  
		
		auto verletDown = std::make_shared<NVT>(pd, pgAll, sys,parDown);
		
		verletDown->addInteractor(bondedforces);
		verletDown->addInteractor(gaussianforces);
		verletDown->addInteractor(pairforces);
		verletDown->addInteractor(extWall);
		verletDown->addInteractor(downwardForceVirus);
        
        potPairForces->setAllSteric(true);
		
		outputState_Wall(outState,sys,pd,box,
                         wallZ,wallRadius);
		
		////////////////////////////////////////////////////////////////
        
        sys->log<System::MESSAGE>("Downward process starts");
		
		forj(0,nstepsDownward){
			verletDown->forwardTime();
		
			//Write results
			if(j%printStepsDownward==1){
				outputState_Wall(outState,sys,pd,box,
                                 wallZ,wallRadius);
			}    
			
			if(j%sortSteps == 0){
				pd->sortParticles();
			}
		}
        
        sys->log<System::MESSAGE>("Downward process ends");
	}
	
    ////////////////////////////RUN/////////////////////////////////////
    
    //Set tip position
    
    real3 tipPosition = {0,0,0};
    
    {
        real maxZ = -INFINITY;
        real radiusClosestMax = 0;
        
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
        
        
        sys->rng().next32();
        sys->rng().next32();
        real rndAngle = sys->rng().uniform(0,real(2)*M_PI);
		
        tipPosition = {virusCentroid.x+tipHorizontalDisplacement*std::cos(rndAngle),
                       virusCentroid.y+tipHorizontalDisplacement*std::sin(rndAngle),
                       maxZ + (tipRadius+radiusClosestMax)+initialVirusTipSep};
	}
	
	
    {
		NVT_AFM::Parameters par;
		par.temperature = kBT; 
		par.dt = dt;
		par.viscosity = viscosity;  
        
        par.kTip = kTip;
        par.harmonicWallTip = harmonicWallTip;

        par.tipMass = tipMass;
        par.tipRadius = tipRadius;
        
		
		auto verlet = std::make_shared<NVT_AFM>(pd, pgAll, sys, par);
		
		verlet->addInteractor(bondedforces);
		verlet->addInteractor(gaussianforces);
		verlet->addInteractor(pairforces);
		verlet->addInteractor(extWall);
        
        verlet->setTipPosition(tipPosition);
		
		outputState_TipWall(outState,sys,pd,box,
                              verlet->getTipPosition(),
                              verlet->getTipRadius(),
                              wallZ,wallRadius);
		
		////////////////////////////////////////////////////////////////
        
        potPairForces->setAllSteric(allSteric);
        
        sys->log<System::MESSAGE>("Thermalization stars");
        
        //Thermalization
        forj(0,nstepsTerm){
			verlet->forwardTime();
		
			//Write results
			if(j%printStepsTerm==1){
				outputState_TipWall(outState,sys,pd,box,
                                    verlet->getTipPosition(),
                                    verlet->getTipRadius(),
                                    wallZ,wallRadius);
			}    
			
			if(j%sortSteps == 0){
				pd->sortParticles();
			}
		}
        
        sys->log<System::MESSAGE>("Thermalization ends");
        
        sys->log<System::MESSAGE>("Simulation starts");
        
		
		//Run the simulation
		forj(0,nsteps){
			verlet->forwardTime();
            
			if(j%printSteps==1){
				outputState_TipWall(outState,sys,pd,box,
                                      verlet->getTipPosition(),
                                      verlet->getTipRadius(),
                                      wallZ,wallRadius);
			}
			
			if(j%measureSteps == 0){
                real currentForce = -verlet->getTipForce().z*real(0.0016605391);
                // 1 kj/(mol·nm) = 0.0016605391 nN
                outTip << verlet->getTipPositionEq() << " " << verlet->getTipPosition().z << " " << currentForce << std::endl;
                
                if(currentForce > maxIndentationForce){break;}
            }
            
            if(j%descentSteps == 0){
                real currentTipPositionEq = verlet->getTipPositionEq();
                verlet->setTipPositionEq(currentTipPositionEq-descentDistace);
                
                if(verlet->getTipPosition().z < maxIndentation){break;}
                
                //Thermalization
                forj(0,nstepsTerm){
		        	verlet->forwardTime();
		        
		        	//Write results
		        	if(j%printStepsTerm==1){
		        		outputState_TipWall(outState,sys,pd,box,
                                            verlet->getTipPosition(),
                                            verlet->getTipRadius(),
                                            wallZ,wallRadius);
		        	}    
		        	
		        	if(j%sortSteps == 0){
		        		pd->sortParticles();
		        	}
		        }
            }
            
			
			if(j%sortSteps == 0){
				pd->sortParticles();
			}
		}
        
        sys->log<System::MESSAGE>("Simulation ends");
    
	}
	
    auto totalTime = tim.toc();
    sys->log<System::MESSAGE>("mean FPS: %.2f", nsteps/totalTime);
    sys->finish();
    
    return EXIT_SUCCESS;
}
