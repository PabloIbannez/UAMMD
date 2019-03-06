#include "uammd.cuh"

#include "Integrator/SteepestDescent.cuh"
#include "Integrator/VerletNVT.cuh"
#include "Interactor/BondedForces.cuh"
#include "Interactor/PairForces.cuh"
#include "Interactor/ExternalForces.cuh"
#include "Interactor/NeighbourList/VerletList.cuh"

#include "AFM_potential.cuh"
#include "iogpf/iogpf.hpp"

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
          real fmod = -real(2.0)*(wellDepth/bi.r0)*exp(-dst*dst*invSigma)*dst*invSigma;
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
          real energy = -(wellDepth/bi.r0)*exp(-dst*dst*invSigma);
          
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

class molId_selector{
		int molId;
	public:    
		molId_selector(int molId):molId(molId){}

		bool isSelected(int particleIndex, shared_ptr<ParticleData> &pd){
			int part_molID = (pd->getMolId(access::cpu, access::read).raw())[particleIndex];
			return part_molID==molId;
		}
};

class molId_selector_positive{
	
	public:    

		bool isSelected(int particleIndex, shared_ptr<ParticleData> &pd){
			int part_molID = (pd->getMolId(access::cpu, access::read).raw())[particleIndex];
			return part_molID>=0;
		}
};

struct WCA_Wall: public ParameterUpdatable{
	
	real zwall;
	
	real wallRadius;
	real epsilon = 1;
	
	WCA_Wall(real zwall,real wallRadius):zwall(zwall),wallRadius(wallRadius){}
	
	__device__ __forceinline__ real3 force(const real4 &pos,const real &radius){
	
		const real z = abs(pos.z-zwall);
		const real effDiam = radius+wallRadius;
            
        if(z > real(1.122462)*effDiam){
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

struct ProbeForce: public ParameterUpdatable{
	
	real fmod = 0;
	real fmodIncr = 0.01;
	
	real2 descentAxis = {0.0,0.0};
	real k = 10;
	
	int termSteps;
	int increSteps;
	
	ProbeForce(int termSteps, int increSteps):termSteps(termSteps),increSteps(increSteps){}
	
	__device__ __forceinline__ real3 force(const real4 &pos,const real &planeRadius){
		
		real dx = pos.x-descentAxis.x;
		real dy = pos.y-descentAxis.y;
		real r  = sqrt(dx*dx+dy*dy);
		
		real fPlane = -real(2.0)*k*(r-planeRadius);
		
		return make_real3(fPlane*dx/r,fPlane*dy/r,-fmod);
	
	}
	
	std::tuple<const real4 *,const real *> getArrays(ParticleData *pd){
		auto pos = pd->getPos(access::location::gpu, access::mode::read);
		auto planeRadius = pd->getPlaneRadius(access::location::gpu, access::mode::read);
		return std::make_tuple(pos.raw(),planeRadius.raw());
	}
	
	void updateStep(int step){
		
		if(step > termSteps and step%increSteps == 0){
			fmod += fmodIncr;
		}
	}
	
	void setDescentAxis(real2 desAxis){
		descentAxis = desAxis;
	}
	
	real getCurrentForce(){
		return fmod;
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

void outputStateAndWall(std::ofstream& os,
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
		    << -2          <<  std::endl;
        
    }}
	
}

class probeMeasuring{
	
	friend std::ostream& operator<< (std::ostream& os, const probeMeasuring& probM);
	
	private:
	
		struct posForce{
			real3 pos;
			real3 force;
		};
	
		std::shared_ptr<System> sys;
		
		std::shared_ptr<ParticleData> pd;
		std::shared_ptr<ParticleGroup> pgProbe;
		
		std::shared_ptr<ExternalForces<ProbeForce>> probeForce;
		
		std::vector<posForce> posForceCurrent;
		std::vector<posForce> posForceMean;
		
		int probeN;
		
		
	public:
		
		probeMeasuring(std::shared_ptr<System> sys,
					   std::shared_ptr<ParticleData> pd,
					   std::shared_ptr<ParticleGroup> pgProbe,
					   std::shared_ptr<ExternalForces<ProbeForce>> probeForce)
					   :sys(sys),pd(pd),pgProbe(pgProbe),probeForce(probeForce){
			
			probeN = pgProbe->getNumberParticles();
		}
					   
		real3 computeProbePosition(){
			
			auto pos    = pd->getPos(access::location::cpu, access::mode::read);
			auto iterProbe = pgProbe->getIndexIterator(access::location::cpu);
			
			real3 probeCentroid = {0,0,0};
			
			fori(0,probeN){
				probeCentroid += make_real3(pos.raw()[iterProbe[i]]);
			}
			
			return probeCentroid/probeN;
			
		}
		
		real3 computeProbeForce(){
			
			auto force     = pd->getForce(access::location::cpu, access::mode::read);
			auto iterProbe = pgProbe->getIndexIterator(access::location::cpu);
			
			real3 probeForce = {0,0,0};
			
			fori(0,probeN){
				probeForce += make_real3(force.raw()[iterProbe[i]]);
			}
			
			return probeForce/probeN;
		}
					   
		void measure(){
			//posForceCurrent.push_back({this->computeProbePosition(),this->computeProbeForce()});
			posForceMean.push_back({this->computeProbePosition(),this->computeProbeForce()});
		}
};

std::ostream& operator<< (std::ostream& os, const probeMeasuring& probM){
	
	for(auto& pF : probM.posForceMean){
		os << pF.pos << " " << pF.force << std::endl;
	}
	
	return os;
	
}

using NVT = VerletNVT::GronbechJensen;

int main(int argc, char *argv[]){
	
	
	
	real wallRadius = 0.5;
	real E = 1;
    real sigma = 0.2;
    real cutOff = 1.5;
    real epsilon = 1;
    int nstepsTerm = 0;
	int printStepsTerm = 1000;
	Box box({70.0,70.0,300.0});
	real downwardForce = -0.1;
	int termStepsProbe = 1000;
	int increStepsProbe = 10000;
	real temperature = 2.479;
	real dt = 0.01;
	real viscosity = 0.1;
	int sortSteps = 500;
	int nsteps = 10000000;
    int printSteps = 1000;
    real initialVirusProbeSep = 0.5;
    real probeAtomRadius = 1.5;
    int measureSteps = 1000;
	
	////////////////////////////////////////////////////////////////////
    
    aminoAcidMap aminoMap;
    
    aminoMap.loadAminoAcids("aminoAcidsList.dat");
    aminoMap.applyMap2File("p22.top","p22_P.top");
    
    auto sys = std::make_shared<System>();
    ullint seed = 0xf31337Bada55D00dULL;
    sys->rng().setSeed(seed);
    
    ///////////////////////////INPUT DATA///////////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    auto pd = iogpf::read<iogpf::id,
                          iogpf::type,
                          iogpf::molId,
                          iogpf::mass,
                          iogpf::radius,
                          iogpf::planeRadius,
                          iogpf::pos>(sys,"p22_P.top");
                          
    int N = pd->getNumParticles();
    
    ////////////////////////PARTICLE GROUPS/////////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    auto pgAll = std::make_shared<ParticleGroup>(pd, sys, "All");
    
    molId_selector_positive mId_sel_pos;
    auto pgVirus = std::make_shared<ParticleGroup>(mId_sel_pos,pd, sys, "Virus");
    
    molId_selector mId_sel(-1);
    auto pgProbe = std::make_shared<ParticleGroup>(mId_sel,pd, sys, "Probe");
    
    /////////////////////INTERNAL INTERACTORS///////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    using ENM = BondedForces<BondedType::HarmonicPBC>;

    ENM::Parameters paramsENM;
    
    paramsENM.file = "p22.enm"; 
    BondedType::HarmonicPBC enm(box);
    auto bondedforces = std::make_shared<ENM>(pd, sys, paramsENM, enm);
    
    ////////////////////////////////////////////////////////////////////
    
    using GaussianForces = BondedForces<BondedType::GaussianPBC>;

    GaussianForces::Parameters paramsGF;
    
    paramsGF.file = "p22.gaussian";  
    BondedType::GaussianPBC gf(box,E,sigma);
    auto gaussianforces = std::make_shared<GaussianForces>(pd, sys, paramsGF, gf);
    
    ////////////////////////////////////////////////////////////////////
    
    using PairForces = PairForces<Potential::AFMPotential, CellList>;
    
    auto pot = std::make_shared<Potential::AFMPotential>(sys,cutOff,epsilon);
    
    PairForces::Parameters params;
    params.box = box;  //Box to work on
    auto pairforces = std::make_shared<PairForces>(pd, pgAll, sys, params, pot);
    
    /////////////////////EXTERNAL INTERACTORS///////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    
    //WALL
    
    real wallZ;
    
    real minZ =  INFINITY;
    real maxZ = -INFINITY;
    real radiusClosestMin = 0;
    real radiusClosestMax = 0;
    
    {
		auto pos      = pd->getPos(access::location::cpu, access::mode::read);
		auto radius   = pd->getRadius(access::location::cpu, access::mode::read);
		
		auto iterVirus = pgVirus->getIndexIterator(access::location::cpu);
		
		fori(0,pgVirus->getNumberParticles()){
			real z = pos.raw()[iterVirus[i]].z;
			if(z < minZ){
				minZ = z;
				radiusClosestMin = radius.raw()[iterVirus[i]];
			}

			if(z > maxZ){
				maxZ = z;
				radiusClosestMax = radius.raw()[iterVirus[i]];
			}
		}
	}
	
	wallZ = minZ-real(1.122462)*(wallRadius+radiusClosestMin);
    
    auto extWall = std::make_shared<ExternalForces<WCA_Wall>>(pd, pgAll, sys,std::make_shared<WCA_Wall>(wallZ,wallRadius));
    
    //DOWNWARD FORCE 
	auto downwardForceVirus = std::make_shared<ExternalForces<ConstantForce>>(pd, pgVirus, sys,std::make_shared<ConstantForce>(downwardForce));
	
	//PROBE FORCE
	std::shared_ptr<ProbeForce> probePot = std::make_shared<ProbeForce>(termStepsProbe,increStepsProbe);
    auto forcesProbe = std::make_shared<ExternalForces<ProbeForce>>(pd, pgProbe, sys,probePot);
    
    ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    ////////////////////////SIMULATION STARTS///////////////////////////
    
    Timer tim;
    tim.tic();
    
    std::ofstream out("p22.sp");
    std::ofstream outProbe("probe.dat");
    
    ////////////////////////THERMALIZATION//////////////////////////////
    
    {
		NVT::Parameters parTherm;
		parTherm.temperature = temperature; 
		parTherm.dt = dt;
		parTherm.viscosity = viscosity;  
		
		auto verletTherm = std::make_shared<NVT>(pd, pgVirus, sys,parTherm);
		
		verletTherm->addInteractor(bondedforces);
		verletTherm->addInteractor(gaussianforces);
		verletTherm->addInteractor(pairforces);
		verletTherm->addInteractor(extWall);
		verletTherm->addInteractor(downwardForceVirus);
		
		outputStateAndWall(out,sys,pd,box,wallZ,wallRadius);
		
		////////////////////////////////////////////////////////////////
		
		forj(0,nstepsTerm){
			verletTherm->forwardTime();
		
			//Write results
			if(j%printStepsTerm==1){
				outputStateAndWall(out,sys,pd,box,wallZ,wallRadius);
			}    
			
			if(j%sortSteps == 0){
				pd->sortParticles();
			}
		}
	}
	
    ////////////////////////////RUN/////////////////////////////////////
    
    //Set probe position
    
    real2 descentAxis;
    real3 virusCentroid = {0,0,0};
    
    {
		auto pos    = pd->getPos(access::location::cpu, access::mode::readwrite);
		auto radius = pd->getRadius(access::location::cpu, access::mode::readwrite);
	
		auto iterVirus = pgVirus->getIndexIterator(access::location::cpu);
		auto iterProbe = pgProbe->getIndexIterator(access::location::cpu);
		
		fori(0,pgVirus->getNumberParticles()){
			virusCentroid += make_real3(pos.raw()[iterVirus[i]]);	
		}
		
		virusCentroid /= pgVirus->getNumberParticles();
        
        real3 probeCentroid = {0,0,0};
		real  probeMinZ = INFINITY;
        
		fori(0,pgProbe->getNumberParticles()){
            probeCentroid += make_real3(pos.raw()[iterProbe[i]]);
			 if(make_real3(pos.raw()[iterProbe[i]]).z < probeMinZ){
                 probeMinZ = make_real3(pos.raw()[iterProbe[i]]).z;
             }
		}
        
        probeCentroid /= pgProbe->getNumberParticles();
		
		//Move probe to {vC.x,vC.y,maxZ+real(0.5)*(probeRadius+radiusClosest)+initialVirusProbeSep}
		
		real3 translationVector = { - probeCentroid.x + virusCentroid.x,
								    - probeCentroid.y + virusCentroid.y,
								    - probeMinZ       + (maxZ+real(1.122462)*(probeAtomRadius+radiusClosestMax)+initialVirusProbeSep)};
		
		fori(0,pgProbe->getNumberParticles()){
			
			pos.raw()[iterProbe[i]].x = pos.raw()[iterProbe[i]].x + translationVector.x;
			pos.raw()[iterProbe[i]].y = pos.raw()[iterProbe[i]].y + translationVector.y;
			pos.raw()[iterProbe[i]].z = pos.raw()[iterProbe[i]].z + translationVector.z;
		
		}
		descentAxis = {virusCentroid.x,virusCentroid.y};
		
		probePot->setDescentAxis(descentAxis);
	}
	
	outProbe << "# " << descentAxis.x << " " << descentAxis.y << std::endl; 
	
	probeMeasuring pM(sys,pd,pgProbe,forcesProbe);
	
    {
		NVT::Parameters par;
		par.temperature = temperature; 
		par.dt = dt;
		par.viscosity = viscosity;  
		
		auto verlet = std::make_shared<NVT>(pd, pgAll, sys, par);
		
		verlet->addInteractor(bondedforces);
		//verlet->addInteractor(gaussianforces);
		verlet->addInteractor(pairforces);
		verlet->addInteractor(extWall);
		//verlet->addInteractor(forcesProbe);
		
		outputStateAndWall(out,sys,pd,box,wallZ,wallRadius);
		
		////////////////////////////////////////////////////////////////
		
		//Run the simulation
		forj(0,nsteps){
			verlet->forwardTime();
		
			//Write results
			if(j%printSteps==1){
				outputStateAndWall(out,sys,pd,box,wallZ,wallRadius);
			}
			
			if(j%measureSteps == 0){
				real3 probePos = pM.computeProbePosition();
				
				real dx = (descentAxis.x-probePos.x);
				real dy = (descentAxis.y-probePos.y);
				
				real  dstAxis = sqrt(dx*dx+dy*dy);
				outProbe << probePot->getCurrentForce() << " " << dstAxis << " " << pM.computeProbePosition().z << std::endl;
				
				if(pM.computeProbePosition().z < virusCentroid.z){
					break;
				}
			}
			
			if(j%sortSteps == 0){
				pd->sortParticles();
			}
		}
    
	}
	
	//outProbe << pM << std::endl;
	
    auto totalTime = tim.toc();
    sys->log<System::MESSAGE>("mean FPS: %.2f", nsteps/totalTime);
    sys->finish();
    
    return EXIT_SUCCESS;
}
