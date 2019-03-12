#include "uammd.cuh"

#include "Integrator/SteepestDescent.cuh"
#include "Integrator/VerletNVT.cuh"
#include "Interactor/BondedForces.cuh"
#include "Interactor/PairForces.cuh"
#include "Interactor/ExternalForces.cuh"
#include "Interactor/NeighbourList/VerletList.cuh"

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

struct ProbeForce: public ParameterUpdatable{
    
	
    real3 probePosition = {0,0,0};
    
    real  epsilon = 1;
    real  probeRadius = 20;
    
	ProbeForce(real epsilon,real probeRadius):epsilon(epsilon),probeRadius(probeRadius){}
	
	__device__ __forceinline__ real3 force(const real4 &pos,const real &radius){
	    
        real3 rtp = probePosition-make_real3(pos);
        real  r2 = dot(rtp,rtp);
        real  r  = sqrt(r2);
        
        real effDiam = probeRadius+radius;
        
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
    
    void setProbePosition(real3 newProbePosition){
        probePosition = newProbePosition;
    }
    
    void setProbeHeight(real newHeight){
        probePosition.z = newHeight;
    }
    
    real3 getProbePosition(){
        return probePosition;
    }
    
    void setProbeRadius(real newProbeRadius){
        probeRadius = newProbeRadius;
    }
    
    real getProbeRadius(){
        return probeRadius;
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

void outputState_ProbeWall(std::ofstream& os,
                           std::shared_ptr<System> sys,
                           std::shared_ptr<ParticleData> pd,
                           Box box,
                           real3 probePosition,
                           real probeRadius,
                           real wallZ,
                           real wallRadius){
							
	outputState(os,sys,pd,box);
    
    os << probePosition                << " " 
       << real(1.122462)*probeRadius   << " "
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

template <class probeForce>
class probeMeasuring{
    
    private:
        
        shared_ptr<System> sys;
        
        shared_ptr<ParticleData> pd;
        shared_ptr<ParticleGroup> pg;
        
        shared_ptr<probeForce> probeF;
        
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
    
        probeMeasuring(shared_ptr<System> sys,
                       shared_ptr<ParticleData> pd,
                       shared_ptr<ParticleGroup> pg,
                       shared_ptr<probeForce> probeF):sys(sys),pd(pd),pg(pg),probeF(probeF){
          
          sys->log<System::MESSAGE>("[probeMeasuring] Created.");
          
          cudaStreamCreate(&stream);
          
          cudaMallocManaged((void**)&totalForce,sizeof(real3));
          this->setUpCubReduction();
        }
        
        ~probeMeasuring(){
          
            sys->log<System::MESSAGE>("[probeMeasuring] Destroyed.");
            
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
            
            probeF->sumForce(stream);
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
	
    real probeInitHeight = 30;
    real initialVirusProbeSep = 3;
    real probeRadius = 20;
    real probeEpsilon = 1;
	real wallRadius = 0.5;
	real E = 0.1;
    real sigma = 0.1;
    real cutOff = 1.5;
    real epsilon = 1;
    int nstepsTerm = 10000;
	int printStepsTerm = 1000;
	Box box({70.0,70.0,300.0});
	real downwardForce = 0;
	int termStepsProbe = 1000;
	int increStepsProbe = 10000;
	real temperature = 2.479;
	real dt = 0.01;
	real viscosity = 0.1;
	int sortSteps = 500;
	int nsteps = 10000000;
    int printSteps = 10000;
    
    real probeAtomRadius = 1.5;
    int measureSteps = 10000;
	
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
                          iogpf::pos>(sys,"p22_P.top");
                          
    int N = pd->getNumParticles();
    
    ////////////////////////PARTICLE GROUPS/////////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    auto pgAll = std::make_shared<ParticleGroup>(pd, sys, "All");
    
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
    
    
    real wallZ;
    real probeZ;
    
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
    
    probeZ = maxZ + real(1.122462)*(probeRadius+radiusClosestMax) + probeInitHeight;
    
	std::shared_ptr<ProbeForce> probePot = std::make_shared<ProbeForce>(probeEpsilon,probeRadius);
    probePot->setProbePosition({0,0,probeZ});
    
    auto forcesProbe = std::make_shared<ExternalForces<ProbeForce>>(pd, pgAll, sys,probePot);
    
    //////////////////////////MEASURING/////////////////////////////////
    
    auto measuring = std::make_shared<probeMeasuring<ExternalForces<ProbeForce>>>(sys,pd,pgAll,forcesProbe);
    
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
		
		auto verletTherm = std::make_shared<NVT>(pd, pgAll, sys,parTherm);
		
		verletTherm->addInteractor(bondedforces);
		verletTherm->addInteractor(gaussianforces);
		verletTherm->addInteractor(pairforces);
		verletTherm->addInteractor(extWall);
		verletTherm->addInteractor(downwardForceVirus);
		
		outputState_ProbeWall(out,sys,pd,box,
                              probePot->getProbePosition(),
                              probePot->getProbeRadius(),
                              wallZ,wallRadius);
		
		////////////////////////////////////////////////////////////////
		
		forj(0,nstepsTerm){
			verletTherm->forwardTime();
		
			//Write results
			if(j%printStepsTerm==1){
				outputState_ProbeWall(out,sys,pd,box,
                                      probePot->getProbePosition(),
                                      probePot->getProbeRadius(),
                                      wallZ,wallRadius);
			}    
			
			if(j%sortSteps == 0){
				pd->sortParticles();
			}
		}
	}
	
    ////////////////////////////RUN/////////////////////////////////////
    
    //Set probe position
    
    real3 probePosition = {0,0,0};
    
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
		
		//Move probe to {vC.x,vC.y,maxZ+(probeRadius+radiusClosest)+initialVirusProbeSep}
		
        probePosition = {virusCentroid.x,virusCentroid.y,maxZ + (probeRadius+radiusClosestMax)+initialVirusProbeSep};
        probePot->setProbePosition(probePosition);
		
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
		verlet->addInteractor(forcesProbe);
		
		outputState_ProbeWall(out,sys,pd,box,
                              probePot->getProbePosition(),
                              probePot->getProbeRadius(),
                              wallZ,wallRadius);
		
		////////////////////////////////////////////////////////////////
		
        int k =0;
		//Run the simulation
		forj(0,nsteps){
			verlet->forwardTime();
            
			if(j%printSteps==1){
				outputState_ProbeWall(out,sys,pd,box,
                                      probePot->getProbePosition(),
                                      probePot->getProbeRadius(),
                                      wallZ,wallRadius);
			}
			
			if(j%measureSteps == 0){
                // 1 kj/(mol·nm) = 0.0016605391 nN
                outProbe << probePot->getProbePosition().z << " " << -measuring->sumForce().z*real(0.0016605391) << std::endl;
                
                if(k ==1){
                    k=-1;
                    real3 currentProbePosition = probePot->getProbePosition();
                    probePot->setProbeHeight(currentProbePosition.z-0.1);
                }
                k++;
                
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
