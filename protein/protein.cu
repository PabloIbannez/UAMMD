#include "uammd.cuh"

#include "Integrator/SteepestDescent.cuh"
#include "Integrator/VerletNVT.cuh"
#include "Interactor/BondedForces.cuh"
#include "Interactor/PairForces.cuh"
#include "Interactor/NeighbourList/VerletList.cuh"



#define DEBUG

#include "proteinPotential.cuh"
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

class surfaceSelector{
    
    public:    

	bool isSelected(int particleIndex, shared_ptr<ParticleData> &pd){

	    return (pd->getSurf(access::cpu, access::read).raw())[particleIndex];
	}

};

int main(int argc, char *argv[]){ 
    
    aminoAcidMap aminoMap;
    
    aminoMap.loadAminoAcids("aminoAcidsList.dat");
    aminoMap.applyMap2File("barnase_barstar.top","barnase_barstar_P.top");
    aminoMap.applyMap2File("sasaWeiPar.dat","sasaWeiPar_P.dat");
    
    auto sys = std::make_shared<System>();
    ullint seed = 0xf31337Bada55D00dULL;
    sys->rng().setSeed(seed);
    
    auto pd = iogpf::read<iogpf::id,
                          iogpf::type,
                          iogpf::molId,
                          iogpf::SASA,
                          iogpf::surface,
                          iogpf::mass,
                          iogpf::pos,
                          iogpf::c12,
                          iogpf::c6,
                          iogpf::charge,
                          iogpf::solvE>(sys,"barnase_barstar_P.top");
                          
    int N = pd->getNumParticles();
    
    auto pg = std::make_shared<ParticleGroup>(pd, sys, "All");
    
    using NVT = VerletNVT::GronbechJensen;
    
    NVT::Parameters par;
    par.temperature = 2.479;
    par.dt = 0.01;
    par.viscosity = 1;  
    
    auto verlet = std::make_shared<NVT>(pd, pg, sys, par);
    
    ////////////////////////////////////////////////////////////////////
    
    Box box(25);
    
    using BondedForces = BondedForces<BondedType::HarmonicPBC>;

    BondedForces::Parameters paramsBF;
    
    paramsBF.file = "barnase_barstar.bond";  //Box to work on
    BondedType::HarmonicPBC bt(box);
    auto bondedforces = std::make_shared<BondedForces>(pd, sys, paramsBF, bt);
    
    ////////////////////////////////////////////////////////////////////
    
    using PairForces = PairForces<Potential::proteinPotential, VerletList>;
    
    real cutOff = 2;
    
    auto pot = std::make_shared<Potential::proteinPotential>(sys,cutOff,0.6,80);
    {
	std::ifstream weiParFile("sasaWeiPar_P.dat");
	
	std::stringstream ss;
	std::string line;
	
	int   intBuffer1;
	int   intBuffer2;
	real  realBuffer;
	
	while(std::getline(weiParFile,line)) {
                
                ss.str(line);
                
                ss >> intBuffer1 >> intBuffer2 >> realBuffer;
		
		pot->setPotParameters(intBuffer1, intBuffer2, {cutOff,realBuffer});
	}
	
	
    }
    
    //Surface atoms group
    
    surfaceSelector surfSel;
    
    auto pgSurf = std::make_shared<ParticleGroup>(surfSel, pd, sys, "surfParticles");
    

    PairForces::Parameters paramsPF;
    paramsPF.box = box;  //Box to work on
    auto pairforces = std::make_shared<PairForces>(pd, pgSurf, sys, paramsPF, pot);
    
    ////////////////////////////////////////////////////////////////////
    
    /*
    SteepestDescent::Parameters STpar;
    STpar.h = 0.1;
    STpar.epsilon = 1;
    STpar.maxSteps = 1000;
    
    auto st = std::make_shared<SteepestDescent>(pd, pg, sys, STpar);
    
    st->addInteractor(bondedforces);
    st->addInteractor(pairforces);
    
    while(st->minimizationStep());
    */
    
    ////////////////////////////////////////////////////////////////////
    
    verlet->addInteractor(bondedforces);
    verlet->addInteractor(pairforces);
    
    ////////////////////////////////////////////////////////////////////
    
    Timer tim;
    tim.tic();
    
    std::ofstream out("barnase_barstar.sp");
    
    int nsteps = 100000;
    int printSteps = 100;
    
    verlet->forwardTime();
    std::cin.get();
    
    //Run the simulation
    forj(0,nsteps){
        verlet->forwardTime();
    
        //Write results
        if(j%printSteps==1){
            sys->log<System::DEBUG1>("[System] Writing to disk...");
            //continue;
            auto pos   = pd->getPos(access::location::cpu, access::mode::read);
	    auto c12   = pd->getC12(access::location::cpu, access::mode::read);
	    auto c6    = pd->getC6(access::location::cpu, access::mode::read);
	    auto surf  = pd->getSurf(access::location::cpu, access::mode::read);
	    auto mId   = pd->getMolId(access::location::cpu, access::mode::read);
            const int * sortedIndex = pd->getIdOrderedIndices(access::location::cpu);
            out<<"#Lx="<<0.5*box.boxSize.x<<";Ly="<<0.5*box.boxSize.y<<";Lz="<<0.5*box.boxSize.z<<";"<<std::endl;
            real3 p;
            fori(0,N){
                real4 pc = pos.raw()[sortedIndex[i]];
		int surfp = surf.raw()[sortedIndex[i]];
		real c12p = c12.raw()[sortedIndex[i]];
		real c6p = c6.raw()[sortedIndex[i]];
		real mIdp = mId.raw()[sortedIndex[i]];
		real sigma = std::pow(c12p/c6p,1.0/3.0);
                p = box.apply_pbc(make_real3(pc));
                int type = pc.w;
                out<<p<<" "<<sigma<<" "<<mIdp<<std::endl;
            }
        }    
        
        if(j%500 == 0){
            //pd->sortParticles();
        }
    }
    
    
    auto totalTime = tim.toc();
    sys->log<System::MESSAGE>("mean FPS: %.2f", nsteps/totalTime);
    //sys->finish() will ensure a smooth termination of any UAMMD module.
    sys->finish();
    
    return EXIT_SUCCESS;
}
