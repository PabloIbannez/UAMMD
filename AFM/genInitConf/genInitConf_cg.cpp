#include <proteinManager/proteinManager.hpp>
#include <proteinManager/tools/geometricTransformations.hpp>
#include <proteinManager/tools/centroid.hpp>
#include <proteinManager/tools/massesManager.hpp>
#include <proteinManager/tools/radiusManager.hpp>
#include <proteinManager/tools/centerOfMass.hpp>
#include <proteinManager/tools/coarseGrainedManager.hpp>
#include <proteinManager/tools/enm.hpp>

using namespace proteinManager;

struct atomType{
	
    int serial;
    int modelId;
    std::string chainId;
    std::string resName;
    int resSeq;
    real mass;
    real3 pos;
	real radius;
};

struct clash{
    
    int i;
    int j;
    double rmin;
    double r;
    
    bool operator<(const clash& rhs) const
    {
        return (r < rhs.r);
    }
};


int main(){
    
    real rCut_ENM = 1;
    real rCut_BOND = 1.5;
    
    STRUCTURE pdbInput;
    STRUCTURE pdbOutput;
    
    pdbInput.loadPDB("p22ClearSep.pdb");
    //pdbInput.loadPDB("p22Test.pdb");
    pdbInput.renumber();
    
    real3 center = computeCentroid(pdbInput);
    geometricTransformations::rotation(pdbInput,center,{1,0,0},34.0*(M_PI/180.0));
    geometricTransformations::rotation(pdbInput,center,{0,1,0},13.0*(M_PI/180.0));
    
    ////////////////////////////////////////////////////////////////////
    
    coarseGrainedManager::coarseGrainedGenerator cg;
    cg.loadCGmodel("./RES2BEAD_noH/aminoAcid2bead_RES2BEAD_noH.map","./RES2BEAD_noH/bead2atom_RES2BEAD_noH.map");
    
    cg.applyCoarseGrainedMap<proteinManager::coarseGrainedManager::coarseGrainedMappingSchemes::ca>(pdbInput,pdbOutput);
    
    ////////////////////////////////////////////////////////////////////
    
    geometricTransformations::uniformScaling(pdbOutput,0.1);
    
    ////////////////////////////////////////////////////////////////////
    
    radiusManager rM;
    
    rM.loadRadiusData("aminoacidsRadius.dat");
    rM.applyRadiusData(pdbOutput);
    
    massesManager mM;
    
    mM.loadMassesData("aminoacidsMasses.dat");
    mM.applyMassesData(pdbOutput);
    
    ////////////////////////////////////////////////////////////////////
    
    std::vector<atomType> atomVector;
    
    atomType aTBuffer;
    
    for(MODEL&   mdl : pdbOutput.model()){
    for(CHAIN&   ch  : mdl.chain()  ){
    for(RESIDUE& res : ch.residue() ){
    for(ATOM&    atm : res.atom()   ){
        
        aTBuffer.serial  = atm.getAtomSerial();
        aTBuffer.modelId = atm.getModelId();
        aTBuffer.resName = atm.getResName();
        aTBuffer.mass    = atm.getAtomMass();
        aTBuffer.pos     = atm.getAtomCoord();
        aTBuffer.radius  = atm.getAtomRadius();
        
        atomVector.push_back(aTBuffer);
    }}}}
    
    std::vector<clash> clashedVector;
    clash clashBuffer;
    
    for(int i = 0;    i<atomVector.size();i++){
        
        std::cout << "Looking for clashed " << i+1 <<"/"<<atomVector.size() << std::endl;
        
        for(int j = i +1; j<atomVector.size();j++){
            
            if(atomVector[i].modelId != atomVector[j].modelId){
                
                real3 ri = atomVector[i].pos;
                real3 rj = atomVector[j].pos;
                real3 rij = rj - ri;
                double r = sqrt(dot(rij,rij));
                
                double radius_i = atomVector[i].radius;
                double radius_j = atomVector[j].radius;
                
                double diamEff  = (radius_i+radius_j);
                       diamEff *= 1.122462;
                
                if(r <= diamEff){
                    clashedVector.push_back({i,j,diamEff,r});
                }
            }
        }
    }
    
    std::sort(clashedVector.begin(),clashedVector.end());
    
    ////////////////////////////////////////////////////////////////////
    
    for(int i = clashedVector.size() - 1; i >= 0 ; i--){
        
        std::cout << "Solving clashed " << clashedVector.size()-i << "/" << clashedVector.size()-1 << std::endl;
        
        //std::cout << i << " " 
        //          << clashedVector[i].i  << " " << clashedVector[i].j    << " "
        //          << clashedVector[i].r  << " " << clashedVector[i].rmin << std::endl << std::endl;
        
        if(clashedVector[i].r < clashedVector[i].rmin){
            
            double gamma = clashedVector[i].r/(1.01*clashedVector[i].rmin);
            
            atomVector[clashedVector[i].i].radius*=gamma;
            atomVector[clashedVector[i].j].radius*=gamma;
            
            //std::cout << " scale radius: " << clashedVector[i].r << " " << gamma << std::endl << std::endl; 
            
            for(auto& cl : clashedVector){
                if(cl.i == clashedVector[i].i){
                    cl.rmin = 1.122462*(atomVector[cl.i].radius+atomVector[cl.j].radius);
                }
                
                if(cl.j == clashedVector[i].j){
                    cl.rmin = 1.122462*(atomVector[cl.i].radius+atomVector[cl.j].radius);
                }
                
            }
        }
    }
    
    //std::cin.get();
    
    
    ////////////////////////////////////////////////////////////////////
    /*
    for(int i = 0;    i<atomVector.size();i++){
        
        std::cout << "Check " << i+1 <<"/"<<atomVector.size() << std::endl;
        
        for(int j = i +1; j<atomVector.size();j++){
            
            if(atomVector[i].modelId != atomVector[j].modelId){
                
                real3 ri = atomVector[i].pos;
                real3 rj = atomVector[j].pos;
                real3 rij = rj - ri;
                double r = sqrt(dot(rij,rij));
                
                double radius_i = atomVector[i].radius;
                double radius_j = atomVector[j].radius;
                
                double diamEff  = (radius_i+radius_j);
                       diamEff *= 1.122462;
                
                if(r <= diamEff){
                    std::cout << "ERROR" << std::endl;
                    std::exit(0);
                }
            }
        }
    }
    */
    ////////////////////////////////////////////////////////////////////
    
    /*
    do{
        
        clashedVector.clear();
        
        for(int i = 0;    i<atomVector.size();i++){
            
            std::cout << "Clashed " << i+1 <<"/"<<atomVector.size() << std::endl;
            
            for(int j = i +1; j<atomVector.size();j++){
                
                if(atomVector[i].modelId != atomVector[j].modelId){
                    
                    real3 ri = atomVector[i].pos;
                    real3 rj = atomVector[j].pos;
                    real3 rij = rj - ri;
                    double r = sqrt(dot(rij,rij));
                    
                    double radius_i = atomVector[i].radius;
                    double radius_j = atomVector[j].radius;
                    
                    double diamEff  = (radius_i+radius_j);
                           diamEff *= 1.122462;
                    
                    if(r <= diamEff){
                        clashedVector.push_back({i,j,diamEff,r});
                    }
                }
            }
        }
        
        if(clashedVector.size()>0){
            
            std::cout << "sorting... " << std::endl;
            std::sort(clashedVector.begin(),clashedVector.end());
            
            clashBuffer = clashedVector.back();
            
            double radius_i = atomVector[clashBuffer.i].radius;
            double radius_j = atomVector[clashBuffer.j].radius;
            
            double diamEff  = (radius_i+radius_j);
                   diamEff *= 1.122462;
                   
            double gamma = clashBuffer.r/(1.01*diamEff);
            
            std::cout << clashBuffer.r << " " << gamma << std::endl;
            
            atomVector[clashBuffer.i].radius*=gamma;
            atomVector[clashBuffer.j].radius*=gamma;
                
        }
        
    } while(clashedVector.size()>0);
    
    */
    
    std::ofstream enmFile("p22.enm");
    std::ofstream bondFile("p22.gaussian");
    
    int removedParticles_counter = 0;
    int atomCount = 0;
    int ENM_counter = 0;
    int Gaussian_counter = 0;
    
    for(int i = 0;    i<atomVector.size();i++){
        if(atomVector[i].radius < 0.2){
            atomVector.erase(atomVector.begin()+i);
            removedParticles_counter++;
            i=0;
        }
    }
    
    for(atomType&  atm : atomVector){
        atm.serial = atomCount;
        atomCount ++;
    }
    
    for(int i = 0;    i<atomVector.size();i++){
        
        std::cout << "Generating ENM and Bonds " << i+1 <<"/"<<atomVector.size() << std::endl;
        
        for(int j = i +1; j<atomVector.size();j++){
            
            real3 ri = atomVector[i].pos;
            real3 rj = atomVector[j].pos;
            real3 rij = rj - ri;
            double r = sqrt(dot(rij,rij));
            
            if(atomVector[i].modelId == atomVector[j].modelId){
                
                if(r < rCut_ENM){
                    
                    enmFile << std::setw(10) << atomVector[i].serial <<
                               std::setw(10) << atomVector[j].serial <<
                               std::setprecision(6)                  <<
                               std::setw(12) << r                    << std::endl;
                               
                    ENM_counter++;
                    
                }
                
            } else {
                
                if(r < rCut_BOND){
                    
                    bondFile << std::setw(10) << atomVector[i].serial <<
                                std::setw(10) << atomVector[j].serial <<
                                std::setprecision(6)                  <<
                                std::setw(12) << r                    << std::endl;
                    
                    Gaussian_counter ++;
                    
                }
                
            }
            
        }
    }
    
    ////////////////////////////////////////////////////////////////////
    
    std::ofstream topFile("p22.top");
    
    
    for(atomType&  atm : atomVector){
        topFile << std::right
                << std::fixed
                << std::setprecision(4)
                << std::setw(10)                   
                << atm.serial                       << " " 
                << std::setw(5)                    
                << atm.resName                      << " " 
                << std::setw(5)                    
                << atm.modelId                      << " " 
                << std::setw(10)                   
                << atm.mass                         << " " 
                << std::setw(10)                   
                << atm.radius                       << " " 
                << std::setw(10)
                << atm.pos                          << std::endl;
        
        atomCount ++;
    }
    
    std::ofstream spFile("p22.sp");
    
    for(atomType&  atm : atomVector){
        spFile  << std::right
                << std::fixed
                << std::setprecision(4)
                << std::setw(10)
                << atm.pos                          << " "
                << std::setw(10)                   
                << real(1.122462)*atm.radius        << " " 
                << std::setw(5)                    
                << atm.modelId                      << std::endl;
    }
    
    ////////////////////////////////////////////////////////////////////
    
    std::cout << "Removed particles: " << removedParticles_counter << std::endl;
    
    int status;
    std::stringstream ss;
    
    ss << "sed -i \'1s/^/" << ENM_counter <<"\\n/\' p22.enm";
    std::cout << ss.str() << std::endl;
    status = std::system(ss.str().c_str());
    
    ss.clear();
    ss.str(std::string());
    ss << "sed -i \'1s/^/" << Gaussian_counter <<"\\n/\' p22.gaussian";
    std::cout << ss.str() << std::endl;
    status = std::system(ss.str().c_str());
    
    return status;
    
}
