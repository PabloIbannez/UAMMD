#include <proteinManager/proteinManager.hpp>
#include <proteinManager/tools/geometricTransformations.hpp>
#include <proteinManager/tools/massesManager.hpp>
#include <proteinManager/tools/radiusManager.hpp>
#include <proteinManager/tools/centerOfMass.hpp>
#include <proteinManager/tools/coarseGrainedManager.hpp>
#include <proteinManager/tools/enm.hpp>

using namespace proteinManager;

struct atomType{
	
    int serial;
    int modelId;
    std::string resName;
    real mass;
    real3 pos;
	real radius;
};

struct probeAtomType{
	
	real3 pos;
    real  R;
	real  rXY;
};

int main(){
    
    real K = 2000;
    
    real rCut_ENM = 1;
    real rCut_BOND = 1.5;
    real rCut_PROBE = 1.1;
    
    real probeZinit = 150;
    
    STRUCTURE pdbInput;
    STRUCTURE pdbOutput;
    
    pdbInput.loadPDB("p22ClearSep.pdb");
    //pdbInput.loadPDB("p22Test.pdb");
    pdbInput.renumber();
    
    ////////////////////////////////////////////////////////////////////
    
    geometricTransformations::uniformScaling(pdbInput,0.1);
    
    ////////////////////////////////////////////////////////////////////
    
    coarseGrainedManager::coarseGrainedGenerator cg;
    cg.loadCGmodel("./RES2BEAD_noH/aminoAcid2bead_RES2BEAD_noH.map","./RES2BEAD_noH/bead2atom_RES2BEAD_noH.map");
    
    cg.applyCoarseGrainedMap<proteinManager::coarseGrainedManager::coarseGrainedMappingSchemes::ca>(pdbInput,pdbOutput);
    
    ////////////////////////////////////////////////////////////////////
    
    radiusManager rM;
    
    rM.loadRadiusData("aminoacidsRadius.dat");
    rM.applyRadiusData(pdbOutput);
    
    massesManager mM;
    
    mM.loadMassesData("aminoacidsMasses.dat");
    mM.applyMassesData(pdbOutput);
    
    real3 centerOfMass = computeCenterOfMass(pdbOutput);
    
    std::cout << "Center of mass: " << centerOfMass << std::endl;
    
    ////////////////////////////////////////////////////////////////////
    
    //Buffer atom vector
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
    
    int ENM_counter = 0;
    int Gaussian_counter = 0;
    
    std::ofstream enmFile("p22.enm");
    std::ofstream bondFile("p22.gaussian");
    
    for(int i = 0;    i<atomVector.size();i++){
        
        std::cout << "Virus " << i+1 <<"/"<<atomVector.size() << std::endl;
        
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
                               std::setw(12) << K                    <<
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
                    
                    double radius_i = atomVector[i].radius;
                    double radius_j = atomVector[j].radius;
                    
                    double diamEff  = (radius_i+radius_j);
                           diamEff *= 1.122462;
                    
                    if(r <= diamEff){ //overlap, 2^(1/6)
                        double gamma = r/(1.01*diamEff);
                        
                        atomVector[i].radius = radius_i*gamma;
                        atomVector[j].radius = radius_j*gamma;
                        
                        //std::cout << "gamma: " << gamma << std::endl;
                        //std::cout << r << " " << diamEff << std::endl;
                        //std::cout << atomVector[i].serial << " " << atomVector[i].resName << " Old radii: " << radius_i << " new radii: " << radius_i*gamma << std::endl;
                        //std::cout << atomVector[j].serial << " " << atomVector[i].resName << " Old radii: " << radius_j << " new radii: " << radius_j*gamma << std::endl;
                        //std::cout << r << " " << diamEff*gamma << std::endl;
                        
                        if(gamma > 1){
                            throw std::runtime_error("ERROR, gamma>1");
                        }
                    } 
                    
                }
                
            }
            
        }
    }
    
    ////////////////////////////////////////////////////////////////////
    
    std::ofstream topFile("p22.top");
    
    int atomCount = 0;
    for(auto    atm : atomVector){
        topFile << std::right
                << std::fixed
                << std::setprecision(4)
                << std::setw(10)                   
                << atomCount                        << " " 
                << std::setw(5)                    
                << atm.resName                      << " " 
                << std::setw(5)                    
                << atm.modelId                      << " " 
                << std::setw(10)                   
                << atm.mass                         << " " 
                << std::setw(10)                   
                << atm.radius                       << " " 
                << std::setw(10)                   
                << "0.0"                            << " " 
                << std::setw(10)
                << atm.pos                          << std::endl;
        
        atomCount ++;
    }
    
    ////////////////////////////////////////////////////////////////////
    
    std::ifstream probeFile("probe.sp");
    
    std::vector<probeAtomType> probeAtoms;
    
    real3 posBuffer;
    real  radiusBuffer;
    int   colorBuffer;
    
    probeAtomType pABuffer;
    
    while(probeFile >> posBuffer.x >> posBuffer.y >> posBuffer.z >> radiusBuffer >> colorBuffer){
        
        //std::cout << posBuffer << " " << radiusBuffer << " " << colorBuffer << std::endl;
        
        if(posBuffer.z < 0){
            
            posBuffer.z += probeZinit;
            
            pABuffer.pos = posBuffer;
            pABuffer.R   = radiusBuffer;
            pABuffer.rXY = sqrt(posBuffer.x*posBuffer.x+posBuffer.y*posBuffer.y);
            
            probeAtoms.push_back(pABuffer);
        }
    }
    
    for(int i=0;i<probeAtoms.size();i++){
        
        std::cout << "Probe " << i << "/" << probeAtoms.size() << std::endl;
        
        topFile << std::right
                << std::fixed
                << std::setprecision(4)
                << std::setw(10)                   
                << atomCount                        << " " 
                << std::setw(5)                    
                << -1                               << " " 
                << std::setw(5)                    
                << -1                               << " " 
                << std::setw(10)                   
                << 100.0                            << " " 
                << std::setw(10)                   
                << probeAtoms[i].R                  << " " 
                << std::setw(10)                   
                << probeAtoms[i].rXY                << " " 
                << std::setw(10)
                << probeAtoms[i].pos                << std::endl;
        
        for(int j=i+1; j<probeAtoms.size();j++){
				
				real3 ri = probeAtoms[i].pos;
				real3 rj = probeAtoms[j].pos;
				real3 rij = rj - ri;
				double r = sqrt(dot(rij,rij));
					
				if(r < rCut_PROBE){
					
					enmFile << std::setw(10) << atomCount                <<
							   std::setw(10) << atomCount+j-i            <<
							   std::setprecision(6)                      <<
							   std::setw(12) << K                        <<
							   std::setw(12) << r                        << std::endl;
							   
					ENM_counter++;
					
				}
		}
        
        atomCount ++;
    }
    
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
