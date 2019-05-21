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

#define CG

real getCharge(std::string& resName){
    
    if(resName == "LYS" or resName == "ARG"){
        return 1.0;
    }
    
    if(resName == "ASP" or resName == "GLU"){
        return -1.0;
    }
    
    if(resName == "HIS"){
        return 0.5;
    }
    
}

int main(int argc, char *argv[]){
    
    bool cargo;
    
    std::string inputFileName;
    std::string inputFileCargoName;
    std::string outputName;
    
    if(argc == 3){
        inputFileName = argv[1];
        outputName    = argv[2];
        cargo = false;
    } else if(argc == 4){
        inputFileName      = argv[1];
        inputFileCargoName = argv[2];
        outputName         = argv[3];
        cargo = true;
    } else {
        std::cerr << "Input format error" << std::endl;
        std::cerr << "No cargo: (virus).pdb output" << std::endl;
        std::cerr << "No cargo: (virus).pdb (cargo).pdb output" << std::endl;
    }
    
    std::string outputTopName      = outputName + std::string(".top");
    std::string outputSpName       = outputName + std::string(".sp");
    std::string outputENMName      = outputName + std::string(".enm");
    std::string outputGaussianName = outputName + std::string(".gaussian");
    
    real rCut_ENM = 1;
    real rCut_BOND = 1.5;
    
    int modelCountOffset = 0;
    
    STRUCTURE pdbInput;
    STRUCTURE pdbOutput;
    
    pdbInput.loadPDB(inputFileName);
    pdbInput.renumber();
    
    real3 center = computeCentroid(pdbInput);
    geometricTransformations::rotation(pdbInput,center,{1,0,0},34.0*(M_PI/180.0));
    geometricTransformations::rotation(pdbInput,center,{0,1,0},13.0*(M_PI/180.0));
    
    ////////////////////////////////////////////////////////////////////
    
    #ifdef CG
    coarseGrainedManager::coarseGrainedGenerator cg;
    cg.loadCGmodel("./RES2BEAD_noH/aminoAcid2bead_RES2BEAD_noH.map","./RES2BEAD_noH/bead2atom_RES2BEAD_noH.map");
    
    cg.applyCoarseGrainedMap<proteinManager::coarseGrainedManager::coarseGrainedMappingSchemes::ca>(pdbInput,pdbOutput);
    #endif
    
    ////////////////////////////////////////////////////////////////////
    
    for(MODEL&   mdl : pdbOutput.model()){
    for(CHAIN&   ch  : mdl.chain()  ){
    for(RESIDUE& res : ch.residue() ){
    for(ATOM&    atm : res.atom()   ){
        atm.setAtomName(res.getResName());
    }}}}
    
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
    
    ////////////////////////////////////////////////////////////////////
    
    int removedParticles_counter = 0;
    
    for(int i = 0;    i<atomVector.size();i++){
        if(atomVector[i].radius < 0.2){
            atomVector.erase(atomVector.begin()+i);
            removedParticles_counter++;
            i=0;
        }
    }
    
    bool tooFarAway;
    
    for(int i = 0;    i<atomVector.size();i++){
		
		std::cout << "Removing too far away particles " << i+1 <<"/"<<atomVector.size() << std::endl;
        
        tooFarAway = true;
        
        for(int j = 0; j<atomVector.size() and tooFarAway;j++){
			
			if(atomVector[i].modelId == atomVector[j].modelId and i != j){
			
				real3 ri = atomVector[i].pos;
				real3 rj = atomVector[j].pos;
				real3 rij = rj - ri;
				double r = sqrt(dot(rij,rij));
                
                if(r < rCut_ENM){
					tooFarAway = false;
				}
				
			}
		}
		
		if(tooFarAway){
			atomVector.erase(atomVector.begin()+i);
            removedParticles_counter++;
            i--;
		}
		
	}
    
    ////////////////////////////////////////////////////////////////////
    
    std::ofstream enmFile(outputENMName);
    std::ofstream bondFile(outputGaussianName);
    
    int atomCount = 0;
    int ENM_counter = 0;
    int Gaussian_counter = 0;
	
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
    
    std::ofstream topFile(outputTopName);
    
    
    for(atomType&  atm : atomVector){
        topFile << std::right
                << std::fixed
                << std::setprecision(4)
                << std::setw(10)                   
                << atm.serial                       << " " 
                << std::setw(5)                    
                << atm.resName                      << " " 
                << std::setw(5)                    
                << -1*atm.modelId                   << " " 
                << std::setw(10)                   
                << atm.mass                         << " " 
                << std::setw(10)                   
                << atm.radius                       << " " 
                << std::setw(10)
                << getCharge(atm.resName)           << " " 
                << std::setw(10)
                << atm.pos                          << std::endl;
    }
    
    std::ofstream spFile(outputSpName);
    
    for(atomType&  atm : atomVector){
        spFile  << std::right
                << std::fixed
                << std::setprecision(4)
                << std::setw(10)
                << atm.pos                          << " "
                << std::setw(10)                   
                << atm.radius                       << " " 
                << std::setw(5)                    
                << -1*atm.modelId                   << std::endl;
    }
    
    ////////////////////////////////////////////////////////////////////
    
    std::cout << "Removed particles: " << removedParticles_counter << std::endl;
    
    int status;
    std::stringstream ss;
    
    ss.clear();
    ss.str(std::string());
    ss << "sed -i \'1s/^/" << Gaussian_counter <<"\\n/\' "<< outputGaussianName;
    std::cout << ss.str() << std::endl;
    status = std::system(ss.str().c_str());
    
    ////////////////////////////////////////////////////////////////////
    ///////////////////////////////CARGO////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    if(cargo){
    
        STRUCTURE pdbInputCargo;
        STRUCTURE pdbOutputCargo;
        
        pdbInputCargo.loadPDB(inputFileCargoName);
        pdbInputCargo.renumber();
        
        ////////////////////////////////////////////////////////////////////
        
        #ifdef CG    
        cg.applyCoarseGrainedMap<proteinManager::coarseGrainedManager::coarseGrainedMappingSchemes::ca>(pdbInputCargo,pdbOutputCargo);
        #endif
        
        ////////////////////////////////////////////////////////////////////
        
        for(MODEL&   mdl : pdbOutputCargo.model()){
        for(CHAIN&   ch  : mdl.chain()  ){
        for(RESIDUE& res : ch.residue() ){
        for(ATOM&    atm : res.atom()   ){
            atm.setAtomName(res.getResName());
        }}}}
        
        ////////////////////////////////////////////////////////////////////
        
        geometricTransformations::uniformScaling(pdbOutputCargo,0.1);
        
        ////////////////////////////////////////////////////////////////////
        
        rM.applyRadiusData(pdbOutputCargo);
        
        mM.applyMassesData(pdbOutputCargo);
        
        ////////////////////////////////////////////////////////////////////
        
        std::vector<atomType> atomVectorCargo;
        
        for(MODEL&   mdl : pdbOutputCargo.model()){
        for(CHAIN&   ch  : mdl.chain()  ){
        for(RESIDUE& res : ch.residue() ){
        for(ATOM&    atm : res.atom()   ){
            
            aTBuffer.serial  = atm.getAtomSerial();
            aTBuffer.modelId = atm.getModelId();
            aTBuffer.resName = atm.getResName();
            aTBuffer.mass    = atm.getAtomMass();
            aTBuffer.pos     = atm.getAtomCoord();
            aTBuffer.radius  = atm.getAtomRadius();
            
            atomVectorCargo.push_back(aTBuffer);
        }}}}
        
        for(atomType&  atm : atomVectorCargo){
            atm.serial = atomCount;
            atomCount ++;
        }
        
        for(int i = 0;    i<atomVectorCargo.size();i++){
            
            std::cout << "Generating ENM (cargo) " << i+1 <<"/"<<atomVectorCargo.size() << std::endl;
            
            for(int j = i +1; j<atomVectorCargo.size();j++){
                
                real3 ri = atomVectorCargo[i].pos;
                real3 rj = atomVectorCargo[j].pos;
                real3 rij = rj - ri;
                double r = sqrt(dot(rij,rij));
                
                if(atomVectorCargo[i].modelId == atomVectorCargo[j].modelId){
                    
                    if(r < rCut_ENM){
                        
                        enmFile << std::setw(10) << atomVectorCargo[i].serial <<
                                   std::setw(10) << atomVectorCargo[j].serial <<
                                   std::setprecision(6)                  <<
                                   std::setw(12) << r                    << std::endl;
                                   
                        ENM_counter++;
                        
                    }
                    
                }
            }
        }
        
        for(atomType&  atm : atomVectorCargo){
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
                    << getCharge(atm.resName)           << " " 
                    << std::setw(10)
                    << atm.pos                          << std::endl;
            
            atomCount ++;
        }
        
        for(atomType&  atm : atomVectorCargo){
            spFile  << std::right
                    << std::fixed
                    << std::setprecision(4)
                    << std::setw(10)
                    << atm.pos                          << " "
                    << std::setw(10)                   
                    << atm.radius                       << " " 
                    << std::setw(5)                    
                    << atm.modelId                      << std::endl;
        }
    
    }
    
    ss.clear();
    ss.str(std::string());
    ss << "sed -i \'1s/^/" << ENM_counter <<"\\n/\' "<< outputENMName;
    std::cout << ss.str() << std::endl;
    status = std::system(ss.str().c_str());
    
    
    return status;
    
}
