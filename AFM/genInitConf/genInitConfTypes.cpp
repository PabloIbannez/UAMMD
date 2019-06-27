#include "common.h"

#define CG

real rCut_ENM = 1;
real rCut_BOND = 1.5;

int main(int argc, char *argv[]){
    
    std::stringstream ss;
    
    std::string inputFileName;
    std::string structureListFileName;
    std::string outputName;
    
    if(argc == 4){
        inputFileName         = argv[1];
        structureListFileName = argv[2];
        outputName            = argv[3];
    } else {
        std::cerr << "Input format error" << std::endl;
        std::cerr << "Format: (virus).pdb (structureList).list output" << std::endl;
        
        return EXIT_FAILURE;
    }
    
    std::string outputTopName      = outputName + std::string(".top");
    std::string outputSpName       = outputName + std::string(".sp");
    std::string outputENMName      = outputName + std::string(".enm");
    std::string outputGaussianName = outputName + std::string(".gaussian");
    
    ////////////////////////////////////////////////////////////////////
    
    std::ofstream topFile(outputTopName);
    std::ofstream enmFile(outputENMName);
    std::ofstream bondFile(outputGaussianName);
    std::ofstream spFile(outputSpName);
    
    ////////////////////////////////////////////////////////////////////

    STRUCTURE pdbInput;
    STRUCTURE pdbOutput;
    
    pdbInput.loadPDB(inputFileName);
    pdbInput.renumber();
    
    ////////////////////////////////////////////////////////////////////
    
    std::map<int,std::string> structureList;
    
    std::ifstream structureListFile(structureListFileName);
    
    {
        
        std::string line;
        
        int intBuffer;
        std::string stringBuffer;
    
        while(std::getline(structureListFile,line)) {
                    
            ss.str(line);
            
            ss >> intBuffer >> stringBuffer;
            
            //std::cout << intBuffer << " " << stringBuffer << std::endl;
            
            structureList[intBuffer] = stringBuffer;
        }
        
        /*
        for(auto str : structureList){
            std::cout << str.first << " " << str.second << std::endl;
        }*/
    
    }
    
    ////////////////////////////////////////////////////////////////////
    
    #ifdef CG
    coarseGrainedManager::coarseGrainedGenerator cg;
    cg.loadCGmodel("../RES2BEAD_noH/aminoAcid2bead_RES2BEAD_noH.map","../RES2BEAD_noH/bead2atom_RES2BEAD_noH.map");
    
    //cg.applyCoarseGrainedMap<proteinManager::coarseGrainedManager::coarseGrainedMappingSchemes::ca>(pdbInput,pdbOutput);
    cg.applyCoarseGrainedMap<caSASA>(pdbInput,pdbOutput);
    #endif
    
    ////////////////////////////////////////////////////////////////////
    
    geometricTransformations::uniformScaling(pdbOutput,0.1);
    
    real3 center = computeCentroid(pdbOutput);
    geometricTransformations::rotation(pdbOutput,center,{1,0,0},34.0*(M_PI/180.0));
    geometricTransformations::rotation(pdbOutput,center,{0,1,0},13.0*(M_PI/180.0));
    
    ////////////////////////////////////////////////////////////////////
    
    
    for(MODEL&   mdl : pdbOutput.model()){
    for(CHAIN&   ch  : mdl.chain()  ){
    for(RESIDUE& res : ch.residue() ){
    for(ATOM&    atm : res.atom()   ){
        atm.setAtomName(res.getResName());
    }}}}
    
    ////////////////////////////////////////////////////////////////////
    
    radiusManager rM;
    
    rM.loadRadiusData("../aminoacidsRadius.dat");
    rM.applyRadiusData(pdbOutput);
    
    massesManager mM;
    
    mM.loadMassesData("../aminoacidsMasses.dat");
    mM.applyMassesData(pdbOutput);
    
    SASArandomCoil SRC("../SASArandomCoil.dat");
    
    ////////////////////////////////////////////////////////////////////
    
    std::vector<atomType> atomVector;
    
    atomType aTBuffer;
    
    for(MODEL&   mdl : pdbOutput.model()){
    for(CHAIN&   ch  : mdl.chain()  ){
    for(RESIDUE& res : ch.residue() ){
    for(ATOM&    atm : res.atom()   ){
        
        aTBuffer.serial     = atm.getAtomSerial();
        
        if(structureList.count(atm.getModelId()) != 0){
            aTBuffer.structType = structureList[atm.getModelId()];
        }else{
            std::cerr << "No structure for model: " << atm.getModelId() << std::endl;
            std::exit(EXIT_FAILURE);
        }
        
        aTBuffer.modelId    = atm.getModelId();
        aTBuffer.chainId    = atm.getChainId();
        aTBuffer.resName    = atm.getResName();
        aTBuffer.name       = atm.getResName();
        aTBuffer.resSeq     = atm.getResSeq();
        aTBuffer.mass       = atm.getAtomMass();
        aTBuffer.pos        = atm.getAtomCoord();
        aTBuffer.radius     = atm.getAtomRadius();
        aTBuffer.SASA       = atm.getAtomSASA();
        
        atomVector.push_back(aTBuffer);
    }}}}
    
    ////////////////////////////////////////////////////////////////////
    
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
    
    int atomCount = 0;
    
    for(atomType&  atm : atomVector){
        atm.serial = atomCount;
        atomCount ++;
    }
    
    ////////////////////////////////////////////////////////////////////
    
    int ENM_counter = 0;
    int Gaussian_counter = 0;
    
    ////////////////////////////////////////////////////////////////////
    
    for(atomType&  atm : atomVector){
        
        real sasaRatio = atm.SASA/SRC.getSASArandomCoil(atm.resName); //Side chain !!!!
        if(sasaRatio>1.0) {sasaRatio = 1.0;}
        
        topFile << std::right
                << std::fixed
                << std::setprecision(4)
                << std::setw(10)                   
                << atm.serial                                    << " " 
                << std::setw(5)                                  
                << atm.resName                                   << " " 
                << std::setw(5)                                  
                << -1*atm.modelId                                << " " 
                << std::setw(10)                                 
                << atm.mass                                      << " " 
                << std::setw(10)                                 
                << atm.radius                                    << " " 
                << std::setw(10)                                 
                << getCharge(atm.resName)                        << " " 
                << std::setw(10)
                << sasaRatio                                     << " " 
                << std::setw(10)
                << atm.pos                                   << std::endl;
    }
    
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
    
    std::map<std::pair<std::string,std::string>, real > interactionMatrix;
    int errorCount = 0;
    
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
                
                bool found;
                real precision = 1000;
                
                if(r < rCut_BOND){
                    
                    bondFile << std::setw(10) << atomVector[i].serial <<
                                std::setw(10) << atomVector[j].serial <<
                                std::setprecision(6)                  <<
                                std::setw(12) << r                    << std::endl;
                    
                    Gaussian_counter ++;
                    
                    atomVector[i].name = atomVector[i].resName + atomVector[i].chainId + std::to_string(atomVector[i].modelId) + 
                                         std::to_string(atomVector[i].resSeq) +
                                         atomVector[i].structType;
                                         
                    atomVector[j].name = atomVector[j].resName + atomVector[j].chainId + std::to_string(atomVector[j].modelId) + 
                                         std::to_string(atomVector[j].resSeq) +
                                         atomVector[j].structType;
                                         
                    //std::cout << atomVector[i].name << " " << atomVector[j].name << std::endl;
                    
                    std::pair<std::string,std::string> currentPair(atomVector[i].name,atomVector[j].name);
                    
                    real currentDst = std::round(r*precision)/precision;
                    
                    if(interactionMatrix.count(currentPair) != 0){
                        if(interactionMatrix[currentPair] != currentDst){
                            std::cout << "Error " << atomVector[i].resSeq << " "
                                                  << atomVector[j].resSeq << " "
                                                  << atomVector[i].structType << " " 
                                                  << atomVector[j].structType << " " 
                                                  << interactionMatrix[currentPair] << " " << currentDst << std::endl;
                            std::cout << interactionMatrix.size() << std::endl;
                            errorCount++;
                            //std::exit(EXIT_FAILURE);
                        }
                    } else {
                        interactionMatrix[currentPair] = currentDst;
                    }
                }
                
            }
            
        }
    }
    
    std::ofstream matrixFile("matrix.dat");
    for(auto& p : interactionMatrix){
        matrixFile << p.first.first << " " << p.first.second << " " << p.second << std::endl;
    }
    
    ////////////////////////////////////////////////////////////////////
    
    std::cout << "Removed particles: " << removedParticles_counter << std::endl;
    
    int status;
    
    ss.clear();
    ss.str(std::string());
    ss << "sed -i \'1s/^/" << Gaussian_counter <<"\\n/\' "<< outputGaussianName;
    std::cout << ss.str() << std::endl;
    status = std::system(ss.str().c_str());
    
    ss.clear();
    ss.str(std::string());
    ss << "sed -i \'1s/^/" << ENM_counter <<"\\n/\' "<< outputENMName;
    std::cout << ss.str() << std::endl;
    status = std::system(ss.str().c_str());
    
    std::cout << interactionMatrix.size() << std::endl;
    std::cout << errorCount << std::endl;
    
    return status;
    
}
