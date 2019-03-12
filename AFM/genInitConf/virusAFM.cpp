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
    std::string chainId;
    std::string resName;
    int resSeq;
    real mass;
    real3 pos;
	real radius;
};

int main(){
    
    real K = 2000;
    
    real rCut_ENM = 1;
    real rCut_BOND = 1.5;
    
    STRUCTURE pdbInput;
    STRUCTURE pdbBuffer;
    STRUCTURE pdbIgnored;
    STRUCTURE pdbOutput;
    
    //pdbInput.loadPDB("p22ClearSep.pdb");
    pdbInput.loadPDB("p22Test.pdb");
    pdbInput.renumber();
    
    ////////////////////////////////////////////////////////////////////
    
    geometricTransformations::uniformScaling(pdbInput,0.1);
    
    ////////////////////////////////////////////////////////////////////
    
    coarseGrainedManager::coarseGrainedGenerator cg;
    cg.loadCGmodel("./RES2BEAD_noH/aminoAcid2bead_RES2BEAD_noH.map","./RES2BEAD_noH/bead2atom_RES2BEAD_noH.map");
    
    cg.applyCoarseGrainedMap<proteinManager::coarseGrainedManager::coarseGrainedMappingSchemes::ca>(pdbInput,pdbBuffer);
    
    ////////////////////////////////////////////////////////////////////
    
    radiusManager rM;
    
    rM.loadRadiusData("mixedRadius.dat");
    rM.applyRadiusData(pdbBuffer);
    
    massesManager mM;
    
    mM.loadMassesData("mixedMasses.dat");
    mM.applyMassesData(pdbBuffer);
    
    ////////////////////////////////////////////////////////////////////
    
    //Buffer atom vector
    
    atomType aTBuffer;
    {
        std::vector<atomType> atomVectorIgnored;
        
        for(MODEL&   mdl : pdbBuffer.model()){
        for(CHAIN&   ch  : mdl.chain()  ){
            pdbIgnored.addChain(mdl.getModelId(),ch.getChainId());
        for(RESIDUE& res : ch.residue() ){
        for(ATOM&    atm : res.atom()   ){
            
            aTBuffer.serial  = atm.getAtomSerial();
            aTBuffer.modelId = atm.getModelId();
            aTBuffer.chainId = atm.getChainId();
            aTBuffer.resName = atm.getResName();
            aTBuffer.resSeq  = atm.getResSeq();
            aTBuffer.mass    = atm.getAtomMass();
            aTBuffer.pos     = atm.getAtomCoord();
            aTBuffer.radius  = atm.getAtomRadius();
            
            atomVectorIgnored.push_back(aTBuffer);
        }}}}
        
        for(int i = 0;    i<atomVectorIgnored.size();i++){
            
            std::cout << "Virus ignored " << i+1 <<"/"<<atomVectorIgnored.size() << std::endl;
            
            for(int j = 0; j<atomVectorIgnored.size();j++){
                
                if(atomVectorIgnored[i].modelId != atomVectorIgnored[j].modelId){
                    
                    real3 ri = atomVectorIgnored[i].pos;
                    real3 rj = atomVectorIgnored[j].pos;
                    real3 rij = rj - ri;
                    double r = sqrt(dot(rij,rij));
                    
                    double radius_i = atomVectorIgnored[i].radius;
                    double radius_j = atomVectorIgnored[j].radius;
                    
                    double diamEff   = (radius_i+radius_j);
                            diamEff *= 1.122462;
                    
                    if(r <= diamEff*1.01){
                        
                        //std::cout << std::endl << std::endl;
                        
                        //std::cout << atomVectorIgnored[i].modelId << " " <<
                        //             atomVectorIgnored[i].chainId << " " <<
                        //             atomVectorIgnored[i].resName << " " <<
                        //             atomVectorIgnored[i].resSeq  << std::endl;
                        
                        pdbIgnored.addResidue(atomVectorIgnored[i].modelId,
                                              atomVectorIgnored[i].chainId,
                                              atomVectorIgnored[i].resName,
                                              atomVectorIgnored[i].resSeq,"");
                                              
                        break;
                        
                        //std::cout << atomVectorIgnored[j].modelId << " " <<
                        //             atomVectorIgnored[j].chainId << " " <<
                        //             atomVectorIgnored[j].resName << " " <<
                        //             atomVectorIgnored[j].resSeq  << std::endl;
                                              
                        //pdbIgnored.addResidue(atomVectorIgnored[j].modelId,
                        //                      atomVectorIgnored[j].chainId,
                        //                      atomVectorIgnored[j].resName,
                        //                      atomVectorIgnored[j].resSeq,"");
                    }
                }
            }
        }
    }
    
    //for(MODEL&   mdl : pdbIgnored.model()){
    //for(CHAIN&   ch  : mdl.chain()  ){
    //for(RESIDUE& res : ch.residue() ){
    //    std::cout << mdl.getModelId() << " " << ch.getChainId() << " " << res.getResName() << std::endl;
    //}}}
    
    
    
    ////////////////////////////////////////////////////////////////////
    
    cg.applyCoarseGrainedMap_IgnoreList<proteinManager::coarseGrainedManager::coarseGrainedMappingSchemes::ca>(pdbInput,pdbOutput,pdbIgnored);
    
    rM.applyRadiusData(pdbOutput);
    mM.applyMassesData(pdbOutput);
    
    std::vector<atomType> atomVector;
    
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
    /*
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
                << atm.pos                          << std::endl;
        
        atomCount ++;
    }*/
    
    for(auto    atm : atomVector){
        topFile << std::right
                << std::fixed
                << std::setprecision(4)
                << std::setw(10)
                << atm.pos                          << " "
                << std::setw(10)                   
                << atm.radius                       << " " 
                << std::setw(5)                    
                << atm.modelId+1                      << std::endl;
        
        atomCount ++;
    }
    
    
    
    ////////////////////////////////////////////////////////////////////
    
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
