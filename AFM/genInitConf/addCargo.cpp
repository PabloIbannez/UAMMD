#include <common.h>

bool moveToStartOfLine(std::ifstream& fs)
{
    fs.seekg(-1, std::ios_base::cur);
    for(long i = fs.tellg(); i > 0; i--)
    {
        if(fs.peek() == '\n')
        {
            fs.get();
            return true;
        }
        fs.seekg(i, std::ios_base::beg);
    }
    return false;
}

std::string getLastLineInFile(std::ifstream& fs)
{
    // Go to the last character before EOF
    fs.seekg(-1, std::ios_base::end);
    if (!moveToStartOfLine(fs))
        return "";

    std::string lastline = "";
    getline(fs, lastline);
    return lastline;
}

real rCut_ENM = 1;
real rCut_BOND = 1.5;

int main(int argc, char *argv[]){
    
    std::string inputCapsid;
    std::string inputFileNameCargo;
    std::string outputName;
    
    if(argc == 4){
        inputCapsid = argv[1];
        inputFileNameCargo  = argv[2];
        outputName          = argv[3];
    } else {
        std::cerr << "Input format error" << std::endl;
        std::cerr << "Format: input (cargo).pdb output" << std::endl;
        
        return EXIT_FAILURE;
    }
    
    std::string inputTopName       = inputCapsid + std::string(".top");
    std::string inputGaussianName  = inputCapsid + std::string(".enm");
    std::string inputENMName       = inputCapsid + std::string(".enm");
    std::string inputSpName        = inputCapsid + std::string(".sp");
    
    std::string outputTopName      = outputName + std::string(".top");
    std::string outputGaussianName = outputName + std::string(".enm");
    std::string outputENMName      = outputName + std::string(".enm");
    std::string outputSpName       = outputName + std::string(".sp");
    
    ////////////////////////////////////////////////////////////////////
    
    std::ifstream inputTopFile(inputTopName);
    std::ifstream inputBondFile(inputGaussianName);
    std::ifstream inputENMFile(inputENMName);
    std::ifstream inputSpFile(inputSpName);
    
    std::ofstream topFile(outputTopName);
    std::ofstream bondFile(outputGaussianName);
    std::ofstream enmFile(outputENMName);
    std::ofstream spFile(outputSpName);
    
    ////////////////////////////////////////////////////////////////////
    //Copy files
    topFile  << inputTopFile.rdbuf();
    bondFile << inputBondFile.rdbuf();
    enmFile  << inputENMFile.rdbuf();
    spFile   << inputSpFile.rdbuf();

    ////////////////////////////////////////////////////////////////////
    
    int atomCount = 0;
    int modelOffset = 0;
    int ENM_counter = 0;
    
    
    std::string line;
    std::string stringBuffer;
    std::stringstream ss;
    
    ////////////////////////////////////////////////////////////////////
    
    //Get atomCount adn ENM_counter from capid files.
    
    inputTopFile.clear();
    ss.str(getLastLineInFile(inputTopFile));
    ss >> atomCount >> stringBuffer >> modelOffset;
    
    if(modelOffset < 0) { modelOffset = -modelOffset;}
    
    atomCount++;
    
    ////////////////////////////////////////////////////////////////////
    
    inputENMFile.clear();
    inputENMFile.seekg (0, std::ios::beg);
    std::getline(inputENMFile,line);
    ss.str(line);
    ss >> ENM_counter;
    
    std::cout << "AtomCount: "    << atomCount   << 
                 " ,ModelOffset: " << modelOffset << 
                 " ,ENM_counter: " << ENM_counter << std::endl;
    
    ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    STRUCTURE pdbInputCargo;
    STRUCTURE pdbOutputCargo;
    
    pdbInputCargo.loadPDB(inputFileNameCargo);
    pdbInputCargo.renumber();
    
    ////////////////////////////////////////////////////////////////////
    
    coarseGrainedManager::coarseGrainedGenerator cg;
    cg.loadCGmodel("../RES2BEAD_noH/aminoAcid2bead_RES2BEAD_noH.map","../RES2BEAD_noH/bead2atom_RES2BEAD_noH.map");
      
    //cg.applyCoarseGrainedMap<proteinManager::coarseGrainedManager::coarseGrainedMappingSchemes::ca>(pdbInputCargo,pdbOutputCargo);
    cg.applyCoarseGrainedMap<caSASA>(pdbInputCargo,pdbOutputCargo);
    
    ////////////////////////////////////////////////////////////////////
    
    geometricTransformations::uniformScaling(pdbOutputCargo,0.1);
    
    //real3 centerCargo = computeCentroid(pdbOutputCargo);
    //geometricTransformations::rotation(pdbOutputCargo,centerCargo,{1,0,0},66.0*(M_PI/180.0));
    
    ////////////////////////////////////////////////////////////////////
    
    for(MODEL&   mdl : pdbOutputCargo.model()){
    for(CHAIN&   ch  : mdl.chain()  ){
    for(RESIDUE& res : ch.residue() ){
    for(ATOM&    atm : res.atom()   ){
        atm.setAtomName(res.getResName());
    }}}}
    
    ////////////////////////////////////////////////////////////////////
    
    radiusManager rM;
    
    rM.loadRadiusData("../aminoacidsRadius.dat");
    rM.applyRadiusData(pdbOutputCargo);
    
    massesManager mM;
    
    mM.loadMassesData("../aminoacidsMasses.dat");
    mM.applyMassesData(pdbOutputCargo);
    
    SASArandomCoil SRC("../SASArandomCoil.dat");
    
    ////////////////////////////////////////////////////////////////////
    
    std::vector<atomType> atomVectorCargo;
    
    atomType aTBuffer;
    
    for(MODEL&   mdl : pdbOutputCargo.model()){
    for(CHAIN&   ch  : mdl.chain()  ){
    for(RESIDUE& res : ch.residue() ){
    for(ATOM&    atm : res.atom()   ){
        
        aTBuffer.serial  = atm.getAtomSerial();
        aTBuffer.modelId = atm.getModelId() + modelOffset;
        aTBuffer.resName = atm.getResName();
        aTBuffer.mass    = atm.getAtomMass();
        aTBuffer.pos     = atm.getAtomCoord();
        aTBuffer.radius  = atm.getAtomRadius();
        
        atomVectorCargo.push_back(aTBuffer);
        
    }}}}
    
    ////////////////////////////////////////////////////////////////////
    
    for(atomType&  atm : atomVectorCargo){
        atm.serial = atomCount;
        atomCount ++;
    }
    
    ////////////////////////////////////////////////////////////////////
    
    for(atomType&  atm : atomVectorCargo){
        
        real sasaRatio = atm.SASA/SRC.getSASArandomCoil(atm.resName); //Side chain !!!!
        if(sasaRatio>1.0) {sasaRatio = 1.0;}
        
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
                << sasaRatio                        << " " 
                << std::setw(10)
                << atm.pos                          << std::endl;
        
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
    
    ////////////////////////////////////////////////////////////////////
    
    int status;
    
    ss.clear();
    ss.str(std::string());
    ss << "sed -i \'1s/.*/" << ENM_counter <<"/\' "<< outputENMName;
    std::cout << ss.str() << std::endl;
    status = std::system(ss.str().c_str());
    
    return status;
}
