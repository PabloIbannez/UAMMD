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
    std::string name;
    std::string structType;
    int modelId;
    std::string chainId;
    std::string resName;
    int resSeq;
    real mass;
    real3 pos;
	real radius;
    real SASA;
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

struct caSASA{
                                    
    void mappingScheme(RESIDUE& resIn, RESIDUE& resOut, std::string const & beadName,std::vector<std::string>& beadComponents){
        
        ////////////////////////////////////////////////
        
        real3 pos = resIn.atom("CA").getAtomCoord();
        
        ////////////////////////////////////////////////
        
        real SASA = 0;
        
        for(auto& atm : resIn.atom()){
            
            std::string atmName = atm.getAtomName();
            
            if(atmName != "N"  and 
               atmName != "CA" and 
               atmName != "C"  and 
               atmName != "O"){  //Only side-chain atoms
                   
                SASA += atm.getAtomSASA()/100.0; //From A^2 to nm^2
            }
        }
        
        ////////////////////////////////////////////////
        
        resOut.atom(beadName).setAtomCoord(pos);
        resOut.atom(beadName).setAtomSASA(SASA);
        
        //Common properties
        resOut.atom(beadName).setAtomAltLoc(" ");
        resOut.atom(beadName).setAtomOccupancy(1);
        resOut.atom(beadName).setAtomTempFactor(0);
        resOut.atom(beadName).setAtomElement("");
    }
                                    
};


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

class SASArandomCoil{
    
    real defaultValue = INFINITY;
    
    std::map<std::string,real> sasaMap;
    
    public:
        
        SASArandomCoil(std::string inputFileName){
            
            std::ifstream inputFile(inputFileName);
                    
            std::stringstream ss;
            std::string line;
            
            std::string resName_buffer;
            real SASA_buffer;
            
            while(std::getline(inputFile,line)) {
                
                ss.str(line);
                
                ss >> resName_buffer >> SASA_buffer;
                
                //std::cout << resName_buffer << " " << SASA_buffer << std::endl;
                
                sasaMap[resName_buffer] = SASA_buffer;
            }
        }
        
        real getSASArandomCoil(std::string& resName){
            if(sasaMap.count(resName)){
                return sasaMap[resName];
            } else {
                std::cerr << "The SASA value for the resiude " << resName << " has not been added. I return the default value: " << defaultValue  << std::endl;
                return defaultValue;
            }
        }
};
