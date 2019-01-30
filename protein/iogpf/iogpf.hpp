/*Pablo Ibáñez Freire*/

/*
Auxiliar functions for reading/writing particle properties into/from UAMMD

Usage:
    
    uammd::iogpf::read< ...format ...>(sys,filePath)
    
    The function expects that each line is a particle definition (empty and comment (#) lines are allowed).
    The format of the particle definitions is specified in the template of the function "read" 
    by using the auxiliary classes found in "iogpf_formats.hpp". 

    For example, if the format of each particle is:
    
    id pos_x pos_and pos_z
    
    ,then we'd have to specify:
    
    iogpf::read<iogpf::id,iogpf::pos,iogpf::charge>read(sys,filePath);
    
New formats can be added in the file "iogpf_formats.hpp" taking care 
that the type of variable has been defined before in "ParticleData.cuh".

*/
#ifndef IOGPF
#define IOGPF

#include "uammd.cuh"
#include "iogpf_formats.hpp"

#include <iostream>
#include <fstream>
#include <string>

namespace uammd{
    
    namespace iogpf{
        
        template<class ...Functor>
        struct SeqMethod:public Functor...{
            public:
                void operator()(std::shared_ptr<ParticleData> pd, int& index, std::stringstream& ss){
                    //int _[sizeof...(Functor)] = { (Functor::operator()(pd,index,ss), 0)... };
                    int _[] = { (Functor::operator()(pd,index,ss), 0)... };
                    return void(_);
                }
        };
        
        template<typename ... Format>
        std::shared_ptr<ParticleData> read(std::shared_ptr<System> sys, std::string filePath){
            
            std::fstream inputFile(filePath);
            std::string line;
            
            
            //Establish particle number.
            //Each line is consider to be a particle unless it starts with the symbol # or it is empty.
            int N = 0;
            
            while (std::getline(inputFile, line)){
                
                if (line.empty() or 
                    line.find_first_not_of (' ') == line.npos or
                    line[line.find_first_not_of(' ')] == '#'){
                        //Not particle line
                } else {
                    N++;
                }
            }
            
            auto pd = std::make_shared<ParticleData>(N,sys);

            //Back to the file first position
            inputFile.clear();
            inputFile.seekg (0, std::ios::beg);
            
            //class allMethods:public SeqMethod<Format ...>{} aM;
            SeqMethod<Format ...> aM;
            std::stringstream ss;
            
            //Now each line is processed
            int i=0;
            while (std::getline(inputFile, line)){
                
                if (line.empty() or 
                    line.find_first_not_of (' ') == line.npos or
                    line[line.find_first_not_of(' ')] == '#'){
                        //Not particle line
                } else {
                    ss.clear();
                    ss.str(line);
                    aM(pd,i,ss);
                    
                    std::cout << std::endl;
                    
                    i++;
                }
            }
            
            
            
            return pd; 
        }
        
    }
    
    
    
}

#endif
