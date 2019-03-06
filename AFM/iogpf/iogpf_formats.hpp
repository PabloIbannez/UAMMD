#ifndef IOGPF_FORMATS
#define IOGPF_FORMATS

namespace uammd{
    
    namespace iogpf{
        
        class id{
        public:
            void operator()(std::shared_ptr<ParticleData> pd, int& index, std::stringstream& ss){
                int   _id;
                ss >> _id;
                auto idArray = pd->getId(access::location::cpu, access::mode::readwrite);
                idArray.raw()[index] = _id;
                
                #ifdef DEBUG
                std::cout << " id[" << index << "]: " << idArray.raw()[index];
                #endif
            }
        };
        
        class type{
        public:
            void operator()(std::shared_ptr<ParticleData> pd, int& index, std::stringstream& ss){
                real   _type;
                ss >> _type;
                auto typeArray = pd->getPos(access::location::cpu, access::mode::readwrite);
                typeArray.raw()[index].w = _type;
                
                #ifdef DEBUG
                std::cout << " type[" << index << "]: " << typeArray.raw()[index].w;
                #endif
            }
        };
        
        class molId{
        public:
            void operator()(std::shared_ptr<ParticleData> pd, int& index, std::stringstream& ss){
                int   _molId;
                ss >> _molId;
                auto molIdArray = pd->getMolId(access::location::cpu, access::mode::readwrite);
                molIdArray.raw()[index] = _molId;
                
                #ifdef DEBUG
                std::cout << " molId[" << index << "]: " << molIdArray.raw()[index];
                #endif
            }
        };
        
        /*
        class SASA{
        public:
            void operator()(std::shared_ptr<ParticleData> pd, int& index, std::stringstream& ss){
                real  _SASA;
                ss >> _SASA;
                
                auto SASA_Array = pd->getSASA(access::location::cpu, access::mode::readwrite);
                SASA_Array.raw()[index] = _SASA;
                
                #ifdef DEBUG
                std::cout << " SASA[" << index << "]: " << SASA_Array.raw()[index] ;
                #endif
            }
        };
        
        class surface{
        public:
            void operator()(std::shared_ptr<ParticleData> pd, int& index, std::stringstream& ss){
                int  _surface;
                ss >> _surface;
                
                auto surfaceArray = pd->getSurf(access::location::cpu, access::mode::readwrite);
                surfaceArray.raw()[index] = _surface;
                
                #ifdef DEBUG
                std::cout << " surface[" << index << "]: " << surfaceArray.raw()[index] ;
                #endif
            }
        };
        */
        class mass{
        public:
            void operator()(std::shared_ptr<ParticleData> pd, int& index, std::stringstream& ss){
                real _mass;
                ss >> _mass;
                
                auto massArray = pd->getMass(access::location::cpu, access::mode::readwrite);
                massArray.raw()[index] = _mass;
                
                #ifdef DEBUG
                std::cout << " mass[" << index << "]: " << massArray.raw()[index] ;
                #endif
            }
        };
        
        class radius{
        public:
            void operator()(std::shared_ptr<ParticleData> pd, int& index, std::stringstream& ss){
                real _radius;
                ss >> _radius;
                
                auto radiusArray = pd->getRadius(access::location::cpu, access::mode::readwrite);
                radiusArray.raw()[index] = _radius;
                
                #ifdef DEBUG
                std::cout << " radius[" << index << "]: " << radiusArray.raw()[index] ;
                #endif
            }
        };
        
        class planeRadius{
        public:
            void operator()(std::shared_ptr<ParticleData> pd, int& index, std::stringstream& ss){
                real _planeRadius;
                ss >> _planeRadius;
                
                auto planeRadiusArray = pd->getPlaneRadius(access::location::cpu, access::mode::readwrite);
                planeRadiusArray.raw()[index] = _planeRadius;
                
                #ifdef DEBUG
                std::cout << " planeRadius[" << index << "]: " << planeRadiusArray.raw()[index] ;
                #endif
            }
        };
        
        class pos{
        public:
            void operator()(std::shared_ptr<ParticleData> pd, int& index, std::stringstream& ss){
                real3 _pos;
                ss >> _pos.x;
                ss >> _pos.y;
                ss >> _pos.z;
                auto posArray = pd->getPos(access::location::cpu, access::mode::readwrite);
                posArray.raw()[index].x = _pos.x;
                posArray.raw()[index].y = _pos.y;
                posArray.raw()[index].z = _pos.z;
                
                #ifdef DEBUG
                std::cout << " pos[" << index << "]: " << posArray.raw()[index].x << " " << posArray.raw()[index].y << " " << posArray.raw()[index].z ;
                #endif
            }
        };
        /*
        class c12{
        public:
            void operator()(std::shared_ptr<ParticleData> pd, int& index, std::stringstream& ss){
                real  _c12;
                ss >> _c12;
                
                auto c12Array = pd->getC12(access::location::cpu, access::mode::readwrite);
                c12Array.raw()[index] = _c12;
                
                #ifdef DEBUG
                std::cout << " c12[" << index << "]: " << c12Array.raw()[index] ;
                #endif
            }
        };
        
        class c6{
        public:
            void operator()(std::shared_ptr<ParticleData> pd, int& index, std::stringstream& ss){
                real  _c6;
                ss >> _c6;
                
                auto c6Array = pd->getC6(access::location::cpu, access::mode::readwrite);
                c6Array.raw()[index] = _c6;
                
                #ifdef DEBUG
                std::cout << " c6[" << index << "]: " << c6Array.raw()[index] ;
                #endif
            }
        };
        
        class charge{
        public:
            void operator()(std::shared_ptr<ParticleData> pd, int& index, std::stringstream& ss){
                real  _chg;
                ss >> _chg;
                
                auto chgArray = pd->getCharge(access::location::cpu, access::mode::readwrite);
                chgArray.raw()[index] = _chg;
                
                #ifdef DEBUG
                std::cout << " chg[" << index << "]: " << chgArray.raw()[index] ;
                #endif
            }
        };
        
        class solvE{
        public:
            void operator()(std::shared_ptr<ParticleData> pd, int& index, std::stringstream& ss){
                real  _solvE;
                ss >> _solvE;
                
                auto solvE_Array = pd->getSolvE(access::location::cpu, access::mode::readwrite);
                solvE_Array.raw()[index] = _solvE;
                
                #ifdef DEBUG
                std::cout << " solvE[" << index << "]: " << solvE_Array.raw()[index] ;
                #endif
            }
        };
        */
    }
}

#endif
