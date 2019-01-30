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
                
                std::cout << " id[" << index << "]: " << idArray.raw()[index];
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
                posArray.raw()[index] = make_real4(_pos);
                
                std::cout << " pos[" << index << "]: " << posArray.raw()[index].x << " " << posArray.raw()[index].y << " " << posArray.raw()[index].z ;
            }
        };
        
        class charge{
        public:
            void operator()(std::shared_ptr<ParticleData> pd, int& index, std::stringstream& ss){
                real  _chg;
                ss >> _chg;
                
                auto chgArray = pd->getCharge(access::location::cpu, access::mode::readwrite);
                chgArray.raw()[index] = _chg;
                
                std::cout << " chg[" << index << "]: " << chgArray.raw()[index] ;
            }
        };
        
    }
}

#endif
