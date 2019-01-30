#include "../iogpf.hpp"

using namespace uammd;
using namespace std;

int main(){
    
  auto sys = make_shared<System>();
  
  auto pd = iogpf::read<iogpf::id,iogpf::pos,iogpf::charge>(sys,"dataTest.dat");
  
  sys->finish();
}
