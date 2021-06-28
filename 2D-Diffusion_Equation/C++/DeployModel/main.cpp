#include <torch/script.h>
#include <memory>
#include <iostream>
int main(int argc, const char* argv[])
{
 if (argc != 2) {
    std::cerr << "usage: main <path-to-exported-script-module>\n";
    return -1;
  }

  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module module;
    module = torch::jit::load("/home/sunny/DeployModel/2D-Diffusion_PDE.pt");
    // Create a vector of inputs.
    double x;
    std::cout<<"Please enter the value of x ";
    std::cin>>x;
    double y;
    std::cout<<"Please enter the value of y ";
    
    std::cin>>y;
    double t;
    std::cout<<"Please enter the value of t ";
    std::cin>>t;
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::tensor({{x,y,t}}));
    // Execute the model and turn its output into a tensor.
    torch::Tensor output = module.forward((inputs)).toTensor();

    std::cout <<  "u(x,y,t) = " << output << std::endl;
    
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

}
