// #include "csa.h"
// #include "shogi.hpp"

// #include <cstdint>
// #include <iostream>




// int32_t main() 
// {
// 	CSARecord record("csa.txt");
// 	std::cout << record.init().str() << std::endl;
// 	for (auto &m : record.moves()) std::cout << m.str() << std::endl;
// }



#include <iostream>
#include <vector>
#include <string>

// Header C++ officiel d'OpenCL (souvent installé via opencl-headers)
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>

int main()
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	if (platforms.empty()) {
		std::cerr << "Aucune plateforme OpenCL trouvée sur ce système.\n";
		return 1;
	}

	std::cout << "Plateformes OpenCL trouvées : " << platforms.size() << "\n\n";

	for (const auto& platform : platforms) {
		std::string platform_name;
		platform.getInfo(CL_PLATFORM_NAME, &platform_name);
		std::cout << "--- Plateforme : " << platform_name << " ---\n";

		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

		if (devices.empty()) std::cout << "\tAucun périphérique trouvé.\n";

		for (const auto& device : devices) {
			std::string device_name;
			device.getInfo(CL_DEVICE_NAME, &device_name);
			
			std::string device_version;
			device.getInfo(CL_DEVICE_VERSION, &device_version);

			std::cout << "\tPériphérique : " << device_name << "\n";
			std::cout << "\tVersion      : " << device_version << "\n\n";
		}
	}

	return 0;
}
