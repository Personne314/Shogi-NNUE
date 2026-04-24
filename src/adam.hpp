#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include "nnue.hpp"


namespace Train
{

/**
 * @class Optimizer
 * @brief GPU-side Adam optimizer managing all NNUE weights and moments.
 */
class Adam
{
private:

	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;
	cl::Kernel adam_kernel;

	cl::Buffer d_weights;
	cl::Buffer d_gradients;
	cl::Buffer d_m;
	cl::Buffer d_v;

	int32_t total_params;
	int32_t timestep;

public:

	Adam(cl::Context ctx, cl::Device device, cl::CommandQueue q, NNUE::NNUEFloat &initial_weights)
	{
		context = ctx;
		queue = q;
		total_params = sizeof(NNUE::NNUEFloat) / sizeof(float);
		timestep = 1;

		// Adam kernel compilation.
		std::ifstream file("adam.cl");
		if (!file.is_open()) {
			std::cerr << "Error: Failed to open 'adam.cl'\n";
			exit(1);
		}
		std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		cl::Program::Sources sources;
		sources.push_back({source.c_str(), source.length()});
		program = cl::Program(context, sources);
		if (program.build({device}) != CL_SUCCESS) {
			std::cerr << "Error:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
			exit(1);
		}
		adam_kernel = cl::Kernel(program, "adam_update");

		// VRAM allocation (~187MB per buffer so a total of ~750MB).
		d_weights = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(NNUE::NNUEFloat), &initial_weights);
		d_gradients = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(NNUE::NNUEFloat));
		d_m = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(NNUE::NNUEFloat));
		d_v = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(NNUE::NNUEFloat));
		
		// Moments and gradients are set to zero.
		float zero = 0.0f;
		queue.enqueueFillBuffer(d_gradients, zero, 0, sizeof(NNUE::NNUEFloat));
		queue.enqueueFillBuffer(d_m, zero, 0, sizeof(NNUE::NNUEFloat));
		queue.enqueueFillBuffer(d_v, zero, 0, sizeof(NNUE::NNUEFloat));
		queue.finish();

	}

	void step(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
	{
		adam_kernel.setArg(0, d_weights);
		adam_kernel.setArg(1, d_gradients);
		adam_kernel.setArg(2, d_m);
		adam_kernel.setArg(3, d_v);
		adam_kernel.setArg(4, lr);
		adam_kernel.setArg(5, beta1);
		adam_kernel.setArg(6, beta2);
		adam_kernel.setArg(7, eps);
		adam_kernel.setArg(8, timestep);
		adam_kernel.setArg(9, total_params);

		// Launch the optimization kernel.
		cl::NDRange global_work_size(total_params);
		queue.enqueueNDRangeKernel(adam_kernel, cl::NullRange, global_work_size, cl::NullRange);
		++timestep;
	}

	// Get the weights from the GPU.
	void download_weights(NNUE::NNUEFloat &host_weights)
	{
		queue.enqueueReadBuffer(d_weights, CL_TRUE, 0, sizeof(NNUE::NNUEFloat), &host_weights);
	}

	// Get the moments from the GPU.
	void download_moments(NNUE::NNUEFloat &m, NNUE::NNUEFloat &v)
	{
		queue.enqueueReadBuffer(d_m, CL_TRUE, 0, sizeof(NNUE::NNUEFloat), &m);
		queue.enqueueReadBuffer(d_v, CL_TRUE, 0, sizeof(NNUE::NNUEFloat), &v);
	}

	cl::Buffer &get_weights()
	{
		return d_weights;
	}
	
		cl::Buffer &get_gradients()
	{
		return d_gradients;
	}

};

};