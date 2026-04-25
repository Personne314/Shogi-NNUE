#include <cstdlib>
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <cstdint>
#include <vector>
#include <iostream>
#include <memory>

#include "train.hpp"
#include "adam.hpp"

int32_t main()
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.empty()) {
		std::cerr << "No OpenCL platform found.\n";
		return 1;
	}

	std::vector<cl::Device> devices;
	platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
	if (devices.empty()) {
		std::cerr << "No OpenCL GPU found.\n";
		return 1;
	}

	cl::Device device = devices[0];
	cl::Context context(device);
	cl::CommandQueue queue(context, device);
	std::cout << "GPU: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

	std::vector<Train::TrainingData> dataset;
	Train::load_dataset("dataset.bin", dataset);

	auto master_net = std::make_unique<NNUE::NNUEFloat>();
	auto engine_net = std::make_unique<NNUE::NNUE>();
	if (NNUE::load_model(*master_net, "master.bin")) abort();

	const int32_t BATCH_SIZE = 16;
	const int32_t EPOCHS = 100;
	const float LEARNING_RATE = 0.001f;

	Train::Adam optimizer(context, device, queue, *master_net);
	Train::Trainer trainer(context, device, queue, BATCH_SIZE);
	std::mt19937 rng(1337);
	
	std::vector<int32_t> indices(dataset.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::vector<Train::TrainingData> current_batch(BATCH_SIZE);
	for (int32_t epoch = 1; epoch <= EPOCHS; ++epoch) {
		std::cout << "--- Epoch " << epoch << "/" << EPOCHS << " ---\n";
		float epoch_loss = 0.0f;
		std::shuffle(indices.begin(), indices.end(), rng);
		int32_t batches = dataset.size() / BATCH_SIZE;
		for (int32_t b = 0; b < batches; ++b) {
			int32_t offset = b * BATCH_SIZE;
			for (int32_t i = 0; i < BATCH_SIZE; ++i) {
				current_batch[i] = dataset[indices[offset + i]];
			}
			float batch_loss = trainer.train_batch(optimizer, current_batch.data(), BATCH_SIZE, 0.5f, 400.0f);
			epoch_loss += batch_loss;
			optimizer.step(LEARNING_RATE);
			printf("\r%d / %d batches", b, batches);
			fflush(stdout);
		}
		printf("\n");

		if (batches > 0) epoch_loss /= batches;
		std::cout << "Epoch " << epoch << " Loss: " << epoch_loss << std::endl;

		// ==========================================================
		// SAUVEGARDE DE FIN D'ÉPOQUE
		// ==========================================================
		
		// 1. On rapatrie les poids du GPU vers le CPU
		optimizer.download_weights(*master_net);
		
		// 2. On effectue la quantification pour le moteur
		NNUE::quantize(*master_net, *engine_net);

		// 3. On écrase les backups "live"
		NNUE::save_model(*master_net, "dump_master.bin");
		NNUE::save_model(*engine_net, "dump_eval.bin");

		// 4. (Optionnel) On garde un historique tous les 10 epochs
		if (epoch % 10 == 0) {
			NNUE::save_model(*master_net, "checkpoint_float_epoch_" + std::to_string(epoch) + ".bin");
		}
	}
	
	return 0;
}











// #include <iostream>
// #include <memory>
// #include "nnue.hpp"

// // RANDOM NETWORK GENERATION
// int32_t main()
// {
// 	auto master_net = std::make_unique<NNUE::NNUEFloat>();
// 	auto engine_net = std::make_unique<NNUE::NNUE>();
// 	NNUE::randomize_weights(*master_net);
// 	NNUE::quantize(*master_net, *engine_net);
// 	if (NNUE::save_model(*master_net, "master.bin") || 
// 		NNUE::save_model(*engine_net, "eval.bin")) {
// 		std::cerr << "Failed to save." << std::endl;
// 		return 1;
// 	}
// 	std::cout << "Network ready." << std::endl;
// 	return 0;
// }







// #include <iostream>
// #include <string>
// #include "nnue.hpp"
// #include "train.hpp"

// // DATASET GENERATION.
// int32_t main()
// {
// 	std::string net_path = "eval.bin";
// 	std::string data_path = "dataset.bin";
// 	int32_t nb_games = 1024;
// 	auto network = std::make_unique<NNUE::NNUE>();
// 	if (NNUE::load_model(*network, net_path)) {
// 		std::cerr << "Error: Failed to load network " << net_path << std::endl;
// 		return 1;
// 	}
// 	Train::add_training_data(data_path, *network, nb_games);
// 	return 0;
// }
