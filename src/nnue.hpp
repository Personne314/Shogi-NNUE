#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <random>

#include "shogi.hpp"



namespace NNUE
{

// Layers dimensions.
static constexpr int32_t INPUT_LAYER_SIZE = 256;
static constexpr int32_t HIDDEN_SIZE_1 = 32;
static constexpr int32_t HIDDEN_SIZE_2 = 32;

// Number of pieces except our king.
static constexpr int32_t PIECE_STATES = 27; 

// Number of features for the hands of both players.
static constexpr int32_t HAND_STATES = 76; 

// Number of features per king.
static constexpr int32_t FEATURES_PER_KING = (PIECE_STATES * BOARD_SQUARES) + HAND_STATES; 

// Total number of features.
static constexpr int32_t TOTAL_FEATURES = BOARD_SQUARES * FEATURES_PER_KING;



/**
 * @struct NNUE
 * @brief Structure to store all NNUE parameters.
 */
struct NNUE
{
	alignas(64) int16_t in_weights[TOTAL_FEATURES][INPUT_LAYER_SIZE];
	alignas(64) int16_t in_bias[INPUT_LAYER_SIZE];

	alignas(64) int8_t fc1_weights[HIDDEN_SIZE_1][2*INPUT_LAYER_SIZE];
	alignas(64) int32_t fc1_bias[HIDDEN_SIZE_1];

	alignas(64) int8_t fc2_weights[HIDDEN_SIZE_2][HIDDEN_SIZE_1];
	alignas(64) int32_t fc2_bias[HIDDEN_SIZE_2];

	alignas(64) int8_t output_weights[HIDDEN_SIZE_2];
	alignas(64) int32_t output_bias;
};

/**
 * @struct NNUEFloat
 * @brief Structure to store all NNUE floating point parameters.
 */
struct NNUEFloat
{
	alignas(64) float in_weights[TOTAL_FEATURES][INPUT_LAYER_SIZE];
	alignas(64) float in_bias[INPUT_LAYER_SIZE];

	alignas(64) float fc1_weights[HIDDEN_SIZE_1][2 * INPUT_LAYER_SIZE];
	alignas(64) float fc1_bias[HIDDEN_SIZE_1];

	alignas(64) float fc2_weights[HIDDEN_SIZE_2][HIDDEN_SIZE_1];
	alignas(64) float fc2_bias[HIDDEN_SIZE_2];

	alignas(64) float output_weights[HIDDEN_SIZE_2];
	alignas(64) float output_bias;
};

/**
 * @brief Load a model from a binary file.
 * @tparam T The model type (NNUE or NNUEFloat).
 * @param model The model to load into.
 * @param path The path to the binary file.
 * @return true on error.
 */
template<typename T>
inline bool load_model(T &model, const std::string &path)
{
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open()) return true;
	file.read(reinterpret_cast<char*>(&model), sizeof(T));
	file.close();
	return !file.good();
}

/**
 * @brief Save a model to a binary file.
 * @tparam T The model type (NNUE or NNUEFloat).
 * @param model The model to save.
 * @param path The path to the binary file.
 * @return true on error.
 */
template<typename T>
inline bool save_model(const T &model, const std::string &path)
{
	std::ofstream file(path, std::ios::binary);
	if (!file.is_open()) return true;
	file.write(reinterpret_cast<const char*>(&model), sizeof(T));
	file.close();
	return !file.good();
}



/**
 * @brief Initialize the network with random weights.
 * @param nnue The network to initialize.
 */
inline void randomize_weights(NNUEFloat &nnue)
{
	std::mt19937 gen(std::random_device{}());

	// Small weigths to avoid saturating the accumulator from the start.
	std::normal_distribution<float> dist_in(0.0f, 1.0f / std::sqrt(static_cast<float>(TOTAL_FEATURES)));

	// Hidden layers distributions.
	std::normal_distribution<float> dist_fc1(0.0f, std::sqrt(2.0f / static_cast<float>(2 * INPUT_LAYER_SIZE)));
	std::normal_distribution<float> dist_fc2(0.0f, std::sqrt(2.0f / static_cast<float>(HIDDEN_SIZE_1)));
	std::normal_distribution<float> dist_out(0.0f, std::sqrt(2.0f / static_cast<float>(HIDDEN_SIZE_2)));

	// Input layer initialization.
	for (int32_t i = 0; i < TOTAL_FEATURES; ++i) {
		for (int32_t j = 0; j < INPUT_LAYER_SIZE; ++j) {
			nnue.in_weights[i][j] = dist_in(gen);
		}
	}
	std::fill(std::begin(nnue.in_bias), std::end(nnue.in_bias), 0.0f);

	// Hidden layer 1 initialization.
	for (int32_t i = 0; i < HIDDEN_SIZE_1; ++i) {
		for (int32_t j = 0; j < 2 * INPUT_LAYER_SIZE; ++j) {
			nnue.fc1_weights[i][j] = dist_fc1(gen);
		}
		nnue.fc1_bias[i] = 0.0f;
	}

	// Hidden layer 2 initialization.
	for (int32_t i = 0; i < HIDDEN_SIZE_2; ++i) {
		for (int32_t j = 0; j < HIDDEN_SIZE_1; ++j) {
			nnue.fc2_weights[i][j] = dist_fc2(gen);
		}
		nnue.fc2_bias[i] = 0.0f;
	}

	// Output layer initialization.
	for (int32_t j = 0; j < HIDDEN_SIZE_2; ++j) {
		nnue.output_weights[j] = dist_out(gen);
	}
	nnue.output_bias = 0.0f;
	
}

/**
 * @brief Reset NNUE weigths optimizer moment buffers to zero.
 * @param nnue The network to reset.
 */
inline void reset_weights(NNUEFloat &nnue)
{
    std::memset(&nnue, 0, sizeof(NNUEFloat));
}



/**
 * @brief Quantize floating point weights into their integer representation.
 * @param src The source floating point model.
 * @param dst The destination quantized model.
 */
inline void quantize(const NNUEFloat &src, NNUE &dst)
{

    // Helpers.
    auto q8  = [](float v, float s) -> int8_t 
	{
		return static_cast<int8_t> (std::clamp<int32_t>(std::lroundf(v * s), -128,   127  ));
	};
    auto q16 = [](float v, float s) -> int16_t 
	{
		return static_cast<int16_t>(std::clamp<int32_t>(std::lroundf(v * s), -32768, 32767));
	};
    auto q32 = [](float v, float s) -> int32_t
	{
		return static_cast<int32_t>(std::lroundf(v * s));
	};

    // Input layer (scale 127).
    for (int i = 0; i < TOTAL_FEATURES; ++i)
        for (int j = 0; j < INPUT_LAYER_SIZE; ++j)
            dst.in_weights[i][j] = q16(src.in_weights[i][j], 127.0f);
    for (int i = 0; i < INPUT_LAYER_SIZE; ++i)
        dst.in_bias[i] = q16(src.in_bias[i], 127.0f);

    // Hidden Layer 1 (scale 64).
    for (int i = 0; i < HIDDEN_SIZE_1; ++i) {
        for (int j = 0; j < 2 * INPUT_LAYER_SIZE; ++j) {
            dst.fc1_weights[i][j] = q8(src.fc1_weights[i][j], 64.0f);
		}
        dst.fc1_bias[i] = q32(src.fc1_bias[i], 64.0f * 127.0f);
    }

    // Hidden Layer 2 (scale 64).
    for (int i = 0; i < HIDDEN_SIZE_2; ++i) {
        for (int j = 0; j < HIDDEN_SIZE_1; ++j) {
            dst.fc2_weights[i][j] = q8(src.fc2_weights[i][j], 64.0f);
		}
        dst.fc2_bias[i] = q32(src.fc2_bias[i], 64.0f * 127.0f);
    }

    // Output Layer (scale 64).
    for (int j = 0; j < HIDDEN_SIZE_2; ++j) {
        dst.output_weights[j] = q8(src.output_weights[j], 64.0f);
	}
    dst.output_bias = q32(src.output_bias, 64.0f * 127.0f);

}

};