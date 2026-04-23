#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

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
 * @brief Load the model from a binary file.
 * @param model The model where to load the file into.
 * @param path The path to the file where to load the model from.
 * @return true if there was an error.
 */
inline bool load_model(NNUE &model, const std::string &path) 
{
	// 1. Ouverture du fichier en mode binaire.
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Error: Failed to open the file : '" << path << "'\n";
		return true;
	}
	if (!file.read(reinterpret_cast<char*>(&model), sizeof(NNUE))) {
		std::cerr << "Error: Corrupted file or invalid size : '" << path << "'\n";
		return true;
	}
	return false;
}

/**
 * @brief Save the model in a binary file.
 * @param model The model to save.
 * @param path The path to the file where to save the model.
 * @return true if there was an error.
 */
inline bool save_model(const NNUE &model, const std::string &path) 
{
	std::ofstream file(path, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Error: Failed to create the file : '" << path << "'\n";
		return false;
	}
	file.write(reinterpret_cast<const char*>(&model), sizeof(NNUE));
	return !file.good();
}

};