#pragma once

#include <cstdint>

#include "features.hpp"
#include "shogi.hpp"
#include "engine.hpp"
#include "nnue.hpp"



namespace Engine
{

// Final scale. This is a bullshit value since the real one depends of the network to match centipawn.
static constexpr int32_t FINAL_SCALE = 16;

/**
 * @brief Evaluate the score of one configuration for one player using the NNUE.
 * @param nnue The network weights.
 * @param acc The current game accumulator.
 * @param player The current player.
 * @return The configuration score.
 */
inline int32_t evaluate(const NNUE::NNUE &nnue, const NNUE::Accumulator &acc, Shogi::Player player)
{
	alignas(64) std::array<int8_t, 2 * NNUE::INPUT_LAYER_SIZE> clipped_acc;

	// Concat the inputs depending of the current player.
	const auto &my_acc  = (player == Shogi::Player::SENTE) ? acc.sente : acc.gote;
	const auto &opp_acc = (player == Shogi::Player::SENTE) ? acc.gote  : acc.sente;

	// Apply clipped ReLU on the input layer.
	const int16_t * __restrict my_ptr = my_acc.data();
	const int16_t * __restrict opp_ptr = opp_acc.data();
	int8_t * __restrict clip_dst = clipped_acc.data();
	#pragma GCC ivdep
	for (size_t i = 0; i < NNUE::INPUT_LAYER_SIZE; ++i) {
		clip_dst[i] = std::clamp<int16_t>(my_ptr[i], 0, 127);
		clip_dst[i + NNUE::INPUT_LAYER_SIZE] = std::clamp<int16_t>(opp_ptr[i], 0, 127);
	}

	// First hidden layer (FC1 : 512 -> 32).
	alignas(64) std::array<int8_t, NNUE::HIDDEN_SIZE_1> fc1_out;
	for (size_t i = 0; i < NNUE::HIDDEN_SIZE_1; ++i) {
		int32_t sum = nnue.fc1_bias[i];
		const int8_t * __restrict w = nnue.fc1_weights[i];
		#pragma GCC ivdep
		for (size_t j = 0; j < 2 * NNUE::INPUT_LAYER_SIZE; ++j) sum += w[j] * clip_dst[j];
		
		// Divide by 64 for quantization then do a ReLU.
		fc1_out[i] = std::clamp<int32_t>(sum >> 6, 0, 127);
	}

	// Second hidden layer (FC2 : 32 -> 32).
	alignas(64) std::array<int8_t, NNUE::HIDDEN_SIZE_2> fc2_out;
	for (size_t i = 0; i < NNUE::HIDDEN_SIZE_2; ++i) {
		int32_t sum = nnue.fc2_bias[i];
		const int8_t * __restrict w = nnue.fc2_weights[i];
		#pragma GCC ivdep
		for (size_t j = 0; j < NNUE::HIDDEN_SIZE_1; ++j) sum += w[j] * fc1_out[j];
		
		// Divide by 64 for quantization then do a ReLU.
		fc2_out[i] = std::clamp<int32_t>(sum >> 6, 0, 127);
	}

	// Output layer (Output : 32 -> 1).
	int32_t score = nnue.output_bias;
	const int8_t * __restrict w = nnue.output_weights;
	#pragma GCC ivdep
	for (size_t j = 0; j < NNUE::HIDDEN_SIZE_2; ++j) score += w[j] * fc2_out[j];
	return score / FINAL_SCALE;

}

/**
 * @brief Alpha-beta implementation for the shogin engine.
 * @param state The current state of the board.
 * @param depth The current depth in the tree.
 * @param alpha The current alpha value.
 * @param beta The current beta value.
 * @param player The current player.
 * @param prev The last move done by the previous player.
 * @return The value of this node.
 */
inline int32_t alpha_beta(
	const NNUE::NNUE &network,
	const Shogi::State &state, 
	const NNUE::Accumulator &acc, 
	int32_t depth, 
	int32_t alpha, 
	int32_t beta, 
	Shogi::Player player, 
	Shogi::Move prev
) {
	if (depth == 0) {
		return evaluate(network, acc, player);
	}

	// Generate the possible moves at the current board configuration.
	Engine::MoveList list;
	Engine::generate_moves(state, player, list);
	list.sort();

	// Try the possible moves.
	int32_t best_score = -200000;
	int32_t legal_moves = 0;
	for (uint32_t i = 0; i < list.size; ++i) {
		const Shogi::Move move = list.moves[i].move;
		const bool is_king = move.piece() == Shogi::K;

		NNUE::Accumulator next_acc = acc;
		Shogi::State next_state = state;
		if (!is_king) NNUE::update_accumulator(network, next_acc, next_state, move);
		next_state(move);
		if (is_king) NNUE::refresh_accumulator(network, next_acc, next_state);

		// Check if the current player king is in check after this move.
		if (Engine::is_attacked(next_state, next_state.king_sq(player), Engine::OPPONENT[player])) continue;
		++legal_moves;

		// Recursive call with inverted alpha and beta.
		int32_t score = -alpha_beta(network, next_state, next_acc, depth - 1, -beta, -alpha, Engine::OPPONENT[player], move);
		if (score > best_score) best_score = score;

		// Alpha-beta pruning.
		if (score > alpha) alpha = score;
		if (alpha >= beta) break;
	}

	// Checkmate detection.
	if (legal_moves != 0) return best_score;
	if (Engine::is_attacked(state, state.king_sq(player), Engine::OPPONENT[player])) {
		if (prev.from() == Shogi::Square::HAND && prev.piece() == Shogi::PieceType::P) 
			return 100000 - depth; // Uchifume.
		return -100000 + depth;    // Mat.
	} else return -100000 + depth; // Pat.

}

};
