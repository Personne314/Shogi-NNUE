#pragma once

#include <random>
#include <vector>

#include "shogi.hpp"
#include "nnue.hpp"
#include "features.hpp"
#include "engine.hpp"
#include "alpha_beta.hpp"



namespace Engine 
{

/**
 * @brief Search the best next move to play for a given player.
 * @param network The NNUE to use for evaluations.
 * @param state The current game state.
 * @param acc The current state accumulator.
 * @param depth The max search depth to use.
 * @param player The player that will execute the move.
 * @param temperature The temperature of the random noise when choosing a move.
 * @param out_score The output move score.
 * @return The best move found by the algorithm.
 */
inline Shogi::Move search_move(
	int32_t &out_score,
	const NNUE::NNUE &network,
	const Shogi::State &state,
	const NNUE::Accumulator &acc,
	int32_t depth,
	Shogi::Player player,
	double temperature = 0.0f
) {
	MoveList list;
	generate_moves(state, player, list);

	// Move evaluation and filtering.
	int32_t best_score = -200000;
	Shogi::Move best_move;
	uint32_t i = 0;
	while (i < list.size) {
		const Shogi::Move move = list.moves[i].move;
		const bool is_king = move.piece() == Shogi::K;

		// Play the move.
		NNUE::Accumulator next_acc = acc;
		Shogi::State next_state = state;
		if (!is_king) NNUE::update_accumulator(network, next_acc, next_state, move);
		next_state(move);
		if (is_king) NNUE::refresh_accumulator(network, next_acc, next_state);

		// Remove it if it is illegal.
		if (is_attacked(next_state, next_state.king_sq(player), OPPONENT[player])) {
			list.moves[i] = list.moves[--list.size];
			continue;
		}

		// Caculate and store the score if it is legal.
		int32_t score = -alpha_beta(
			network, 
			next_state, next_acc, 
			depth - 1, 
			-200000, 200000, 
			OPPONENT[player], move
		);
		list.moves[i].score = score;
		if (score > best_score) {
			best_score = score;
			best_move = move;
		}
		++i;
	}

	// If there is no move left the player lost.
	if (list.size == 0) {
		out_score = -200000;
		return Shogi::Move();
	}

	// Shortcut the temperature if it is null.
	if (temperature <= 0.0f) return best_move;

	// Apply the temperature.
	out_score = best_score;
	double temp = std::max(0.1, temperature); 
	std::vector<double> probabilities;
	probabilities.reserve(list.size);
	for (uint32_t j = 0; j < list.size; ++j) {
		probabilities.push_back(std::exp(static_cast<double>(list.moves[j].score - best_score) / temp));
	}

	// Get a random move.
	std::random_device rd; 
	std::mt19937 rng(rd());
	std::discrete_distribution<size_t> dist(probabilities.begin(), probabilities.end());
	return list.moves[dist(rng)].move;

}

};