#pragma once

#include <cstdint>
#include <vector>

#include "engine.hpp"
#include "features.hpp"
#include "shogi.hpp"
#include "play.hpp"



namespace Train
{

/**
 * @struct TrainingData
 * @brief Contains the description on one board state for training.
 */
struct TrainingData
{
	int32_t features_current[40]; // current player features at this game state. 
	int32_t features_other[40];   // Other player features at this game state.
	int32_t num_current;          // Number of current player active features.
	int32_t num_other;            // Number of other player active features.
	int32_t score;                // Score of this state. 
	float game_result;            // 1.0 (current player win), 0.0 (lost), 0.5 (null).
};



/**
 * @brief Exract all active features from a given game state.
 * @param state The state to extract the features from.
 * @param current_player The currently playing player.
 * @param data A reference to the place where to store the features indexes.  
 */
inline void extract_active_features(const Shogi::State &state, Shogi::Player current_player, TrainingData &data)
{
	const int32_t king_sente = state.king_sq(Shogi::Player::SENTE);
	const int32_t king_gote  = state.king_sq(Shogi::Player::GOTE);
	data.num_current = 0;
	data.num_other = 0;

	// Lambda to put the features automatically in the right buffer.
	auto push_feature = [&](int32_t id_sente, int32_t id_gote)
	{
		if (current_player == Shogi::Player::SENTE) {
			data.features_current[data.num_current++] = id_sente;
			data.features_other[data.num_other++] = id_gote;
		} else {
			data.features_current[data.num_current++] = id_gote;
			data.features_other[data.num_other++] = id_sente;
		}
	};

	// Get the board pieces features.
	const Shogi::Piece *board = state.data();
	for (int32_t sq = 0; sq < 81; ++sq) {
		Shogi::Piece p = board[sq];
		if (p.piece() == Shogi::PieceType::UNKNOWN) continue;	
		int32_t f_sente = -1;
		int32_t f_gote = -1;

		// Specific case to get the other player king feature.
		if (!(p.piece() == Shogi::PieceType::K && p.player() == Shogi::Player::SENTE))
			f_sente = NNUE::get_board_feature_index(Shogi::Player::SENTE, king_sente, p, sq);
		if (!(p.piece() == Shogi::PieceType::K && p.player() == Shogi::Player::GOTE))
			f_gote = NNUE::get_board_feature_index(Shogi::Player::GOTE, king_gote, p, sq);

		// Push the features in the buffers.
		if (f_sente != -1 || f_gote != -1) push_feature(f_sente, f_gote);
	
	}

	// Get the hand pieces features for each player.
	for (int32_t player = 0; player < 2; ++player) {
		Shogi::Player owner = static_cast<Shogi::Player>(player);
		const auto &hand = state.hands(owner);
		
		// Get the number of each piece in hand.
		for (int32_t pt = Shogi::PieceType::R; pt <= Shogi::PieceType::P; ++pt) {
			int32_t count = hand[pt];
			if (count <= 0) continue;
			
			// Get the feature id and push it in the buffers.
			Shogi::PieceType p_type = static_cast<Shogi::PieceType>(pt);
			int32_t f_sente = NNUE::get_hand_feature_index(Shogi::Player::SENTE, king_sente, owner, p_type, count);
			int32_t f_gote  = NNUE::get_hand_feature_index(Shogi::Player::GOTE, king_gote, owner, p_type, count);
			push_feature(f_sente, f_gote);
			
		}
	}

}



/**
 * @brief Generate a whole game and dump its history in a binary file.
 * @param network The NNUE to use to play.
 * @param dataset ofstream to the dataset binary file where to store the history.
 * @return 1.0f if Sente wins, 0.5f for a draw, and 0.0f if Gote wins. 
 */
inline float generate_selfplay_game(const NNUE::NNUE &network, std::ofstream &dataset)
{
    Shogi::State state = Shogi::State::make_default();
	NNUE::Accumulator acc;
    NNUE::refresh_accumulator(network, acc, state);

	// Prepare the game history.
    struct HistoryEntry 
	{
        TrainingData data;
        Shogi::Player turn;
    };
    std::vector<HistoryEntry> game_history;
    game_history.reserve(256);

	// Generate the game.
    Shogi::Player current_player = Shogi::Player::SENTE;
    int32_t current_turn = 0;
    bool is_draw = false;
    Shogi::Player winner;
    while (current_turn < 256) {
        
		// Save the current state features.
		game_history.emplace_back();
		HistoryEntry &current_entry = game_history.back(); 
        extract_active_features(state, current_player, current_entry.data); 
		current_entry.turn = current_player;

		// Find the best move to do and save its score.
		// Use high temperature at the beginning for exploration.
        const double temp = (current_turn < 16) ? 30.0 : 0.1;
        int32_t best_move_score = 0;
        const Shogi::Move best_move = Engine::search_move(best_move_score, network, state, acc, 3, current_player, temp);
        current_entry.data.score = best_move_score;

        // Detect end of games.
		if (best_move.piece() == Shogi::PieceType::UNKNOWN) {
            winner = Engine::OPPONENT[current_player];
            break; 
        }

		// Play the move and update the accumulator.
        const bool is_king = best_move.piece() == Shogi::K;
        if (!is_king) NNUE::update_accumulator(network, acc, state, best_move);
        state(best_move);
        if (is_king) NNUE::refresh_accumulator(network, acc, state);

		// Pas to the next player turn.
        current_player = Engine::OPPONENT[current_player];
        ++current_turn;
    }

	// If the game is too long it is a draw.
    if (current_turn >= 256) is_draw = true;

	// Annotate all entries.
    for (auto &entry : game_history) {
        if (is_draw) entry.data.game_result = 0.5f;
        else if (winner == entry.turn) entry.data.game_result = 1.0f;
        else  entry.data.game_result = 0.0f;
    }

	// Write the history.
	dataset.write(
		reinterpret_cast<const char*>(game_history.data()), 
		game_history.size() * sizeof(TrainingData)
	);
	return game_history.size() ? game_history[0].data.game_result : 0.5f;

}

};
