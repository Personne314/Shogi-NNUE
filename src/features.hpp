#pragma once

#include <array>

#include "shogi.hpp"
#include "nnue.hpp"



namespace NNUE
{

/**
 * @brief Map a given piece to an index depending of the viewing player.
 * @param view_player The player from which point of view we are doing the calculus.
 * @param piece The piece on the board.
 * @return The index in [0,26]
 */
inline constexpr int32_t get_piece_index(Shogi::Player view_player, Shogi::Piece piece) noexcept
{
	const bool is_friend = (piece.player() == view_player);
	const int32_t p_type = piece.piece(); 
	if (is_friend) {
		if (p_type < Shogi::PieceType::K) return p_type - 1;
		else return p_type - 2;
	}
	return p_type + 12;
}

/**
 * @brief Compute the absolute feature index of one piece on the board.
 * @param view_player The player from which point of view we are doing the calculus.
 * @param king_sq The square of the king.
 * @param piece The piece on the board.
 * @param piece_sq The square of the piece.
 */
inline constexpr int32_t get_board_feature_index(
	Shogi::Player view_player, 
	int32_t king_sq, 
	Shogi::Piece piece, 
	int32_t piece_sq
) noexcept {
	const int32_t p_idx = get_piece_index(view_player, piece);
	return (king_sq * FEATURES_PER_KING) + (p_idx * BOARD_SQUARES) + piece_sq;
}



// Base offsets for in-hand pieces.
static constexpr int32_t HAND_OFFSET[8] = 
{
	[Shogi::PieceType::UNKNOWN] = 0, 
	[Shogi::PieceType::R]       = 36, // 36 to 37
	[Shogi::PieceType::B]       = 34, // 34 to 35
	[Shogi::PieceType::G]       = 30, // 30 to 33
	[Shogi::PieceType::S]       = 26, // 26 to 29
	[Shogi::PieceType::N]       = 22, // 22 to 25
	[Shogi::PieceType::L]       = 18, // 18 to 21
	[Shogi::PieceType::P]       = 0   //  0 to 17
};

/**
 * @brief Compute the absolute index of the feature for a given quantity of pieces in hand.
 * @param view_player The player from which point of view we are doing the calculus.
 * @param king_sq The square of the king.
 * @param hand_owner The player who possess the pieces.
 * @param piece The piece on the board.
 * @param count The number of pieces in hand (>0).
 */
inline constexpr int32_t get_hand_feature_index(Shogi::Player view_player, int32_t king_sq, Shogi::Player hand_owner, Shogi::PieceType piece, int32_t count) noexcept
{
	const int32_t base_offset = PIECE_STATES * BOARD_SQUARES; 
	const int32_t owner_offset = (view_player == hand_owner) ? 0 : 38;
	const int32_t piece_offset = HAND_OFFSET[piece];
	return (king_sq * FEATURES_PER_KING) + base_offset + owner_offset + piece_offset + (count - 1);
}



/**
 * @struct Accumulator
 * @brief Contains the whole accumulated output of the first linear layer.
 * @note Aligned as 64 for AVX2/AVX512.
 */
struct Accumulator
{
	alignas(64) std::array<int16_t, INPUT_LAYER_SIZE> sente;
	alignas(64) std::array<int16_t, INPUT_LAYER_SIZE> gote;
};

/**
 * @brief Differential update of the accumulator for one feature.
 * @param weights The weights of the first layer.
 * @param acc The accumulator to update.
 * @param view The player from which point of view we are doing the calculus.
 * @param feature_id The id of the feature to update.
 * @param add Wether to add or remove the feature form the accumulator.
 */
inline void update_feature(
	const int16_t weights[TOTAL_FEATURES][INPUT_LAYER_SIZE],
	Accumulator &acc,
	Shogi::Player view,
	int32_t feature_id,
	bool add
) noexcept {
	auto &target_acc = (view == Shogi::Player::SENTE) ? acc.sente : acc.gote;
	const int16_t *feature_weights = weights[feature_id];
	if (add) for (int32_t i = 0; i < INPUT_LAYER_SIZE; ++i) target_acc[i] += feature_weights[i];
	else     for (int32_t i = 0; i < INPUT_LAYER_SIZE; ++i) target_acc[i] -= feature_weights[i];
}



/**
 * @brief Recompute all the accumulator from scratch. Used upon king move or board initialization.
 * @param weights Weights of the first linear layer.
 * @param acc The accumulator to recompute.
 * @param state The current game state.
 */
inline void refresh_accumulator(
	const NNUE &network,
	Accumulator &acc,
	const Shogi::State &state
) noexcept {

	// Reset.
	std::copy(std::begin(network.in_bias), std::end(network.in_bias), acc.sente.begin());
	std::copy(std::begin(network.in_bias), std::end(network.in_bias), acc.gote.begin());

	// Get the king positions.
	const int32_t king_sente = state.king_sq(Shogi::Player::SENTE);
	const int32_t king_gote  = state.king_sq(Shogi::Player::GOTE);

	// Lambda to apply a piece change to both player features.
	auto apply_change = [&](Shogi::Piece piece, int32_t sq) {
		int32_t id_sente = get_board_feature_index(Shogi::Player::SENTE, king_sente, piece, sq);
		int32_t id_gote  = get_board_feature_index(Shogi::Player::GOTE,  king_gote,  piece, sq);
		update_feature(network.in_weights, acc, Shogi::Player::SENTE, id_sente, true);
		update_feature(network.in_weights, acc, Shogi::Player::GOTE,  id_gote,  true);
	};

	// Lambda to apply a hand piece change to both player features.
	auto apply_hand_change = [&](Shogi::Player owner, Shogi::PieceType p_type, int32_t count) {
		if (count == 0) return;
		int32_t idx_sente = get_hand_feature_index(Shogi::Player::SENTE, king_sente, owner, p_type, count);
		int32_t idx_gote  = get_hand_feature_index(Shogi::Player::GOTE,  king_gote,  owner, p_type, count);
		update_feature(network.in_weights, acc, Shogi::Player::SENTE, idx_sente, true);
		update_feature(network.in_weights, acc, Shogi::Player::GOTE,  idx_gote,  true);
	};

	// Add all board pieces.
	const Shogi::Piece* board = state.data();
	for (int32_t sq = 0; sq < 81; ++sq) {
		Shogi::Piece p = board[sq];
		if (p.piece() != Shogi::PieceType::UNKNOWN && p.piece() != Shogi::PieceType::K) {
			apply_change(p, sq);
		}
	}

	// Add all hand pieces.
	for (int player = 0; player < 2; ++player) {
		Shogi::Player p = static_cast<Shogi::Player>(player);
		const auto& hand = state.hands(p);
		for (int pt = Shogi::PieceType::R; pt <= Shogi::PieceType::P; ++pt) {
			apply_hand_change(p, static_cast<Shogi::PieceType>(pt), hand[pt]);
		}
	}

}

/**
 * @brief Differential update of an accumulator after a move.
 * @param weights Weights of the first linear layer.
 * @param acc The accumulator to recompute.
 * @param state The current game state.
 * @param move The move that have just been played.
 */
inline void update_accumulator(
	const NNUE &network,
	Accumulator &acc,
	const Shogi::State &state,
	Shogi::Move move
) noexcept {
	const Shogi::PieceType p_type = move.piece();
	if (p_type == Shogi::PieceType::K) return;

	const Shogi::Player player = move.player();
	const Shogi::Square from = move.from();
	const Shogi::Square to = move.to();
	const Shogi::Piece piece_obj(player, p_type);
	const int32_t king_sente = state.king_sq(Shogi::Player::SENTE);
	const int32_t king_gote  = state.king_sq(Shogi::Player::GOTE);

	// Lambda to apply a piece change to both player features.
	auto apply_change = [&](Shogi::Piece piece, int32_t sq, bool add) {
		int32_t idx_sente = get_board_feature_index(Shogi::Player::SENTE, king_sente, piece, sq);
		int32_t idx_gote  = get_board_feature_index(Shogi::Player::GOTE,  king_gote,  piece, sq);
		update_feature(network.in_weights, acc, Shogi::Player::SENTE, idx_sente, add);
		update_feature(network.in_weights, acc, Shogi::Player::GOTE,  idx_gote,  add);
	};

	// Lambda to apply a hand piece change to both player features.
	auto apply_hand_change = [&](Shogi::Player owner, Shogi::PieceType p_t, int32_t count, bool add) {
		if (count <= 0) return; // La feature 0 n'existe pas
		int32_t idx_sente = get_hand_feature_index(Shogi::Player::SENTE, king_sente, owner, p_t, count);
		int32_t idx_gote  = get_hand_feature_index(Shogi::Player::GOTE,  king_gote,  owner, p_t, count);
		update_feature(network.in_weights, acc, Shogi::Player::SENTE, idx_sente, add);
		update_feature(network.in_weights, acc, Shogi::Player::GOTE,  idx_gote,  add);
	};

	// Case of drops.
	if (from == Shogi::Square::HAND) {
		int32_t old_count = state.hands(player)[p_type];
		apply_hand_change(player, p_type, old_count, false);
		apply_hand_change(player, p_type, old_count - 1, true);
		apply_change(piece_obj, to, true);
		return;
	}

	// Normal moves on the board.
	const Shogi::Piece original_piece = state.data()[from];
	apply_change(original_piece, from, false);
	apply_change(piece_obj, to, true);

	// Captures.
	const Shogi::Piece captured_piece = state.data()[to];
	if (captured_piece.piece() != Shogi::PieceType::UNKNOWN) {
		apply_change(captured_piece, to, false);
		Shogi::PieceType base_captured = captured_piece.piece();
		if (base_captured > Shogi::PieceType::K) {
			base_captured = static_cast<Shogi::PieceType>(base_captured - 8); 
		}
		int32_t old_hand_count = state.hands(player)[base_captured];
		apply_hand_change(player, base_captured, old_hand_count, false);    // Enlever l'ancien compte
		apply_hand_change(player, base_captured, old_hand_count + 1, true); // Ajouter le nouveau compte
	}

}

}
