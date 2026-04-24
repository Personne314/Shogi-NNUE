#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "shogi.hpp"



namespace Engine
{

// Constants.

#define MAX_MOVES 600

// Starting point of each ray for translating pieces.
static constexpr int RAY_STARTS[8] = 
{
	Shogi::Direction::U1,
	Shogi::Direction::D1,
	Shogi::Direction::L1,
	Shogi::Direction::R1,
	Shogi::Direction::UL1,
	Shogi::Direction::UR1,
	Shogi::Direction::DL1,
	Shogi::Direction::DR1
};

// Starting point of the opposite of each ray for attacing detection.
static constexpr int OPPOSITE_RAY_STARTS[8] = 
{
    Shogi::Direction::D1,
    Shogi::Direction::U1,
    Shogi::Direction::R1,
    Shogi::Direction::L1,
    Shogi::Direction::DR1,
    Shogi::Direction::DL1,
    Shogi::Direction::UR1,
    Shogi::Direction::UL1
};

// Offset corresponding to ray directions.
static constexpr int RAY_OFFSETS[8] = 
{ 
	-9, +9, +1, -1, -8, -10, +10, +8
};


// Get the given player opponent. 
static constexpr Shogi::Player OPPONENT[2] =
{
	[Shogi::Player::SENTE] = Shogi::Player::GOTE,
	[Shogi::Player::GOTE]  = Shogi::Player::SENTE
};


// Direction of the squares for the knight movements indexed by Player.
static constexpr uint8_t KNIGHT_DIRS[2][2] =
{
	[Shogi::Player::SENTE] = { Shogi::Direction::KUL, Shogi::Direction::KUR },
	[Shogi::Player::GOTE]  = { Shogi::Direction::KDL, Shogi::Direction::KDR },
};

// Offset of the squares for the knight movements indexed by Player.
static constexpr int32_t KNIGHT_OFFSETS[2][2] =
{
	[Shogi::Player::SENTE] = { -17, -19 },
	[Shogi::Player::GOTE]  = { +19, +17 },
};


// PieceType list indexed by PieceType to check the promoted value of one piece.
static constexpr Shogi::PieceType PROMOTED[15] = 
{
	[Shogi::PieceType::UNKNOWN] = Shogi::PieceType::UNKNOWN,
	[Shogi::PieceType::R]       = Shogi::PieceType::pR,
	[Shogi::PieceType::B]       = Shogi::PieceType::pB,
	[Shogi::PieceType::G]       = Shogi::PieceType::G,
	[Shogi::PieceType::S]       = Shogi::PieceType::pS,
	[Shogi::PieceType::N]       = Shogi::PieceType::pN,
	[Shogi::PieceType::L]       = Shogi::PieceType::pL,
	[Shogi::PieceType::P]       = Shogi::PieceType::pP,
	[Shogi::PieceType::K]       = Shogi::PieceType::UNKNOWN,
	[Shogi::PieceType::pR]      = Shogi::PieceType::UNKNOWN,
	[Shogi::PieceType::pB]      = Shogi::PieceType::UNKNOWN,
	[Shogi::PieceType::pS]      = Shogi::PieceType::UNKNOWN,
	[Shogi::PieceType::pN]      = Shogi::PieceType::UNKNOWN,
	[Shogi::PieceType::pL]      = Shogi::PieceType::UNKNOWN,
	[Shogi::PieceType::pP]      = Shogi::PieceType::UNKNOWN
};

// Boolean list indexed by PieceType to know which piece can be promoted.
static constexpr bool CAN_PROMOTE[15] = 
{
	[Shogi::PieceType::UNKNOWN] = false,
	[Shogi::PieceType::R]       = true,
	[Shogi::PieceType::B]       = true,
	[Shogi::PieceType::G]       = false,
	[Shogi::PieceType::S]       = true,
	[Shogi::PieceType::N]       = true,
	[Shogi::PieceType::L]       = true,
	[Shogi::PieceType::P]       = true,
	[Shogi::PieceType::K]       = false,
	[Shogi::PieceType::pR]      = false,
	[Shogi::PieceType::pB]      = false,
	[Shogi::PieceType::pS]      = false,
	[Shogi::PieceType::pN]      = false,
	[Shogi::PieceType::pL]      = false,
	[Shogi::PieceType::pP]      = false
};


// Mask for Sente promotion zone (Gote side).
static constexpr std::bitset<BOARD_SQUARES> SENTE_PROMO_ZONE =
	([]{ std::bitset<BOARD_SQUARES> b; for(int i=0;i<27;++i)  b.set(i); return b; })();

// Mask for Gote promotion zone (Sente side).
static constexpr std::bitset<BOARD_SQUARES> GOTE_PROMO_ZONE =
	([]{ std::bitset<BOARD_SQUARES> b; for(int i=54;i<BOARD_SQUARES;++i) b.set(i); return b; })();


// Mask defining forbidden layers for sente pawns and lances.
static constexpr std::bitset<BOARD_SQUARES> SENTE_PAWN_LANCE_FORBIDDEN =
	([]{ std::bitset<BOARD_SQUARES> b; for(int i=0;i<9;++i)  b.set(i); return b; })();

// Mask defining forbidden layers for gote pawns and lances.
static constexpr std::bitset<BOARD_SQUARES> GOTE_PAWN_LANCE_FORBIDDEN =
	([]{ std::bitset<BOARD_SQUARES> b; for(int i=72;i<BOARD_SQUARES;++i) b.set(i); return b; })();

// Mask defining forbidden layers for sente knights.
static constexpr std::bitset<BOARD_SQUARES> SENTE_KNIGHT_FORBIDDEN =
	([]{ std::bitset<BOARD_SQUARES> b; for(int i=0;i<18;++i)  b.set(i); return b; })();

// Mask defining forbidden layers for gote knights.
static constexpr std::bitset<BOARD_SQUARES> GOTE_KNIGHT_FORBIDDEN =
	([]{ std::bitset<BOARD_SQUARES> b; for(int i=63;i<BOARD_SQUARES;++i) b.set(i); return b; })();

// Mask to define the columns.
static constexpr std::array<std::bitset<BOARD_SQUARES>, 9> COL_MASKS = []()
{
	std::array<std::bitset<BOARD_SQUARES>, 9> masks{};
	for (int col = 0; col < 9; ++col) {
		for (int row = 0; row < 9; ++row) masks[col].set(row * 9 + col);
	}
	return masks;
}();


// Piece values LUT for move euristics.
static constexpr std::array<int32_t, 15> PIECE_VALUE = []()
{
	std::array<int32_t, 15> t{};
	t[Shogi::PieceType::P]  = 100;
	t[Shogi::PieceType::L]  = 300;
	t[Shogi::PieceType::N]  = 400;
	t[Shogi::PieceType::S]  = 500;
	t[Shogi::PieceType::G]  = 600;
	t[Shogi::PieceType::pP] = 600;
	t[Shogi::PieceType::pL] = 600;
	t[Shogi::PieceType::pN] = 600;
	t[Shogi::PieceType::pS] = 600;
	t[Shogi::PieceType::B]  = 800;
	t[Shogi::PieceType::R]  = 1000;
	t[Shogi::PieceType::pB] = 1100;
	t[Shogi::PieceType::pR] = 1300;
	t[Shogi::PieceType::K]  = 10000;
	return t;
}();

// Precomputed MVV-LVA LUT for move euristics. 
static constexpr std::array<std::array<int32_t, 15>, 15> MVV_LVA = []()
{
	std::array<std::array<int32_t, 15>, 15> t{};
	for (int atk = 0; atk < 15; ++atk) {
		for (int vic = 0; vic < 15; ++vic) {
			t[atk][vic] = (vic == Shogi::PieceType::UNKNOWN) ? 
				0 : 100000 + PIECE_VALUE[vic] * 10 - PIECE_VALUE[atk];
		}
	}
	return t;
}();


// Sente drop agressivity bonus.
constexpr std::array<int32_t, 9> DROP_ZONE_BONUS_SENTE =
{
	800, 700, 600, 500, 400, 300, 200, 100, 50
};

// Gote drop agressivity bonus.
constexpr std::array<int32_t, 9> DROP_ZONE_BONUS_GOTE =
{
	50, 100, 200, 300, 400, 500, 600, 700, 800
};



/**
 * @struct ScoredMove
 * @brief Contains one move and its score. 
 */
struct ScoredMove
{
	Shogi::Move move; // The movement.
	int32_t score;    // Its score.
};

/**
 * @struct MoveList
 * @brief Container for moves and their scores.
 */
struct MoveList
{

	std::array<ScoredMove, MAX_MOVES> moves; // The actual moves.
	uint32_t size{0};                        // Number of stored moves.

	/**
	 * @brief Sort the moves according to their euristics.
	 */
	void sort()
	{
		std::sort(
			moves.begin(), moves.begin() + size, 
			[](const ScoredMove &a, const ScoredMove &b) { return a.score > b.score;}
		);
	}

};



/**
 * @brief Heuristic to score a move for move ordering.
 * @param state The current game state.
 * @param move The move to score.
 * @return The heuristic score of the move.
 */
inline int32_t score_move(const Shogi::State& state, Shogi::Move move, Shogi::Player player) noexcept
{
	const Shogi::Square from = move.from();
	const Shogi::Square to = move.to();

	// Flat bous for drops depending of the agressivity.
	if (from == Shogi::Square::HAND) {
		const uint8_t line = to / 9;
		int32_t score = PIECE_VALUE[move.piece()]; 
		score += (player == Shogi::Player::SENTE) ? DROP_ZONE_BONUS_SENTE[line] : DROP_ZONE_BONUS_GOTE[line];
		return score; 
	}
	
	const Shogi::PieceType original_attacker = state[from].piece();
	const Shogi::PieceType final_piece       = move.piece();
	const Shogi::PieceType victim            = state[to].piece();

	// MVV-LVA score.
	int32_t score = MVV_LVA[original_attacker][victim];

	// Promotion bonus.
	if (original_attacker != final_piece) score += 5000;
	return score;
}



/**
 * @brief Check if a square is being attacked by a player.
 * @param state The current state of the board.
 * @param sq The attacked square.
 * @param attacker The attacking player.
 */
inline bool is_attacked(const Shogi::State &state, int32_t sq, Shogi::Player attacker) noexcept
{
	const Shogi::Piece* board  = state.data();
	const Shogi::Player defender = OPPONENT[attacker];

	// Get the possible attacking positions.
	const std::bitset<68> sq_dirs = Shogi::LUTs::MLUT[sq].flags();

	// Special case of the knights.
	for (int k = 0; k < 2; ++k) {
		if (!sq_dirs.test(KNIGHT_DIRS[defender][k])) continue;
		const Shogi::Piece p = board[sq + KNIGHT_OFFSETS[defender][k]];
		if (p.raw() && p.piece() == Shogi::PieceType::N && p.player() == attacker) return true;
	}

	// Raycasting for other pieces.
	for (int r = 0; r < 8; ++r) {
		if (!sq_dirs.test(RAY_STARTS[r])) continue;

		// Extend the ray until we have found a piece.
		const int32_t offset = RAY_OFFSETS[r];
		for (int32_t step = 0; step < 8; ++step) {
			if (!sq_dirs.test(RAY_STARTS[r] + step)) break;
			const Shogi::Piece p = board[sq + offset * (step + 1)];
			if (!p.raw()) continue;

			// Blocked ray. We check if it is an enemy piece that can attack us..
			if (p.player() == attacker) {
				const auto opp_dir = static_cast<Shogi::Direction>(OPPOSITE_RAY_STARTS[r] + step);
				if (Shogi::LUTs::PLUT[attacker][p.piece()].has(opp_dir)) return true;
			}
			break;
		}
	}
	return false;
}



/**
 * @brief Add a move to the list and generate a promoted version if possible.
 * @param list The move list to append to.
 * @param player The player making the move.
 * @param from The source square.
 * @param to The destination square.
 * @param p_type The type of the piece being moved.
 * @param promo_zone_mask Bitset of the current player promotion zone.
 * @note This is a macro to guarantee inlining regardless of compiler heuristics.
 */
#define ADD_MOVE_WITH_PROMO(list, player, from, to, p_type, promo_zone_mask)  \
do {                                                                          \
	const bool in_zone = CAN_PROMOTE[(p_type)] &&                             \
		((promo_zone_mask).test(static_cast<size_t>(from)) ||                 \
		 (promo_zone_mask).test(static_cast<size_t>(to)));                    \
	if (in_zone && must_promote((p_type), (player), (to))) {                  \
		auto& sm = (list).moves[(list).size++];                               \
		sm.move  = Shogi::Move((player), (from), static_cast<Shogi::Square>(to), PROMOTED[(p_type)]); \
		sm.score = score_move(state, sm.move, player);                        \
	} else {                                                                  \
		auto& sm = (list).moves[(list).size++];                               \
		sm.move  = Shogi::Move((player), (from), static_cast<Shogi::Square>(to), (p_type)); \
		sm.score = score_move(state, sm.move, player);                        \
		if (in_zone) {                                                        \
			auto& smp = (list).moves[(list).size++];                          \
			smp.move  = Shogi::Move((player), (from), static_cast<Shogi::Square>(to), PROMOTED[(p_type)]); \
			smp.score = score_move(state, smp.move, player);                  \
		}                                                                     \
	}                                                                         \
} while(0)

/**
 * @brief Check if the piece must be promoted at a given row. 
 * @param p The type of the piece to move.
 * @param player The player moving the piece.
 * @param to The destination square id.
 * @return true if the piece MUST be promoted. 
 */
inline bool must_promote(Shogi::PieceType p, Shogi::Player player, int32_t to)
{
	if (player == Shogi::Player::SENTE) {
		if (p == Shogi::PieceType::N) return to < 18;
		if (p == Shogi::PieceType::P || p == Shogi::PieceType::L) return to < 9;
	} else {
		if (p == Shogi::PieceType::N) return to >= 63;
		if (p == Shogi::PieceType::P || p == Shogi::PieceType::L) return to >= 72;
	}
	return false;
}

/**
 * @brief Move generation function.
 * @param state The current state of the game.
 * @param player The current turn player.
 * @param list The list where to store the possible moves.
 */
inline void generate_moves(const Shogi::State &state, Shogi::Player player, MoveList &list)
{
	
	// First add the possible piece movements.

	// Local copy of the current player promotion zone.
	const std::bitset<BOARD_SQUARES>& promo_zone = (player == Shogi::Player::SENTE) ? SENTE_PROMO_ZONE : GOTE_PROMO_ZONE;
	
	// Loop over the board to get the current player pieces.
	const Shogi::Piece* board = state.data();
	for (int32_t sq = 0; sq < BOARD_SQUARES; ++sq) {
		const uint8_t raw = board[sq].raw();
		if (!raw) continue;
		if ((raw & 1) != static_cast<uint8_t>(player)) continue;

		// Get the piece type and starting square.
		const Shogi::PieceType p_type = board[sq].piece();
		const Shogi::Square from = static_cast<Shogi::Square>(sq);

		// Get all valid directions on the board for the piece.
		std::bitset<68> valid_dirs = Shogi::LUTs::MLUT[sq] & Shogi::LUTs::PLUT[player][p_type];

		// Special case for the knight because its the only piece that have
		// moves that are not on the 8 canonical axis.
		if (p_type == Shogi::PieceType::N) {
			for (int k = 0; k < 2; ++k) {
				if (!valid_dirs.test(KNIGHT_DIRS[player][k])) continue;
				const int32_t to = sq + KNIGHT_OFFSETS[player][k];
				const uint8_t to_raw = board[to].raw();
				if (to_raw != 0 && (to_raw & 1) == static_cast<uint8_t>(player)) continue;
				ADD_MOVE_WITH_PROMO(list, player, from, to, p_type, promo_zone);
			}
			continue;
		}

		// Case for other pieces. Loop over each direction.
		for (int r = 0; r < 8; ++r)
		{
			if (!valid_dirs.test(RAY_STARTS[r])) continue;

			// Cast a ray startig from the piece.
			const int32_t offset = RAY_OFFSETS[r];
			for (int32_t step = 0; step < 8; ++step)
			{
				if (!valid_dirs.test(RAY_STARTS[r] + step)) break;
				const int32_t to = sq + offset * (step + 1);
				const uint8_t to_raw = board[to].raw();

				// While we find an empty square, add a movement.
				if (to_raw == 0) ADD_MOVE_WITH_PROMO(list, player, from, to, p_type, promo_zone);

				// If we encounter a piece from the other player, add the movement then stop the ray.
				else if ((to_raw & 1) != static_cast<uint8_t>(player)) {
					ADD_MOVE_WITH_PROMO(list, player, from, to, p_type, promo_zone);
					break;

				// If we encouter a piece from this player, stop the ray.
				} else break;
				
			}
		}
	}


	// Then add the drops.

	// Get this player hand and the free squares on the board.
	const auto &hand = state.hands(player);
	const std::bitset<BOARD_SQUARES> free_sq = state.free_square();

	// Pawn masking for last line and nifu.
	std::bitset<BOARD_SQUARES> pawn_valid_sq = free_sq;
	pawn_valid_sq &= (player == Shogi::Player::SENTE) ? ~SENTE_PAWN_LANCE_FORBIDDEN : ~GOTE_PAWN_LANCE_FORBIDDEN;
	const std::bitset<9> blocked_cols = ~state.pawn(player);
	for (int col = 0; col < 9; ++col) {
		if (!blocked_cols.test(col)) continue;
		pawn_valid_sq &= ~COL_MASKS[col];
	}

	// Lance masking for the last line only.
	std::bitset<BOARD_SQUARES> lance_valid_sq = free_sq;
	lance_valid_sq &= (player == Shogi::Player::SENTE) ? ~SENTE_PAWN_LANCE_FORBIDDEN : ~GOTE_PAWN_LANCE_FORBIDDEN;

	// Knight mask with the last two lines.
	std::bitset<BOARD_SQUARES> knight_valid_sq = free_sq;
	knight_valid_sq &= (player == Shogi::Player::SENTE) ? ~SENTE_KNIGHT_FORBIDDEN : ~GOTE_KNIGHT_FORBIDDEN;

	// Loop over each type of piece in the hand.
	for (int32_t p_val = Shogi::PieceType::R; p_val <= Shogi::PieceType::P; ++p_val) {
		if (hand[p_val] == 0) continue;
		const Shogi::PieceType drop_piece = static_cast<Shogi::PieceType>(p_val);

		// Get the valid square mask depending of the piece type.
		const std::bitset<BOARD_SQUARES>* valid_sq;
		switch (drop_piece) {
			case Shogi::PieceType::P: valid_sq = &pawn_valid_sq;   break;
			case Shogi::PieceType::L: valid_sq = &lance_valid_sq;  break;
			case Shogi::PieceType::N: valid_sq = &knight_valid_sq; break;
			default:                  valid_sq = &free_sq;         break;
		}

		// For each valid drop square, add a new move.
		for (size_t to = valid_sq->_Find_first(); to < BOARD_SQUARES; to = valid_sq->_Find_next(to)) {
			uint32_t id = list.size++;
			list.moves[id].move = Shogi::Move(
				player, Shogi::Square::HAND, static_cast<Shogi::Square>(to), drop_piece
			);
			list.moves[id].score = score_move(state, list.moves[id].move, player);
		}
	}

}

}
