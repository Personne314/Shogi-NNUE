#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <sstream>
#include <string>



namespace Shogi
{

// Constants to describe the structure of a move and a hand. 

#define MOVE_PLAYER_OFF 0  // Player id offset.
#define MOVE_PIECE_OFF  1  // Piece id offset.
#define MOVE_FROM_OFF   5  // Starting position offset.
#define MOVE_TO_OFF     12 // Ending position offset.

#define MOVE_PLAYER_MASK 0b0000'0000'0000'0000'0001 // Player id mask.
#define MOVE_PIECE_MASK  0b0000'0000'0000'0001'1110 // Piece id mask.
#define MOVE_FROM_MASK   0b0000'0000'1111'1110'0000 // Starting position mask.
#define MOVE_TO_MASK     0b0111'1111'0000'0000'0000 // Ending position mask.

// Typedef for the hand content.
typedef std::array<uint8_t, 8> Hand;

#define FULL_HAND Shogi::Hand({0,2,2,4,4,4,4,18}); // Hand with all possible pieces in it.

// Number of squares in the board.
#define BOARD_SQUARES 81



/**
 * @enum PieceType
 * @brief List all existing piece type.
 */
enum PieceType : uint8_t // 4bit
{
	UNKNOWN,
	R,	// Rook
	B,	// Bishop
	G,	// Gold
	S,	// Silver
	N,	// Knight
	L,	// Lance
	P,	// Pawn
	K,	// King
	pR,	// Promoted Rook
	pB,	// Promoted Bishop
	pS,	// Promoted Silver
	pN,	// Promoted Knight
	pL,	// Promoted Lance
	pP	// Promoted Pawn
};

/**
 * @brief Helper to get a PieceType from its string representation.
 * @param str The string representation.
 * @return The Piece type value.
 */
inline PieceType toPieceType(std::string &str)
{
	if (str.length() != 2) return PieceType::UNKNOWN;
	if      (str[0] == 'O' && str[1] == 'U') return PieceType::K;
	else if (str[0] == 'H' && str[1] == 'I') return PieceType::R;
	else if (str[0] == 'K' && str[1] == 'A') return PieceType::B;
	else if (str[0] == 'K' && str[1] == 'I') return PieceType::G;
	else if (str[0] == 'G' && str[1] == 'I') return PieceType::S;
	else if (str[0] == 'K' && str[1] == 'E') return PieceType::N;
	else if (str[0] == 'K' && str[1] == 'Y') return PieceType::L;
	else if (str[0] == 'F' && str[1] == 'U') return PieceType::P;
	else if (str[0] == 'R' && str[1] == 'Y') return PieceType::pR;
	else if (str[0] == 'U' && str[1] == 'M') return PieceType::pB;
	else if (str[0] == 'N' && str[1] == 'G') return PieceType::pS;
	else if (str[0] == 'N' && str[1] == 'K') return PieceType::pN;
	else if (str[0] == 'N' && str[1] == 'Y') return PieceType::pL;
	else if (str[0] == 'T' && str[1] == 'O') return PieceType::pP;
	else return PieceType::UNKNOWN;
}

/**
 * @brief Helper to get a string representation from a PieceType.
 * @param type The Piece type value.
 * @return The string representation.
 */
inline std::string to_string(PieceType type)
{
	switch (type) {
	case UNKNOWN: return "* ";
	case R:       return "HI";
	case B:       return "KA";
	case G:       return "KI";
	case S:       return "GI";
	case N:       return "KE";
	case L:       return "KY";
	case P:       return "FU";
	case K:       return "OU";
	case pR:      return "RY";
	case pB:      return "UM";
	case pS:      return "NG";
	case pN:      return "NK";
	case pL:      return "NY";
	case pP:      return "TO";
	}
}



/**
 * @enum Player
 * @brief List the players.
 */
enum Player : uint8_t // 1bit
{
	SENTE,	// First Player
	GOTE	// Second Player
};

/**
 * @brief Helper to get a string representation from a Player.
 * @param player The player value.
 * @return The string representation.
 */
inline std::string to_string(Player player)
{
	return player == Player::SENTE ? "+" : "-";
}



/**
 * @enum Square
 * @brief Represent one square of the board.
 * @note Column, line. Origin is upper right corner.
 */
enum Square : uint8_t // 7bit. 
{
	S91, S81, S71, S61, S51, S41, S31, S21, S11,
	S92, S82, S72, S62, S52, S42, S32, S22, S12,
	S93, S83, S73, S63, S53, S43, S33, S23, S13,
	S94, S84, S74, S64, S54, S44, S34, S24, S14,
	S95, S85, S75, S65, S55, S45, S35, S25, S15,
	S96, S86, S76, S66, S56, S46, S36, S26, S16,
	S97, S87, S77, S67, S57, S47, S37, S27, S17,
	S98, S88, S78, S68, S58, S48, S38, S28, S18,
	S99, S89, S79, S69, S59, S49, S39, S29, S19,
	HAND
};

/**
 * @brief Helper to get a string representation from a Square.
 * @param square The square value.
 * @return The string representation.
 */
inline std::string to_string(Square square)
{
	if (square == BOARD_SQUARES) return "00";
	return std::string(1, '9'-(square%9)) + std::string(1, '1'+square/9);
}



/**
 * @class Piece
 * @brief Represent one unique piece on the board.
 * @note A piece is considered to be defined by its type and owner. 
 */
class Piece
{
public:

	constexpr Piece() noexcept = default; // <-> ShogiPiece(SENTE, NONE)
	constexpr Piece(Player player, PieceType piece) noexcept
	{
		m_value = 
			(player << MOVE_PLAYER_OFF) |
			(piece  << MOVE_PIECE_OFF);
	}

	constexpr Player player() const noexcept
	{ 
		return static_cast<Player>((m_value & MOVE_PLAYER_MASK) >> MOVE_PLAYER_OFF); 
	}

	
	constexpr PieceType piece() const noexcept
	{ 
		return static_cast<PieceType>((m_value & MOVE_PIECE_MASK) >> MOVE_PIECE_OFF); 
	}

	constexpr uint8_t raw() const noexcept
	{
		return m_value;
	} 

	std::string str() const
	{
		return m_value ? to_string(player()) + to_string(piece()) : " * ";
	}

	constexpr bool operator==(const Piece piece) const noexcept
	{
		return m_value == piece.m_value;
	}

	constexpr bool operator!=(const Piece piece) const noexcept
	{
		return m_value != piece.m_value;
	}

private:

	uint8_t m_value{0}; // Piece integer value.

};

// Typedef for the Board content.
typedef std::array<Piece, BOARD_SQUARES> Board;

/**
 * @namespace Pieces
 * @brief Define all possible unique pieces.
 */
namespace Pieces
{
	constexpr Piece S_K  = Piece(Player::SENTE, PieceType::K);
	constexpr Piece S_R  = Piece(Player::SENTE, PieceType::R);
	constexpr Piece S_B  = Piece(Player::SENTE, PieceType::B);
	constexpr Piece S_G  = Piece(Player::SENTE, PieceType::G);
	constexpr Piece S_S  = Piece(Player::SENTE, PieceType::S);
	constexpr Piece S_N  = Piece(Player::SENTE, PieceType::N);
	constexpr Piece S_L  = Piece(Player::SENTE, PieceType::L);
	constexpr Piece S_P  = Piece(Player::SENTE, PieceType::P);
	constexpr Piece S_pR = Piece(Player::SENTE, PieceType::pR);
	constexpr Piece S_pB = Piece(Player::SENTE, PieceType::pB);
	constexpr Piece S_pS = Piece(Player::SENTE, PieceType::pS);
	constexpr Piece S_pN = Piece(Player::SENTE, PieceType::pN);
	constexpr Piece S_pL = Piece(Player::SENTE, PieceType::pL);
	constexpr Piece S_pP = Piece(Player::SENTE, PieceType::pP);
	constexpr Piece G_K  = Piece(Player::GOTE,  PieceType::K);
	constexpr Piece G_R  = Piece(Player::GOTE,  PieceType::R);
	constexpr Piece G_B  = Piece(Player::GOTE,  PieceType::B);
	constexpr Piece G_G  = Piece(Player::GOTE,  PieceType::G);
	constexpr Piece G_S  = Piece(Player::GOTE,  PieceType::S);
	constexpr Piece G_N  = Piece(Player::GOTE,  PieceType::N);
	constexpr Piece G_L  = Piece(Player::GOTE,  PieceType::L);
	constexpr Piece G_P  = Piece(Player::GOTE,  PieceType::P);
	constexpr Piece G_pR = Piece(Player::GOTE,  PieceType::pR);
	constexpr Piece G_pB = Piece(Player::GOTE,  PieceType::pB);
	constexpr Piece G_pS = Piece(Player::GOTE,  PieceType::pS);
	constexpr Piece G_pN = Piece(Player::GOTE,  PieceType::pN);
	constexpr Piece G_pL = Piece(Player::GOTE,  PieceType::pL);
	constexpr Piece G_pP = Piece(Player::GOTE,  PieceType::pP);
	constexpr Piece NONE = Piece(Player::SENTE, PieceType::UNKNOWN);
};



/**
 * @class Move
 * @brief Describe a move.
 * @note A move is defined by the Piece, the starting square and the ending square.
 */
class Move
{
public:

	constexpr Move() noexcept = default;
	Move(const std::string &str)
	{

		if (str[0] == '+') m_value = 0;
		else m_value = 1;
		
		m_value |= ((str[4]-'1')*9 + '9'-str[3]) << MOVE_TO_OFF;
		if (str[1] != '0') m_value |= ((str[2]-'1')*9 + '9'-str[1]) << MOVE_FROM_OFF;
		else m_value |= Square::HAND << MOVE_FROM_OFF;

		std::string piece_str = str.substr(5,2);
		PieceType piece = Shogi::toPieceType(piece_str);
		m_value |= piece << MOVE_PIECE_OFF;

	}

	constexpr Move(Player player, Square from, Square to, PieceType piece) noexcept
	{
		m_value = 
			(player << MOVE_PLAYER_OFF) |
			(from   << MOVE_FROM_OFF)   |
			(to     << MOVE_TO_OFF)     |
			(piece  << MOVE_PIECE_OFF);
	}

	~Move() noexcept = default;

	constexpr Player player() const noexcept
	{ 
		return static_cast<Player>((m_value & MOVE_PLAYER_MASK) >> MOVE_PLAYER_OFF); 
	}
	
	constexpr PieceType piece() const noexcept
	{ 
		return static_cast<PieceType>((m_value & MOVE_PIECE_MASK) >> MOVE_PIECE_OFF); 
	}
	
	constexpr Square from() const noexcept
	{ 
		return static_cast<Square>((m_value & MOVE_FROM_MASK) >> MOVE_FROM_OFF); 
	}
	
	constexpr Square to() const noexcept
	{ 
		return static_cast<Square>((m_value & MOVE_TO_MASK) >> MOVE_TO_OFF); 
	}

	constexpr uint32_t raw() const noexcept
	{
		return m_value;
	} 

	std::string str() const 
	{
		return to_string(player()) + to_string(from()) + to_string(to()) + to_string(piece());
	}

private:

	uint32_t m_value{0}; // Move integer value.

};



/**
 * @enum Direction
 * @brief List of all possible moves from one given position.
 * @note Represent move directions feasable by at least one piece of the game regardless of board limits.
 */
enum Direction : uint8_t // 7bits.
{
	U1, U2, U3, U4, U5, U6, U7, U8,
	D1, D2, D3, D4, D5, D6, D7, D8,
	L1, L2, L3, L4, L5, L6, L7, L8,
	R1, R2, R3, R4, R5, R6, R7, R8,
	UL1, UL2, UL3, UL4, UL5, UL6, UL7, UL8,
	UR1, UR2, UR3, UR4, UR5, UR6, UR7, UR8,
	DL1, DL2, DL3, DL4, DL5, DL6, DL7, DL8,
	DR1, DR2, DR3, DR4, DR5, DR6, DR7, DR8,
	KUL, KUR, KDL, KDR
};

/**
 * @class DirectionLUT
 * @brief LUT to get all feasable moves by at least one piece of the game at one given position.
 * @note This LUT is used to remove out-of-board directions at compile time.
 */
class DirectionLUT
{
public:

	consteval DirectionLUT() noexcept = default;
	consteval DirectionLUT(const Square square) noexcept
	{
		const uint8_t l = square/9;
		const uint8_t c = square%9;
		
		const uint8_t ur = std::min(l, c);
		const uint8_t ul = std::min(l, static_cast<uint8_t>(8-c));
		const uint8_t dr = std::min(static_cast<uint8_t>(8-l), c);
		const uint8_t dl = std::min(static_cast<uint8_t>(8-l), static_cast<uint8_t>(8-c));

		for (uint8_t i = 0; i < l;   ++i) m_flags.set(Direction::U1 + i);
		for (uint8_t i = 0; i < 8-l; ++i) m_flags.set(Direction::D1 + i);
		for (uint8_t i = 0; i < 8-c; ++i) m_flags.set(Direction::L1 + i);
		for (uint8_t i = 0; i < c;   ++i) m_flags.set(Direction::R1 + i);

		for (uint8_t i = 0; i < ul; ++i) m_flags.set(Direction::UL1 + i);
		for (uint8_t i = 0; i < ur; ++i) m_flags.set(Direction::UR1 + i);
		for (uint8_t i = 0; i < dl; ++i) m_flags.set(Direction::DL1 + i);
		for (uint8_t i = 0; i < dr; ++i) m_flags.set(Direction::DR1 + i);

		if (l >= 2 && c <= 7) m_flags.set(Direction::KUL);
		if (l >= 2 && c >= 1) m_flags.set(Direction::KUR);
		if (l <= 6 && c <= 7) m_flags.set(Direction::KDL);
		if (l <= 6 && c >= 1) m_flags.set(Direction::KDR);

	}

	consteval bool has(Direction d) const noexcept
	{
		return m_flags.test(static_cast<size_t>(d));
	}

	constexpr std::bitset<68> flags() const noexcept
	{
		return m_flags;
	}

private:

	std::bitset<68> m_flags{}; // LUT flags.

};

/**
 * @class PieceLUT
 * @brief LUT to get all feasable moves by one given piece of the game regardless of its position.
 * @note This LUT is used to get piece moves at compile time.
 */
class PieceLUT
{
public:

	consteval PieceLUT() noexcept = default;
	consteval PieceLUT(std::initializer_list<Direction> dirs) noexcept
	{
		for (Direction d : dirs) m_flags.set(static_cast<size_t>(d));
	}

	constexpr bool has(Direction d) const noexcept
	{
		return m_flags.test(static_cast<size_t>(d));
	}

	constexpr std::bitset<68> flags() noexcept
	{
		return m_flags;
	}

private:

	std::bitset<68> m_flags{}; // LUT flags.

};

/**
 * @brief Bitwise AND for Direction and Piece LUTs. 
 * @param dlut The Direction LUT to use.
 * @param dlut The Piece LUT to use.
 * @return A bitset representing the valid moves of one piece at a given position on the board.
 */
inline std::bitset<68> operator&(DirectionLUT dlut, PieceLUT plut)
{
	return dlut.flags() & plut.flags();
}

/**
 * @brief Bitwise AND for Direction and Piece LUTs.
 * @param dlut The Piece LUT to use.
 * @param dlut The Direction LUT to use.
 * @return A bitset representing the valid moves of one piece at a given position on the board.
 */
inline std::bitset<68> operator&(PieceLUT dlut, DirectionLUT plut)
{
	return dlut.flags() & plut.flags();
}



/**
 * @namespace LUTs
 * @brief Contains all useful compile-time LUTs.
 */
namespace LUTs
{

	/**
	 * @brief Generate Direction LUTs for each square on the board.
	 * @return An array containing the LUTs indexed by Square as the id. 
	 */
	consteval std::array<DirectionLUT, BOARD_SQUARES> gen_direction_LUTs()
	{
		std::array<DirectionLUT, BOARD_SQUARES> table{};
		for (int i = 0; i < BOARD_SQUARES; ++i) table[i] = DirectionLUT(static_cast<Square>(i));
		return table;
	}

	/**
	 * @brief Generate Piece LUTs for each sente piece on the board.
	 * @return An array containing the LUTs indexed by the PieceType as the id. 
	 */
	consteval std::array<PieceLUT, 15> gen_sente_piece_LUTs()
	{
		using enum Direction;
		std::array<PieceLUT, 15> table {
			PieceLUT(),	// Unknown
			PieceLUT({
				U1,U2,U3,U4,U5,U6,U7,U8,D1,D2,D3,D4,D5,D6,D7,D8,
				L1,L2,L3,L4,L5,L6,L7,L8,R1,R2,R3,R4,R5,R6,R7,R8
			}), // Rook
			PieceLUT({
				UL1,UL2,UL3,UL4,UL5,UL6,UL7,UL8,UR1,UR2,UR3,UR4,UR5,UR6,UR7,UR8,
				DL1,DL2,DL3,DL4,DL5,DL6,DL7,DL8,DR1,DR2,DR3,DR4,DR5,DR6,DR7,DR8
			}), // Bishop
			PieceLUT({UL1,U1,UR1,L1,R1,D1}),         // Gold
			PieceLUT({UL1,U1,UR1,DL1,DR1}),          // Silver
			PieceLUT({KUL,KUR}),                     // Knight
			PieceLUT({U1,U2,U3,U4,U5,U6,U7,U8}),     // Lance
			PieceLUT({U1}),                          // Pawn
			PieceLUT({UL1,U1,UR1,L1,R1,DL1,D1,DR1}), // King
			PieceLUT({
				U1,U2,U3,U4,U5,U6,U7,U8,D1,D2,D3,D4,D5,D6,D7,D8,
				L1,L2,L3,L4,L5,L6,L7,L8,R1,R2,R3,R4,R5,R6,R7,R8,
				UL1,UR1,DL1,DR1
			}),	// Promoted Rook
			PieceLUT({
				UL1,UL2,UL3,UL4,UL5,UL6,UL7,UL8,UR1,UR2,UR3,UR4,UR5,UR6,UR7,UR8,
				DL1,DL2,DL3,DL4,DL5,DL6,DL7,DL8,DR1,DR2,DR3,DR4,DR5,DR6,DR7,DR8,
				U1,D1,L1,R1
			}),	// Promoted Bishop
			PieceLUT({UL1,U1,UR1,L1,R1,D1}), // Promoted Silver
			PieceLUT({UL1,U1,UR1,L1,R1,D1}), // Promoted Knight
			PieceLUT({UL1,U1,UR1,L1,R1,D1}), // Promoted Lance
			PieceLUT({UL1,U1,UR1,L1,R1,D1})  // Promoted Pawn
		};
		return table;
	}	

	/**
	 * @brief Generate Piece LUTs for each gote piece on the board.
	 * @return An array containing the LUTs indexed by the PieceType as the id. 
	 */
	consteval std::array<PieceLUT, 15> gen_gote_piece_LUTs()
	{
		std::array<PieceLUT, 15> table {
			PieceLUT(),	// Unknown
			PieceLUT({
				U1,U2,U3,U4,U5,U6,U7,U8,D1,D2,D3,D4,D5,D6,D7,D8,
				L1,L2,L3,L4,L5,L6,L7,L8,R1,R2,R3,R4,R5,R6,R7,R8
			}), // Rook
			PieceLUT({
				UL1,UL2,UL3,UL4,UL5,UL6,UL7,UL8,UR1,UR2,UR3,UR4,UR5,UR6,UR7,UR8,
				DL1,DL2,DL3,DL4,DL5,DL6,DL7,DL8,DR1,DR2,DR3,DR4,DR5,DR6,DR7,DR8
			}), // Bishop
			PieceLUT({DL1,D1,DR1,L1,R1,U1}),         // Gold
			PieceLUT({DL1,D1,DR1,UL1,UR1}),          // Silver
			PieceLUT({KDL,KDR}),                     // Knight
			PieceLUT({D1,D2,D3,D4,D5,D6,D7,D8}),     // Lance
			PieceLUT({D1}),                          // Pawn
			PieceLUT({UL1,U1,UR1,L1,R1,DL1,D1,DR1}), // King
			PieceLUT({
				U1,U2,U3,U4,U5,U6,U7,U8,D1,D2,D3,D4,D5,D6,D7,D8,
				L1,L2,L3,L4,L5,L6,L7,L8,R1,R2,R3,R4,R5,R6,R7,R8,
				UL1,UR1,DL1,DR1
			}),	// Promoted Rook
			PieceLUT({
				UL1,UL2,UL3,UL4,UL5,UL6,UL7,UL8,UR1,UR2,UR3,UR4,UR5,UR6,UR7,UR8,
				DL1,DL2,DL3,DL4,DL5,DL6,DL7,DL8,DR1,DR2,DR3,DR4,DR5,DR6,DR7,DR8,
				U1,D1,L1,R1
			}),	// Promoted Bishop
			PieceLUT({DL1,D1,DR1,L1,R1,U1}), // Promoted Silver
			PieceLUT({DL1,D1,DR1,L1,R1,U1}), // Promoted Knight
			PieceLUT({DL1,D1,DR1,L1,R1,U1}), // Promoted Lance
			PieceLUT({DL1,D1,DR1,L1,R1,U1})  // Promoted Pawn
		};
		return table;
	}

	// Direction LUTs for each square on the board.
	constexpr std::array<DirectionLUT, BOARD_SQUARES> MLUT = gen_direction_LUTs();

	// Piece LUTs for each sente and gote piece on the board. Indexed by Player, PieceType.
	constexpr std::array<PieceLUT, 15> PLUT[2] = 
	{
		gen_sente_piece_LUTs(),
		gen_gote_piece_LUTs()
	};

};



/**
 * @class State
 * @brief Represent the complete state of a game using the previously defined representations.
 */
class State
{

public:

	consteval State() noexcept = default;
	constexpr State(const Board &board) noexcept : 
		m_board(board) 
	{
		for (uint32_t i = 0; i < BOARD_SQUARES; ++i) {
			if (m_board[i] == Pieces::NONE) m_free.set(i);
			if (m_board[i] == Pieces::S_P) m_sente_pawn.set(8-i%9);
			if (m_board[i] == Pieces::G_P) m_gote_pawn.set(8-i%9);
			if (m_board[i] == Pieces::S_K) m_king_sq[SENTE] = i;
			if (m_board[i] == Pieces::G_K) m_king_sq[GOTE] = i;
		}
		m_sente_pawn = ~m_sente_pawn;
		m_gote_pawn = ~m_gote_pawn;
	}

	constexpr State &operator()(Move move)
	{

		const Square from = move.from();
		const Square to = move.to();
		const Player player = move.player();
		const PieceType piece = move.piece();

		const Piece old_to = m_board[to];
		const PieceType old_to_piece = old_to.piece();
		
		if (old_to_piece != PieceType::UNKNOWN) {
			switch (old_to_piece) {
			case PieceType::pR:	++m_hands[player][PieceType::R]; break;
			case PieceType::pB:	++m_hands[player][PieceType::B]; break;
			case PieceType::pS:	++m_hands[player][PieceType::S]; break;
			case PieceType::pN:	++m_hands[player][PieceType::N]; break;
			case PieceType::pL:	++m_hands[player][PieceType::L]; break;
			case PieceType::pP:	++m_hands[player][PieceType::P]; break;
			case PieceType::K:		break;
			default: ++m_hands[player][old_to_piece];
			}
		}

		if (old_to == Pieces::S_P) m_sente_pawn.set(8-to%9);
		if (old_to == Pieces::G_P) m_gote_pawn.set(8-to%9);

		if (from == Square::HAND) --m_hands[player][piece];	
		else {
			const Piece old_from = m_board[from];
			if (old_from == Pieces::S_P) m_sente_pawn.set(8-from%9);
			if (old_from == Pieces::G_P) m_gote_pawn.set(8-from%9);
			m_board[from] = Pieces::NONE;
			m_free.set(from);
		}

		const Piece new_to = Piece(player, piece);
		if (new_to == Pieces::S_P) m_sente_pawn.reset(8-to%9);
		if (new_to == Pieces::G_P) m_gote_pawn.reset(8-to%9);
		m_board[to] = new_to;
		m_free.reset(to);

		if (piece == PieceType::K) m_king_sq[player] = to;
		return *this;
	}

	constexpr Piece operator[](Square square) const noexcept
	{
		return m_board[square];
	}
	
	constexpr Piece &operator[](Square square) noexcept
	{
		return m_board[square];
	}

	constexpr Piece operator[](uint8_t c, uint8_t l) const noexcept
	{
		return m_board[9*l-c];
	}

	constexpr Piece &operator[](uint8_t c, uint8_t l) noexcept
	{ 
		return m_board[9*l-c];
	}
	
	constexpr std::bitset<BOARD_SQUARES> free_square() const noexcept
	{
		return m_free;
	}
	
	constexpr std::bitset<9> pawn(Player player) const noexcept
	{
		return player == Player::SENTE ? m_sente_pawn : m_gote_pawn;
	}
	
	constexpr const Hand &hands(Player player) const noexcept
	{
		return m_hands[player];
	}

	constexpr Hand &hands(Player player) noexcept
	{
		return m_hands[player];
	}

	constexpr const Shogi::Piece *data() const noexcept
	{
		return m_board.data();
	}

	constexpr uint32_t king_sq(Player p) const noexcept
	{
		return m_king_sq[p];
	}

	std::string str() const
	{
		std::stringstream sstr;
		for (uint8_t l = 1; l <= 9; ++l) {
			sstr << "P" << std::to_string(l);
			for (uint8_t c = 9; c >= 1; --c) sstr << (*this)[c,l].str();
			sstr << "\n";
		}
		return sstr.str();
	}

	static consteval State make_default() noexcept
	{
		using namespace Pieces;
		return State({
			G_L,  G_N,  G_S,  G_G,  G_K,  G_G,  G_S,  G_N,  G_L,
			NONE, G_R,  NONE, NONE, NONE, NONE, NONE, G_B,  NONE,
			G_P,  G_P,  G_P,  G_P,  G_P,  G_P,  G_P,  G_P,  G_P,
			NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
			NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
			NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE,
			S_P,  S_P,  S_P,  S_P,  S_P,  S_P,  S_P,  S_P,  S_P,
			NONE, S_B,  NONE, NONE, NONE, NONE, NONE, S_R,  NONE,
			S_L,  S_N,  S_S,  S_G,  S_K,  S_G,  S_S,  S_N,  S_L
		});
	}

private:

	Board m_board{Pieces::NONE};         // The actual board content. 
	Hand m_hands[2]{ {0}, {0} };         // Both player hands.
	std::bitset<BOARD_SQUARES> m_free{}; // Bitset to store the free squares ids.
	std::bitset<9> m_sente_pawn{};       // Bitset to get the columns where there is a sente pawn.
	std::bitset<9> m_gote_pawn{};        // Bitset to get the columns where there is a gote pawn.
	int32_t m_king_sq[2]{-1, -1};        // Positions of both player kings.

};

}
