#include "csa.h"
#include "shogi.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>



void parse_pi(Shogi::State &state, const std::string &line)
{
	state = Shogi::State::make_default();
	for (size_t i = 2; i + 2 <= line.length(); i += 4) {
		state[line[i]-'0',line[i+1]-'0'] = Shogi::Pieces::NONE;
	}
}

void parse_row(Shogi::State &state, const std::string &line)
{
	for (uint32_t column = 9; column >= 1; --column) {
		uint32_t offset = 2 + (9-column)*3;
		if (line[offset] == ' ') continue;
		std::string piece_str = line.substr(offset+1, 2);
		Shogi::Player pl = (line[offset] == '+') ? Shogi::Player::SENTE : Shogi::Player::GOTE;
		Shogi::PieceType type = Shogi::toPieceType(piece_str);
		state[column,line[1]-'0'] = Shogi::Piece(pl, type);
	}
}

void fill_remaining_hand(Shogi::State &state, Shogi::Player player)
{
	state.hands(player) = FULL_HAND;
	for (uint32_t i = 0; i < 81; ++i) {
		Shogi::Piece p = state[static_cast<Shogi::Square>(i)];
		switch (p.piece()) {
		case Shogi::PieceType::R: 
		case Shogi::PieceType::pR: --state.hands(player)[Shogi::PieceType::R]; break;
		case Shogi::PieceType::B: 
		case Shogi::PieceType::pB: --state.hands(player)[Shogi::PieceType::B]; break;
		case Shogi::PieceType::G:  --state.hands(player)[Shogi::PieceType::G]; break;
		case Shogi::PieceType::S: 
		case Shogi::PieceType::pS: --state.hands(player)[Shogi::PieceType::S]; break;
		case Shogi::PieceType::N: 
		case Shogi::PieceType::pN: --state.hands(player)[Shogi::PieceType::N]; break;
		case Shogi::PieceType::L: 
		case Shogi::PieceType::pL: --state.hands(player)[Shogi::PieceType::L]; break;
		case Shogi::PieceType::P: 
		case Shogi::PieceType::pP: --state.hands(player)[Shogi::PieceType::P]; break;
		default: break;
		}
	}
	for (uint32_t i = 0; i < 8; ++i) 
		state.hands(player)[i] -= state.hands(static_cast<Shogi::Player>(1 - player))[i];
}

void parse_arbitrary(Shogi::State &state, const std::string &line)
{
	Shogi::Player player = (line[1] == '+') ? Shogi::Player::SENTE : Shogi::Player::GOTE;
	for (size_t i = 2; i + 3 < line.length(); i += 4) {
		std::string piece_str = line.substr(i+2, 2);
		if (piece_str == "AL") fill_remaining_hand(state, player);
		else {
			Shogi::PieceType type = Shogi::toPieceType(piece_str);
			if (line[i] == '0' && line[i+1] == '0') ++state.hands(player)[type];
			else state[line[i]-'0',line[i+1]-'0'] = Shogi::Piece(player, type);
		}
	}
}

void parse_p(Shogi::State &state, const std::string &line, Shogi::Player &player)
{
	if (line.length() < 2) {
		if (line[0] == '+') player = Shogi::Player::SENTE;
		else player = Shogi::Player::GOTE;
		return;
	}
	if (line[1] == 'I') parse_pi(state, line);
	else if (std::isdigit(line[1])) parse_row(state, line);
	else parse_arbitrary(state, line);
}

void parse_cmd(const std::string &line, const Shogi::Player player, bool &sente_win, bool &gote_win)
{
	if (line.find("%TORYO") != std::string::npos || 
		line.find("%TIME_UP") != std::string::npos ||
		line.find("%TSUMI") != std::string::npos) {
		if (player == Shogi::Player::SENTE) gote_win = true;
		else sente_win = true;
	} else if (line.find("%ILLEGAL_MOVE") != std::string::npos ||
		line.find("%KACHI") != std::string::npos) {
		if (player == Shogi::Player::SENTE) sente_win = true;
		else gote_win = true;
	} else if (line.find("%+ILLEGAL_ACTION") != std::string::npos) {
		gote_win = true;
	} else if (line.find("%-ILLEGAL_ACTION") != std::string::npos) {
		sente_win = true;
	}
}

void parse_csa(
	const std::string &path, 
	bool &sente_win, 
	bool &gote_win, 
	std::vector<Shogi::Move> &moves, 
	Shogi::State &state
) {

	std::ifstream file(path.c_str());
	if (!file.is_open()) {
		std::cerr << std::format("Failed to open file '{}'.", path) << std::endl; 
		return;
	}

	sente_win = false;
	gote_win = false;
	Shogi::Player current_player = Shogi::Player::SENTE;

	std::string line;
	while (std::getline(file, line)) {
		const size_t first = line.find_first_not_of(" \t\r\n\v\f");
		if (first == std::string::npos) continue;
		if (moves.size()) current_player = moves.back().player(); 

		switch (line[0]) {
		case 'P':
			parse_p(state, line, current_player);
			break;
		case '+':
		case '-':
			if (line.size() >= 7) moves.push_back(Shogi::Move(line.c_str()));
			break;
		case '%':
			parse_cmd(line, current_player, sente_win, gote_win);
			break;
		default: break;
		}
	}

}



CSARecord::CSARecord(const std::string &path)
{
	parse_csa(path, m_sente_win, m_gote_win, m_moves, m_init);
}

Shogi::State CSARecord::operator[](uint32_t move)
{
	Shogi::State state = m_init;
	const uint32_t N = std::max(move, static_cast<uint32_t>(m_moves.size()));
	for (uint32_t i = 0; i < N; ++i) state(m_moves[i]);
	return state;
}
