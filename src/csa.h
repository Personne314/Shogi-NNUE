#pragma once

#include <cstdint>
#include <map>
#include <vector>

#include "shogi.hpp"




class CSARecord
{
public:

	CSARecord(const std::string &path);
	~CSARecord() = default;

	Shogi::State operator[](uint32_t move);

	const Shogi::State &init() const { return m_init; }
	const std::vector<Shogi::Move> &moves() const { return m_moves; }
	bool win(Shogi::Player player) const { return player == Shogi::Player::SENTE ? m_sente_win : m_gote_win; }
	bool draw() const { return !(m_sente_win || m_gote_win); }  

private:

	bool m_sente_win{false};
	bool m_gote_win{false};

	Shogi::State m_init{};
	std::vector<Shogi::Move> m_moves{};

};
