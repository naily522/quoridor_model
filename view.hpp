#ifndef VIEW_HPP
#define VIEW_HPP

#include "quoridor.hpp"
#include <windows.h>

inline void display_board(const Quoridor::State& state)
{
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    const int N = 2 * Quoridor::ROW_SIZE + 1; // 19

    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            COORD pos = {(SHORT)(c * 2), (SHORT)r};
            SetConsoleCursorPosition(h, pos);

            if (state.board[r][c] == 1) {              // wall
                SetConsoleTextAttribute(h, 0);
                std::cout << "  ";
            } else if (r % 2 == 1 && c % 2 == 1) {     // playable square
                int attr = BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_INTENSITY; // yellow bg
                if (state.pos[0].first == r && state.pos[0].second == c)
                    attr |= FOREGROUND_BLUE | FOREGROUND_INTENSITY;
                else if (state.pos[1].first == r && state.pos[1].second == c)
                    attr |= FOREGROUND_GREEN | FOREGROUND_INTENSITY;
                SetConsoleTextAttribute(h, attr);
                std::cout << (attr & 0x0F ? "O " : "  ");
            } else {                                   // wall channel / crossing
                SetConsoleTextAttribute(h, BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE);
                std::cout << "  ";
            }
        }
        std::cout << "\n";
    }
    SetConsoleTextAttribute(h, 7);
}

#endif
