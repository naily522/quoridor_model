#include "quoridor.hpp"
#include "player.hpp"
#include "view.hpp"

int main()
{
    Quoridor::State state;
    state.reset();
    HumanPlayer p1, p2;
    Player* players[2] = {&p1, &p2};

    while (true)
    {
        system("cls");
        display_board(state);
        Player* cur = players[state.turn - 1];

        if (cur->choose_type(state) == 'm')
        {
            while (!cur->choose_move(state).apply(state));
        }
        else
        {
            while (!cur->choose_wall(state).apply(state));
        }

        if (state.pos[0].first == 2 * Quoridor::ROW_SIZE - 1)
        {
            display_board(state);
            std::cout << "Player 1 wins!\n";
            break;
        }
        if (state.pos[1].first == 1)
        {
            display_board(state);
            std::cout << "Player 2 wins!\n";
            break;
        }
    }
    return 0;
}
