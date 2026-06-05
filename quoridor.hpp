#ifndef QUORIDOR_HPP
#define QUORIDOR_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <queue>

using std::pair;
using std::vector;
using std::string;
using std::queue;

/******************* Environment Definition *******************/
class Quoridor
{
public:
    static constexpr int ROW_SIZE=9, COLUMN_SIZE=9, WALL_LENGTH=2, WALL_NUM=10;
    // 动作空间: 81 个落子位置 + 72 垂直墙 + 72 水平墙
    static constexpr int ACTION_NUM = 225;

    bool verbose;

    struct State
    {
        pair<int, int> pos[2];
        bool board[2*ROW_SIZE+1][2*COLUMN_SIZE+1]; // (odd,odd)=square; (even,*)=wall/channel
        bool h_wall[ROW_SIZE][COLUMN_SIZE];         // horizontal wall tracking (9x9)
        bool v_wall[ROW_SIZE][COLUMN_SIZE];         // vertical wall tracking (9x9)
        int turn, wall_num[2]; // turn=1,2 for player1, player2; wall_num for remaining walls

        void reset();
    };

    struct Action
    {
        pair<int, int> pos; // target pos for move, or wall start coordinate for wall
        bool isWall;
        int wall_dir; // 0: vertical, 1: horizoncal

        Action(pair<int, int> p, bool w, int d) : pos(p), isWall(w), wall_dir(d) {}
        bool apply(State& state) const;
    };
    
    void log_msg(const string& msg)
    { if (verbose) log.push_back(msg); }

    void flush_log(const string& path)
    {
        std::ofstream f(path);
        for(auto& l : log) f << l << std::endl;
    }

    void clear_log()
    { log.clear(); }
private:
    vector<string> log;
};

void Quoridor::State::reset()
{
    memset(this->board, 0, sizeof(this->board));
    memset(this->h_wall, 0, sizeof(this->h_wall));
    memset(this->v_wall, 0, sizeof(this->v_wall));
    // boundary walls (outer ring)
    for (int i = 0; i < 2*ROW_SIZE+1; i++)
    {
        this->board[i][0] = 1;
        this->board[i][2*COLUMN_SIZE] = 1;
    }
    for (int j = 0; j < 2*COLUMN_SIZE+1; j++)
    {
        this->board[0][j] = 1;
        this->board[2*ROW_SIZE][j] = 1;
    }
    this->turn = 1;
    this->wall_num[0] = this->wall_num[1] = WALL_NUM;
    this->pos[0] = {1, COLUMN_SIZE + (1-COLUMN_SIZE%2)};         // top center
    this->pos[1] = {2*ROW_SIZE-1, COLUMN_SIZE + (1-COLUMN_SIZE%2)}; // bottom center
}

bool Quoridor::Action::apply(State& state) const
{
    if (this->isWall)
    {
        // accept only odd coordinates (wall start at player grid positions)
        if (this->pos.first % 2 != 0 || this->pos.second % 2 != 0) return false;

        State temp_state = state;
        int wr = this->pos.first / 2;   // 9x9 index
        int wc = this->pos.second / 2;

        if (this->wall_dir == 0) // vertical
        {
            for (int i = -(WALL_LENGTH - 1); i < WALL_LENGTH; i++)
            {
                if (temp_state.board[this->pos.first + i][this->pos.second] == 1) return false;
                temp_state.board[this->pos.first + i][this->pos.second] = 1;
            }
            temp_state.v_wall[wr][wc] = true;
        }
        else // horizontal
        {
            for (int i = -(WALL_LENGTH - 1); i < WALL_LENGTH; i++)
            {
                if (temp_state.board[this->pos.first][this->pos.second + i] == 1) return false;
                temp_state.board[this->pos.first][this->pos.second + i] = 1;
            }
            temp_state.h_wall[wr][wc] = true;
        }

        // check if the wall blocks all paths for either player
        bool isconnect(const bool board[2*ROW_SIZE+1][2*COLUMN_SIZE+1], pair<int, int> start, int target_row);

        if (!isconnect(temp_state.board, state.pos[0], 2*ROW_SIZE-1) || !isconnect(temp_state.board, state.pos[1], 1)) return false;

        state = temp_state; // update state with the new wall
        state.wall_num[state.turn-1]--; // decrease wall count for the player
    }
    else // move
    {
        state.pos[state.turn-1] = this->pos;
    }

    state.turn = 3 - state.turn;
    return true;
}

bool isconnect(const bool board[2*Quoridor::ROW_SIZE+1][2*Quoridor::COLUMN_SIZE+1], pair<int, int> start, int target_row)
{
    queue<pair<int, int>> q;
    bool isvisited[2*Quoridor::ROW_SIZE+1][2*Quoridor::COLUMN_SIZE+1] = {0};
    q.push(start);
    while (!q.empty())
    {
        auto [x, y] = q.front();
        q.pop();
        if (x == target_row) return true;
        for (auto& [dx, dy] : vector<pair<int, int>>{{-2, 0}, {2, 0}, {0, -2}, {0, 2}})
        {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx > 2*Quoridor::ROW_SIZE || ny < 0 || ny > 2*Quoridor::COLUMN_SIZE) continue;
            if (board[x+(dx/2)][y+(dy/2)] == 1) continue; // wall
            if (isvisited[nx][ny]) continue;
            isvisited[nx][ny] = true;
            q.push({nx, ny});
        }
    }
    return false;
}

// BFS 最短步数 — 从玩家位置到目标行（绕开墙）
inline int min_distance_to_goal(const Quoridor::State& state, int player) {
    auto start = state.pos[player - 1];
    int target_row = (player == 1) ? (2 * Quoridor::ROW_SIZE - 1) : 1;

    if (start.first == target_row) return 0;

    bool visited[2 * Quoridor::ROW_SIZE + 1][2 * Quoridor::COLUMN_SIZE + 1] = {0};
    queue<pair<pair<int, int>, int>> q;
    q.push({start, 0});
    visited[start.first][start.second] = true;

    while (!q.empty()) {
        auto [pos, d] = q.front();
        q.pop();

        for (auto& [dr, dc] : vector<pair<int, int>>{{-2, 0}, {2, 0}, {0, -2}, {0, 2}}) {
            int nr = pos.first + dr, nc = pos.second + dc;
            if (nr < 0 || nr > 2 * Quoridor::ROW_SIZE || nc < 0 || nc > 2 * Quoridor::COLUMN_SIZE) continue;
            if (nr % 2 == 0 || nc % 2 == 0) continue;
            if (state.board[pos.first + dr / 2][pos.second + dc / 2]) continue;
            if (visited[nr][nc]) continue;
            if (nr == target_row) return d + 1;
            visited[nr][nc] = true;
            q.push({{nr, nc}, d + 1});
        }
    }
    return 999; // 不可达用大数代替 inf
}

#endif