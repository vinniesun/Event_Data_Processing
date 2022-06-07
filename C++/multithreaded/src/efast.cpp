#include <iostream>
#include <vector>
#include "efast.hpp"

int circle3_[16][2] = { {0, 3}, {1, 3}, {2, 2}, {3, 1},
                        {3, 0}, {3, -1}, {2, -2}, {1, -3},
                        {0, -3}, {-1, -3}, {-2, -2}, {-3, -1},
                        {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3} };
int circle4_[20][2] = { {0, 4}, {1, 4}, {2, 3}, {3, 2},
                        {4, 1}, {4, 0}, {4, -1}, {3, -2},
                        {2, -3}, {1, -4}, {0, -4}, {-1, -4},
                        {-2, -3}, {-3, -2}, {-4, -1}, {-4, 0},
                        {-4, 1}, {-3, 2}, {-2, 3}, {-1, 4} };

bool eFast(std::vector<std::vector<int>> sae_, int x, int y, int t, bool p) {
    const int max_scale = 1;

    // only check if not too close to border
    const int cs = max_scale*4;
    if (x < cs || x >= 240-cs || y < cs || y >= 180-cs)
    {
        return false;
    }

    bool found_streak = false;

    for (int i=0; i<16; i++)
    {
        for (int streak_size = 3; streak_size<=6; streak_size++)
        {
            // check that streak event is larger than neighbor
            if (sae_[y+circle3_[i][1]][x+circle3_[i][0]] <  sae_[y+circle3_[(i-1+16)%16][1]][x+circle3_[(i-1+16)%16][0]])
                continue;

            // check that streak event is larger than neighbor
            if (sae_[y+circle3_[(i+streak_size-1)%16][1]][x+circle3_[(i+streak_size-1)%16][0]] < sae_[y+circle3_[(i+streak_size)%16][1]][x+circle3_[(i+streak_size)%16][0]])
                continue;

            int min_t = sae_[y+circle3_[i][1]][x+circle3_[i][0]];
            for (int j=1; j<streak_size; j++)
            {
                const int tj = sae_[y+circle3_[(i+j)%16][1]][x+circle3_[(i+j)%16][0]];
                if (tj < min_t)
                    min_t = tj;
            }

            bool did_break = false;
            for (int j=streak_size; j<16; j++)
            {
                const int tj = sae_[y+circle3_[(i+j)%16][1]][x+circle3_[(i+j)%16][0]];

                if (tj >= min_t)
                {
                    did_break = true;
                    break;
                }
            }

            if (!did_break)
            {
                found_streak = true;
                break;
            }

        }
        if (found_streak)
        {
            break;
        }
    }

    if (found_streak)
    {
        found_streak = false;
        for (int i=0; i<20; i++)
        {
            for (int streak_size = 4; streak_size<=8; streak_size++)
            {
                // check that first event is larger than neighbor
                if (sae_[y+circle4_[i][1]][x+circle4_[i][0]] <  sae_[y+circle4_[(i-1+20)%20][1]][x+circle4_[(i-1+20)%20][0]])
                    continue;

                // check that streak event is larger than neighbor
                if (sae_[y+circle4_[(i+streak_size-1)%20][1]][x+circle4_[(i+streak_size-1)%20][0]] < sae_[y+circle4_[(i+streak_size)%20][1]][x+circle4_[(i+streak_size)%20][0]])
                    continue;

                int min_t = sae_[y+circle4_[i][1]][x+circle4_[i][0]];
                for (int j=1; j<streak_size; j++)
                {
                    const int tj = sae_[y+circle4_[(i+j)%20][1]][x+circle4_[(i+j)%20][0]];
                    if (tj < min_t)
                        min_t = tj;
                }

                bool did_break = false;
                for (int j=streak_size; j<20; j++)
                {
                    const int tj = sae_[y+circle4_[(i+j)%20][1]][x+circle4_[(i+j)%20][0]];
                    if (tj >= min_t)
                    {
                        did_break = true;
                        break;
                    }
                }

                if (!did_break)
                {
                    found_streak = true;
                    break;
                }
            }
            if (found_streak)
            {
                break;
            }
        }
    }

    return found_streak;
}