WIDTH = 240
HEIGHT = 180

# Circle Param
SMALL_CIRCLE = [[0, 3], [1, 3], [2, 2], [3, 1],
                [3, 0], [3, -1], [2, -2], [1, -3],
                [0, -3], [-1, -3], [-2, -2], [-3, -1],
                [-3, 0], [-3, 1], [-2, 2], [-1, 3]]
BIG_CIRCLE = [[0, 4], [1, 4], [2, 3], [3, 2],
              [4, 1], [4, 0], [4, -1], [3, -2],
              [2, -3], [1, -4], [0, -4], [-1, -4],
              [-2, -3], [-3, -2], [-4, -1], [-4, 0],
              [-4, 1], [-3, 2], [-2, 3], [-1, 4]]

def isCornerEFast(img, centerX, centerY, pol):
    found = False
    image = img

    # if there is no event at the center, return False
    if image[centerY][centerX] <= 0:
        return False

    # Check if it's too close to the border of the SAE
    max_scale = 1
    cs = max_scale * 4
    if (centerX < cs or centerX >= WIDTH-cs or centerY < cs or centerY >= HEIGHT-cs):
        return False
    
    for i in range(16):
        for streak_size in range(3, 7):
            if image[centerY+SMALL_CIRCLE[i][1]][centerX+SMALL_CIRCLE[i][0]] < image[centerY+SMALL_CIRCLE[(i-1+16)%16][1]][centerX+SMALL_CIRCLE[(i-1+16)%16][0]]:
                continue

            if image[centerY+SMALL_CIRCLE[(i+streak_size-1)%16][1]][centerX+SMALL_CIRCLE[(i+streak_size-1)%16][0]] < image[centerY+SMALL_CIRCLE[(i+streak_size)%16][1]][centerX+SMALL_CIRCLE[(i+streak_size)%16][0]]:
                continue

            min_t = image[centerY + SMALL_CIRCLE[i][1]][centerX + SMALL_CIRCLE[i][0]]

            for j in range(1, streak_size):
                tj = image[centerY + SMALL_CIRCLE[(i+j)%16][1]][centerX + SMALL_CIRCLE[(i+j)%16][0]]
                if tj < min_t:
                    min_t = tj
            
            did_break = False
            for j in range(streak_size, 16):
                tj = image[centerY + SMALL_CIRCLE[(i+j)%16][1]][centerX + SMALL_CIRCLE[(i+j)%16][0]]
                if tj >= min_t:
                    did_break = True
                    break

            if not did_break:
                found = True
                break

        if found:
            break

    if found:
        found = False
        for i in range(20):
            for streak_size in range(4, 9):
                if image[centerY + BIG_CIRCLE[i][1]][centerX + BIG_CIRCLE[i][0]] < image[centerY + BIG_CIRCLE[(i-1+20)%20][1]][centerX + BIG_CIRCLE[(i-1+20)%20][0]]:
                    continue

                if image[centerY + BIG_CIRCLE[(i + streak_size - 1)%20][1]][centerX + BIG_CIRCLE[(i + streak_size - 1)%20][0]] < image[centerY + BIG_CIRCLE[(i+streak_size)%20][1]][centerX + BIG_CIRCLE[(i+streak_size)%20][0]]:
                    continue

                min_t = image[centerY + BIG_CIRCLE[i][1]][centerX + BIG_CIRCLE[i][0]]
                for j in range(1, streak_size):
                    tj = image[centerY + BIG_CIRCLE[(i+j)%20][1]][centerX + BIG_CIRCLE[(i+j)%20][0]]
                    if tj < min_t:
                        min_t = tj
                
                did_break = False
                for j in range(streak_size, 20):
                    tj = image[centerY + BIG_CIRCLE[(i+j)%20][1]][centerX + BIG_CIRCLE[(i+j)%20][0]]
                    if tj >= min_t:
                        did_break = True
                        break

                if not did_break:
                    found = True
                    break

            if found:
                break

    return found