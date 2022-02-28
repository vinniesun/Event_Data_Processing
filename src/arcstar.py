# The HEIGHT and WIDTH of a DAVIS Camera
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

def isCornerArcStar(img, prev_state, prev_state_inv, centerX, centerY, pol, filter_threshold=0.05):
    if pol == 0:
        pol_inv = 1
    else:
        pol_inv = 0

    t_last = prev_state
    t_last_inv = prev_state_inv

    # Filter out redundant spikes, e.g. spikes of the same polarity that's fired off consecutively in short period
    if ((img[centerY][centerX] > t_last + filter_threshold) or (t_last_inv > t_last)):
        t_last = img[centerY][centerX]
    else:
        t_last = img[centerY][centerX]
        return False

    # Check if it's too close to the border of the SAE
    max_scale = 1
    cs = max_scale * 4
    if (centerX < cs or centerX >= WIDTH-cs or centerY < cs or centerY >= HEIGHT-cs):
        return False

    found = False
    image = img
    segment_new_min_t = image[centerY + SMALL_CIRCLE[0][1]][centerX + SMALL_CIRCLE[0][0]]
    arc_left_idx, arc_right_idx = 0, 0 # this is the CCW & CW index in the original paper

    # if there is no event at the center, return False
    if image[centerY][centerX] <= 0:
        return False
    
    for i in range(1, len(SMALL_CIRCLE)):
        t = image[centerY + SMALL_CIRCLE[i][1]][centerX + SMALL_CIRCLE[i][0]]
        if t > segment_new_min_t:
            segment_new_min_t = t
            arc_right_idx = i
    
    arc_left_idx = (arc_right_idx - 1 + len(SMALL_CIRCLE))%len(SMALL_CIRCLE)
    arc_right_idx = (arc_right_idx + 1)%len(SMALL_CIRCLE)

    arc_left_val = image[centerY + SMALL_CIRCLE[arc_left_idx][1]][centerX + SMALL_CIRCLE[arc_left_idx][0]]
    arc_right_val = image[centerY + SMALL_CIRCLE[arc_right_idx][1]][centerX + SMALL_CIRCLE[arc_right_idx][0]]
    arc_left_min_t = arc_left_val
    arc_right_min_t = arc_right_val

    for j in range(0, 3): # 3 is the smallest segment length of an acceptable arc
        if arc_right_val > arc_left_val:
            if arc_right_min_t < segment_new_min_t:
                segment_new_min_t = arc_right_min_t
            
            arc_right_idx = (arc_right_idx + 1)%len(SMALL_CIRCLE)
            arc_right_val = image[centerY + SMALL_CIRCLE[arc_right_idx][1]][centerX + SMALL_CIRCLE[arc_right_idx][0]]
            if arc_right_val < arc_right_min_t:
                arc_right_min_t = arc_right_val

        else:
            if arc_left_min_t < segment_new_min_t:
                segment_new_min_t = arc_left_min_t

            arc_left_idx = (arc_left_idx - 1 + len(SMALL_CIRCLE))%len(SMALL_CIRCLE)
            arc_left_val = image[centerY + SMALL_CIRCLE[arc_left_idx][1]][centerX + SMALL_CIRCLE[arc_left_idx][0]]
            if arc_left_val < arc_left_min_t:
                arc_left_min_t = arc_left_val
    
    newest_segment_size = 3

    for j in range(3, len(SMALL_CIRCLE)): # look through the rest of the circle
        if arc_right_val > arc_left_val:
            if arc_right_val >= segment_new_min_t:
                newest_segment_size = j+1
                if arc_right_min_t < segment_new_min_t:
                    segment_new_min_t = arc_right_min_t
        
            arc_right_idx = (arc_right_idx+1)%len(SMALL_CIRCLE)
            arc_right_val = image[centerY + SMALL_CIRCLE[arc_right_idx][1]][centerX + SMALL_CIRCLE[arc_right_idx][0]]
            if arc_right_val < arc_right_min_t:
                arc_right_min_t = arc_right_val

        else:
            if arc_left_val >= segment_new_min_t:
                newest_segment_size = j+1
                if arc_left_min_t < segment_new_min_t:
                    segment_new_min_t = arc_left_min_t

            arc_left_idx = (arc_left_idx - 1 + len(SMALL_CIRCLE))%len(SMALL_CIRCLE)
            arc_left_val = image[centerY + SMALL_CIRCLE[arc_left_idx][1]][centerX + SMALL_CIRCLE[arc_left_idx][0]]
            if arc_left_val < arc_left_min_t:
                arc_left_min_t = arc_left_val

    if ((newest_segment_size <= 6) or (newest_segment_size >= len(SMALL_CIRCLE) - 6) and (newest_segment_size <= (len(SMALL_CIRCLE) - 3))): # Check the arc size satisfy the requirement
        found = True
    
    # Search through the large circle if small circle verifies
    if found:
        found = False
        segment_new_min_t = image[centerY + BIG_CIRCLE[0][1]][centerX + BIG_CIRCLE[0][0]]
        arc_right_idx = 0

        for i in range(1, len(BIG_CIRCLE)):
            t = image[centerY + BIG_CIRCLE[i][1]][centerX + BIG_CIRCLE[i][0]]
            if t > segment_new_min_t:
                segment_new_min_t = t
                arc_right_idx = i

        arc_left_idx = (arc_right_idx - 1 + len(BIG_CIRCLE))%len(BIG_CIRCLE)
        arc_right_idx = (arc_right_idx + 1)%len(BIG_CIRCLE)
        arc_left_val = image[centerY + BIG_CIRCLE[arc_left_idx][1]][centerX + BIG_CIRCLE[arc_left_idx][0]]
        arc_right_val = image[centerY + BIG_CIRCLE[arc_right_idx][1]][centerX + BIG_CIRCLE[arc_right_idx][0]]

        arc_left_min_t = arc_left_val
        arc_right_min_t = arc_right_val

        for j in range(1, 4):
            if (arc_right_val > arc_left_val):
                if (arc_right_min_t > arc_left_min_t):
                    segment_new_min_t
                arc_right_idx = (arc_right_idx+1)%len(BIG_CIRCLE)
                arc_right_val = image[centerY + BIG_CIRCLE[arc_right_idx][1]][centerX + BIG_CIRCLE[arc_right_idx][0]]

                if arc_right_val < arc_right_min_t:
                    arc_right_min_t = arc_right_val
            else:
                if arc_left_min_t < segment_new_min_t:
                    segment_new_min_t = arc_left_min_t
                arc_left_idx = (arc_left_idx - 1 + len(BIG_CIRCLE))%len(BIG_CIRCLE)
                arc_left_val = image[centerY + BIG_CIRCLE[arc_left_idx][1]][centerX + BIG_CIRCLE[arc_left_idx][0]]
                if arc_left_val < arc_left_min_t:
                    arc_left_min_t = arc_left_val

        newest_segment_size = 4

        for j in range(4, 8):
            if arc_right_val > arc_left_val:
                if arc_right_val >= segment_new_min_t:
                    newest_segment_size = j+1
                    if arc_right_min_t < segment_new_min_t:
                        segment_new_min_t = arc_right_min_t

                arc_right_idx = (arc_right_idx + 1)%len(BIG_CIRCLE)
                arc_right_val = image[centerY + BIG_CIRCLE[arc_right_idx][1]][centerX + BIG_CIRCLE[arc_right_idx][0]]
                if arc_right_val < arc_right_min_t:
                    arc_right_min_t = arc_right_val
            
            else:
                if arc_left_val >= segment_new_min_t:
                    newest_segment_size = j+1
                    if arc_left_min_t < segment_new_min_t:
                        segment_new_min_t = arc_left_min_t
                    
                arc_left_idx = (arc_left_idx - 1 + len(BIG_CIRCLE))%len(BIG_CIRCLE)
                arc_left_val = image[centerY + BIG_CIRCLE[arc_left_idx][1]][centerX + BIG_CIRCLE[arc_left_idx][0]]
                if arc_left_val < arc_left_min_t:
                    arc_left_min_t = arc_left_val

            if ((newest_segment_size <= 8) or ((newest_segment_size >= (len(BIG_CIRCLE) - 8)) and (newest_segment_size <= (len(BIG_CIRCLE) - 4)))):
                found = True

    return found
