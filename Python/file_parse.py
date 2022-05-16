from tqdm import tqdm

EFAST_CORNERS = "./Rotation_Absolute/eFast_Corners.txt"
ARCSTAR_CORNERS = "./Rotation_Absolute/arcStar_Corners.txt"
EFAST_CORNERS_QUANT = "./Timestamp_Quant/eFast_Corners.txt"
ARCSTAR_CORNERS_QUANT = "./Timestamp_Quant/arcStar_Corners.txt"

def compare_events(filename1, filename2):
    length = 0
    with open(filename1, 'r') as f:
        file1_lines = f.readlines()

    count, total = 0, 0
    with open(filename2, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            if line in file1_lines:
                count += 1
            total += 1
    
    return count, total

if __name__ == "__main__":
    eFast_count, eFast_total = compare_events(EFAST_CORNERS_QUANT, EFAST_CORNERS)
    arcStar_count, arcStar_total = compare_events(ARCSTAR_CORNERS_QUANT, ARCSTAR_CORNERS)

    print(eFast_count, eFast_total)
    print(arcStar_count, arcStar_total)