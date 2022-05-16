from tqdm import tqdm

def compare_events(filename1):
    time, total = [], 0
    with open(filename1, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            time.append(int(line))
            total += 1
    
    return time, total

if __name__ == "__main__":
    time, total = compare_events("Time_Taken.txt")

    print(sum(time)/total)