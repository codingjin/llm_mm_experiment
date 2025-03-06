import sys

def parse_and_compute(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    ops = None
    for line in lines:
        if line.startswith("Total OPs is "):
            try:
                ops = int(line.strip().split()[-1])
                break
            except (IndexError, ValueError) as e:
                raise ValueError(f"Malformed OPs line: {line.strip()}") from e
    if ops is None:
        raise ValueError("Missing 'Total OPs is' line in file")
    
    # Extract the 9 numbers for median calculation
    numbers = []
    capture = False
    for line in lines:
        if "Median of:" in line:
            capture = True
            continue
        if capture:
            stripped = line.strip()
            if stripped: # Skip empty lines
                try:
                    numbers.append(int(stripped))
                except ValueError as e:
                    raise ValueError(f"Non-numeric value found: '{stripped}'") from e
    
    if len(numbers) != 9:
        raise ValueError(f"Expected 9 time measurements, found {len(numbers)}")
    
    sorted_numbers = sorted(numbers)
    median = sorted_numbers[4] # the 5th is the median of 9 sorted numbers
    gflops_result = (ops * 0.001) / median

    return {
        "OPs" : ops,
        "MedianTime" : median,
        "GFLOPs" : round(gflops_result)
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gflops.py <input_file>")
        sys.exit(1)
    
    try:
        results = parse_and_compute(sys.argv[1])
        print(f"OPs: {results['OPs']}")
        print(f"MedianTime: {results['MedianTime']}")
        print(f"Performance(GFLOPs): {results['GFLOPs']}")
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

