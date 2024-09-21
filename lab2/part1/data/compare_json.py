import json
import math
import os
import sys

def compare_json_files(file1, file2, tolerance=1e-9):
    # Load JSON data from both files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # Check if both files have the same keys
    if set(data1.keys()) != set(data2.keys()):
        return False

    # Compare the content of both files
    for key in data1:
        list1 = data1[key]
        list2 = data2[key]

        # Ensure both lists have the same length
        if len(list1) != len(list2):
            return False

        # Compare each element in the list with a tolerance for floating-point differences
        for val1, val2 in zip(list1, list2):
            if not math.isclose(val1, val2, abs_tol=tolerance):
                return False

    return True

def compare_directories(dir1, dir2, tolerance=1e-9):
    # Get the list of JSON files from both directories
    files1 = {f for f in os.listdir(dir1) if f.endswith('.json')}
    files2 = {f for f in os.listdir(dir2) if f.endswith('.json')}

    # Ensure both directories have the same files
    if files1 != files2:
        print("The directories do not contain the same set of files.")
        return False

    # Compare corresponding files from both directories
    for filename in files1:
        file1 = os.path.join(dir1, filename)
        file2 = os.path.join(dir2, filename)

        if not compare_json_files(file1, file2, tolerance):
            print(f"Files {filename} in {dir1} and {dir2} are not equivalent.")
            return False

    print("All files in both directories are equivalent.")
    return True

if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python compare_json.py <dir1> <dir2>")
        sys.exit(1)

    dir1 = sys.argv[1]
    dir2 = sys.argv[2]

    compare_directories(dir1, dir2)