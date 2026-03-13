"""
Vor7reX Binary Image Ingestion Utility.
Low-level I/O operations for scanning directory structures and extracting 
multidimensional image matrices based on specific format filters.
"""

import os
import cv2

def readAllImg(path, *suffix):
    """
    Scans a directory and loads images into memory as a list of NumPy matrices.
    Index [0] of the returned list is the directory identifier (class name).
    """
    try:
        # Path validation to ensure filesystem integrity
        if not os.path.exists(path):
            print(f"I/O FAILURE: Path {path} is unreachable.")
            return []

        file_list = os.listdir(path)
        resultArray = []
        
        # Extract the terminal directory name as the identity label
        folderName = os.path.basename(path)
        resultArray.append(folderName)

        for i in file_list:
            # Enforce strict format compliance via suffix matching
            if endwith(i, *suffix):
                document_path = os.path.join(path, i)
                img = cv2.imread(document_path)
                
                if img is not None:
                    # Append validated matrix to the return buffer
                    resultArray.append(img)
                else:
                    print(f"RESOURCE WARNING: Unable to decode {i}")

    except Exception as e:
        print(f"RUNTIME EXCEPTION during ingestion: {e}")
        return []
    else:
        print(f"SUCCESS: Identity [{folderName}] buffer initialized.")
        return resultArray

def endwith(filename, *endstring):
    """
    Efficient case-insensitive suffix validation for file extensions.
    Supports multiple extension inputs via tuple mapping.
    """
    return filename.lower().endswith(tuple(ext.lower() for ext in endstring))

if __name__ == '__main__':
    # Local I/O integration test
    test_path = r"D:\myProject\pictures\jerry"
    result = readAllImg(test_path, '.jpg', '.png', '.pgm')
    
    if len(result) > 0:
        print(f"--- VOR7REX BUFFER STATUS ---")
        print(f"Subject Identifier: {result[0]}")
        print(f"Total Tensors Loaded: {len(result) - 1}")