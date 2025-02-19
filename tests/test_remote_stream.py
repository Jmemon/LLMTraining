"""
Test for remote_stream.py using a temporary file (with file:// URL).
"""

import os
import tempfile
from data.remote_stream import open_remote

def test_open_remote_with_local_file():
    # Create a temporary file and write some lines.
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write("testline1\ntestline2\n")
        temp_path = f.name
    url = f"file://{temp_path}"
    
    with open_remote(url) as f:
        lines = list(f)
    
    assert lines == ["testline1\n", "testline2\n"]
    os.remove(temp_path)

if __name__ == "__main__":
    test_open_remote_with_local_file()
    print("test_open_remote_with_local_file passed!") 