import sys
import os

try:
    import QES
    print(f"QES imported from: {QES.__file__}")
    print(f"QES version: {QES.__version__}")

    print("Attempting: from QES.general_python import algebra")
    from QES.general_python import algebra
    print("Success!")

except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
