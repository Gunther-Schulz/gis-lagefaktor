import runpy
import sys

if __name__ == "__main__":
    # Create a new list of arguments, keeping the script name and adding the rest of the command line arguments
    args = [sys.argv[0]] + sys.argv[1:]
    runpy.run_module("gis_lagefaktor.main",
                     run_name="__main__", alter_sys=True)
