import subprocess
import matlab.engine

def run_python_script(filename):
    print(f"Running {filename}...")
    subprocess.run(["python", filename], check=True)

def run_matlab_script(filename, eng):
    print(f"Running {filename}...")
    eng.run(filename, nargout=0)

def main():
    try:
        # Start MATLAB engine
        eng = matlab.engine.start_matlab()

        # Run Python scripts
        run_python_script("gen_histogram.py")
        run_python_script("Unet_training.py")
        run_python_script("Pole_test2.py")
        run_python_script("cube_merger.py")

        # Run MATLAB scripts
        run_matlab_script("minkowskimergerd.m", eng)
        run_python_script("all_post_matlab_plots.python", eng)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {e.cmd}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'eng' in locals():
            eng.quit()

if __name__ == "__main__":
    main()
