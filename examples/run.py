import argparse
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Run NNI trial')
    parser.add_argument('--target', type=str, required=True, help='Target script to run')
    parser.add_argument('--enable_nni', action='store_true', help='Enable NNI')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--regression', action='store_true', help='Enable regression')
    parser.add_argument('--use_huber', action='store_true', help='Use Huber loss')
    parser.add_argument('--clip_grad', action='store_true', help='Clip gradients')
    args = parser.parse_args()

    # Set the PYTHONPATH to include the parent directory of 'examples'
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    os.environ['PYTHONPATH'] = parent_dir + os.pathsep + os.environ.get('PYTHONPATH', '')

    # Get the GPU index assigned by NNI
    gpu_index = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index

    # Construct the command to run the target script with the provided arguments
    target_module = args.target.replace('.py', '').replace('/', '.').replace('\\', '.')
    command = f'python -m examples.{target_module} --dataset {args.dataset}'
    if args.enable_nni:
        command += ' --enable_nni'
    if args.regression:
        command += ' --regression'
    if args.use_huber:
        command += ' --use_huber'
    if args.clip_grad:
        command += ' --clip_grad'

    # Print the constructed command for debugging
    print(f"Running command: {command}")

    # Run the command
    subprocess.run(command, shell=True)

if __name__ == '__main__':
    main()