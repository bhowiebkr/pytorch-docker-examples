import argparse
import configparser
import os

config = configparser.ConfigParser()
config.read("config.ini")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="input folder", required=True)
    parser.add_argument("--output_path", help="output folder", required=True)
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path)

    # input path needs to exist to mount
    if not os.path.exists(input_path):
        raise (OSError(f"--input: {input_path} path does not exists."))

    # output path needs to exist to mount
    if not os.path.exists(output_path):
        raise (OSError(f"--output: {output_path} path does not exists."))

    # we want the output path to be empty
    elif not len(os.listdir(output_path)) == 0:
        raise (OSError(f"--output: {output_path} is not empty."))

    cmd = "sudo docker run --gpus all"  # we are using nvidia extension for docker so we include gpu info
    cmd += " --shm-size=16g"
    interactive = config.getboolean("DEFAULT", "Interactive")
    if interactive:
        cmd += " -it"
    cmd += ' --mount type=bind,src="$(pwd)/app",target=/app'
    cmd += f" --mount type=bind,src={input_path},target=/app/input"
    cmd += f" --mount type=bind,src={output_path},target=/app/output"
    cmd += (
        f" --mount type=bind,src={os.environ['HOME']}/.cache,target=/home/user/.cache"
    )
    cmd += " --workdir=/app"
    tag = config.get("DEFAULT", "DockerTag")
    cmd += " -e PYTHONUNBUFFERED=0"
    cmd += f" {tag}"

    if interactive:
        cmd += " bash"
    else:
        cmd += " python3 main.py"

    print(cmd, "\n")
    os.system(cmd)


if __name__ == "__main__":
    main()
