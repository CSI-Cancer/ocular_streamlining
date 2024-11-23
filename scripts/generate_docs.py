import subprocess

def main():
    subprocess.run(["pdoc", "--output-dir", "docs", "streamlining_training"], check=True)

if __name__ == "__main__":
    main()