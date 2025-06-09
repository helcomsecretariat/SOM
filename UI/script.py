import sys
import time

def display_progress(completion: float, size: int = 20, text: str = 'Progress: '):
    x = int(size*completion)
    print(f"{text}[{"#"*x}{"."*(size-x)}] {completion*100} %", end='\r', flush=True)

def main(t: str = None, f: bool = False):
    print(f"Calling main() with parameters:\n\tt = {t}\n\tf = {f}", flush=True)
    size = 10
    display_progress(0)
    for i in range(size):
        time.sleep(0.5)
        display_progress((i+1) / size)
    print('')

if __name__ == "__main__":
    print("Starting script...", flush=True)
    print(f"sys.argv: {sys.argv}", flush=True)
    t = None
    f = False
    for i in range(len(sys.argv)):
        if sys.argv[i] in ['--text', '-t']:
            t = sys.argv[i+1]
        if sys.argv[i] in ['--flag', '-f']:
            f = True
    main(t=t, f=f)
    print("Terminating script...", flush=True)
