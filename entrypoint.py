import sys
import json
from train import train

def main():
    if len(sys.argv) != 2:
        print("[ERROR] Usage: %s job.json, where job.json is a suitable JSON file" % (sys.argv[0]))
        sys.exit(-1)

    job_cfg = json.load(open(sys.argv[1],"r"))  

    if isinstance(job_cfg, list) is False:
        job_cfg = [job_cfg]

    for job in job_cfg:
        print(job)
        train(**job)


if __name__=="__main__":
    main()
