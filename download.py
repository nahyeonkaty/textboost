#!/usr/bin/env python3
import argparse
import os

import gdown

DB = {

}

TI = {
    "cat_statue/2.jpeg": "13MHHN99hWVf4_BF6NVXoxPUjtobGW2Rd",
    "clock/1.jpeg": "1hbysyK688nagfNHaunbfwGmkpLDSW11l",
    "colorful_teapot/1.jpeg": "1A2kbBFoCNIK6DjnBCU1fVUUTclchIs3S",
    "elephant/3.jpg": "1xM43EM1D6T9esQwNfne0C3ZdIa63hua2",
    "mug_skulls/3.jpeg": "1--AY_FbK0_VP1sjgFzCDCPKpM9rhiu1L",
    "physics_mug/3.jpeg": "1vaRwcNASmxx62VPJmsPH_qPjQDrDmmXK",
    "red_teapot/1.jpeg": "11UNMcWroD9b4y1npxKrCUsVeLEmvajvC",
    "round_bird/4.jpg": "1e0KpoKiCa0kcqMpe7Mu5gjh-IcRy6R7_",
    "thin_bird/4.jpeg": "1-iZ_VEu4IQDOv0ywRLXab-z-1WlP5h1G",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=".")
    return parser.parse_args()


def main(args):
    for key, id in TI.items():
        folder = key.split("/")[0]
        dst = os.path.join(args.out_dir, key)
        os.makedirs(os.path.join(args.out_dir, folder), exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={id}", dst, quiet=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
