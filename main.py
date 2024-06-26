from reconstruct import Finder
import time
from reconstruct.data.loader import Loader


#  -------------------  MAIN DRIVER  -------------------  #

def main():

    time_now = time.time()

    items = ["Hit_x", "Hit_z", "Hit_y", "Hit_time"]
    path = "./simdata/muons_32pT/20231204/203837/run0.root"  # ./simdata/muons_28pT/20231120/164248/run0.root, ./simdata/muons_34pT/20231204/203837/run0.root, ./simdata/muons_27pT/20231116/051933/run0.root, ./simdata/muons_50p_mag/20231106/034221/run0.root
    sim_data = Loader(path, items)

    print("\n[ METADATA ]\n")

    for name, value in sim_data.metadata.items():
        print("{}: {}".format(name.ljust(20), value))

    print("\n")

    transform = Finder(sim_data)
    transform.run()

    print("\n--- Complete in {:.4f} seconds ---\n".format(time.time() - time_now))


if __name__ == "__main__":
    main()
