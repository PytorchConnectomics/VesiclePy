import os
import numpy as np
import glob
import h5py


def main():
    PATH = "/projects/weilab/dataset/hydra/results/"
    files = sorted(
        glob.glob(
            os.path.join(
                PATH,
                "vesicle_small_*_30-8-8_patch.h5",
            )
        )
    )
    bbs = sorted(
        glob.glob(
            os.path.join(
                PATH,
                "vesicle_small-bbs_*_30-8-8.h5",
            )
        )
    )
    assert len(files) == len(bbs)
    for file, bb in zip(files, bbs):
        assert file.split("_")[2] == bb.split("_")[2]
    print(f"Found {len(files)} files")
    patches = []
    ids = []
    for i, file in enumerate(files):
        neuron_id = file.split("_")[2]
        file = h5py.File(file, "r")
        vol = file["key0"][:]
        assert vol.shape[-2:] == (11, 11)
        assert vol.shape[1] == 1
        patches.extend(vol.squeeze(1))

        bb = h5py.File(bbs[i], "r")["main"][:]
        assert bb.shape[1] == 7  # ie first is ID
        assert bb.shape[0] == vol.shape[0]

        ids.extend([(neuron_id, j) for j in bb[:, 0]])

    assert all([np.sum(x) > 0 for x in patches])
    patches = np.stack(patches, axis=0)
    print(patches.shape)
    np.savez("patches.npz", patches=patches, ids=ids)


if __name__ == "__main__":
    main()
