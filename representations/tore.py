# author: Nikola Zubic

import numpy as np


def events2ToreFeature(x, y, ts, pol, sampleTimes, k, frameSize):
    # print('Starting feature generation...')

    oldPosTore, oldNegTore = np.inf * np.ones(
        (frameSize[0], frameSize[1], 2 * k)
    ), np.inf * np.ones((frameSize[0], frameSize[1], 2 * k))
    Xtore = np.zeros((frameSize[0], frameSize[1], 2 * k, 1), dtype=np.float32)

    priorSampleTime = -np.inf

    for sampleLoop, currentSampleTime in enumerate([sampleTimes]):
        addEventIdx = np.logical_and(ts >= priorSampleTime, ts < currentSampleTime)

        p = np.logical_and(addEventIdx, pol > 0)
        px, py, pts_diff = x[p], y[p], currentSampleTime - ts[p]

        newPosTore = np.full(frameSize + (k,), np.inf)
        for t, i, j in zip(pts_diff, py, px):
            try:
                newPosTore[i - 1, j - 1] = np.partition(
                    np.concatenate(([t], newPosTore[i - 1, j - 1, : k - 1])), k - 1
                )[:k]
            except:
                i, j = int(i), int(j)
                newPosTore[i - 1, j - 1] = np.partition(
                    np.concatenate(([t], newPosTore[i - 1, j - 1, : k - 1])), k - 1
                )[:k]

        p = np.logical_and(addEventIdx, pol <= 0)
        px, py, pts_diff = x[p], y[p], currentSampleTime - ts[p]

        newNegTore = np.full(frameSize + (k,), np.inf)
        for t, i, j in zip(pts_diff, py, px):
            try:
                newNegTore[i - 1, j - 1] = np.partition(
                    np.concatenate(([t], newNegTore[i - 1, j - 1, : k - 1])), k - 1
                )[:k]
            except:
                i, j = int(i), int(j)
                newNegTore[i - 1, j - 1] = np.partition(
                    np.concatenate(([t], newNegTore[i - 1, j - 1, : k - 1])), k - 1
                )[:k]

        oldPosTore += currentSampleTime - priorSampleTime
        oldPosTore = (
            np.dstack([oldPosTore[..., :k], newPosTore])
            .reshape(frameSize[0], frameSize[1], 2, -1)
            .min(axis=2)
        )

        oldNegTore += currentSampleTime - priorSampleTime
        oldNegTore = (
            np.dstack([oldNegTore[..., :k], newNegTore])
            .reshape(frameSize[0], frameSize[1], 2, -1)
            .min(axis=2)
        )

        Xtore[:, :, :, sampleLoop] = np.concatenate(
            [oldPosTore, oldNegTore], axis=2
        ).astype(np.float32)

        priorSampleTime = currentSampleTime

    minTime = 150
    maxTime = 500e6

    for loop in range(Xtore.shape[3]):
        tmp = Xtore[:, :, :, loop]
        tmp[np.isnan(tmp)] = maxTime
        tmp[tmp > maxTime] = maxTime
        tmp = np.log(tmp + 1)
        tmp -= np.log(minTime + 1)
        tmp[tmp < 0] = 0
        Xtore[:, :, :, loop] = tmp

    return Xtore[
        ..., 0
    ]  # (width, height, number of channels, 1) to (width, height, number of channels)


def read_dataset(filename):
    """Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset file specified by 'filename'"""
    f = open(filename, "rb")
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint32(raw_data)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    # Process time stamp overflow events
    time_increment = 2**13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    # Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    td = {}

    td["x"] = all_x[td_indices]
    td["y"] = all_y[td_indices]
    td["ts"] = all_ts[td_indices]
    td["p"] = all_p[td_indices]
    return td


if __name__ == "__main__":
    item = read_dataset("/data/storage/nzubic/01499.bin")
    x = item["x"]
    y = item["y"]
    ts = item["ts"]
    pol = item["p"]

    p = pol - 1

    x = x - min(x) + 1
    y = y - min(y) + 1
    ts = ts - min(ts)

    sampleTimes = ts[-1]
    frameSize = (max(y), max(x))

    Xtore = events2ToreFeature(x, y, ts, p, sampleTimes, 3, frameSize)
    exit(0)
