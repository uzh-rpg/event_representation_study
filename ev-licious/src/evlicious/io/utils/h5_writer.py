from pathlib import Path
import weakref

import h5py

from .events import Events

def _compression_opts():
    compression_level = 1  # {0, ..., 9}
    shuffle = 2  # {0: none, 1: byte, 2: bit}
    # From https://github.com/Blosc/c-blosc/blob/7435f28dd08606bd51ab42b49b0e654547becac4/blosc/blosc.h#L66-L71
    # define BLOSC_BLOSCLZ   0
    # define BLOSC_LZ4       1
    # define BLOSC_LZ4HC     2
    # define BLOSC_SNAPPY    3
    # define BLOSC_ZLIB      4
    # define BLOSC_ZSTD      5
    compressor_type = 5
    compression_opts = (0, 0, 0, 0, compression_level, shuffle, compressor_type)
    return compression_opts

H5_BLOSC_COMPRESSION_FLAGS = dict(
    compression=32001,
    compression_opts=_compression_opts(), # Blosc
    chunks=True
)


class H5Writer:
    def __init__(self, outfile: Path):
        assert not outfile.exists(), str(outfile)
        self.h5f = h5py.File(str(outfile), 'w')
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        # create hdf5 datasets
        shape = (2**16,)
        maxshape = (None,)
        self.h5f.create_dataset('events/x', shape=shape, dtype='u2', maxshape=maxshape, **H5_BLOSC_COMPRESSION_FLAGS)
        self.h5f.create_dataset('events/y', shape=shape, dtype='u2', maxshape=maxshape, **H5_BLOSC_COMPRESSION_FLAGS)
        self.h5f.create_dataset('events/p', shape=shape, dtype='i1', maxshape=maxshape, **H5_BLOSC_COMPRESSION_FLAGS)
        self.h5f.create_dataset('events/t', shape=shape, dtype='i8', maxshape=maxshape, **H5_BLOSC_COMPRESSION_FLAGS)
        self.row_idx = 0
        self.first = True

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def add_data(self, events: Events):
        if self.first:
            self.h5f.create_dataset('events/width', data=events.width, dtype='i4')
            self.h5f.create_dataset('events/height', data=events.height, dtype='i4')
            self.h5f.create_dataset('events/divider', data=events.divider, dtype='i4')
            self.first = False

        current_size = len(events)
        new_size = self.row_idx + current_size
        self.h5f['events/x'].resize(new_size, axis=0)
        self.h5f['events/y'].resize(new_size, axis=0)
        self.h5f['events/p'].resize(new_size, axis=0)
        self.h5f['events/t'].resize(new_size, axis=0)

        self.h5f['events/x'][self.row_idx:new_size] = events._x
        self.h5f['events/y'][self.row_idx:new_size] = events._y
        self.h5f['events/p'][self.row_idx:new_size] = events.p
        self.h5f['events/t'][self.row_idx:new_size] = events.t
        
        self.row_idx = new_size