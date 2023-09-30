from .events import Events

import os
import numpy as np
import sys


DECODE_DTYPES = {
    0: {'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'],
        'offsets': [0, 2, 4, 8], 'itemsize': 16},
    12: {'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'],
         'offsets': [0, 2, 4, 8], 'itemsize': 16},
    40: {'names': ['x', 'y', 'p', 't', 'vx', 'vy', 'center_x', 'center_y', 'id'],
         'formats': ['<u2', '<u2', '<i2', '<i8', 'f4', 'f4', 'f4', 'f4', 'u4'],
         'offsets': [0, 2, 4, 8, 16, 20, 24, 28, 32], 'itemsize': 36}
}

EV_TYPES = {
    0: [('t', 'u4'), ('_', 'i4')],
    12: [('t', 'u4'), ('_', 'i4')],
    40: [('t', 'u4'), ('_', 'i4'), ('vx', 'f4'), ('vy', 'f4'), ('center_x', 'f4'), ('center_y', 'f4'),
         ('id', 'u4')]
}

EV_STRINGS = {
    0: 'Event2D',
    12: 'EventCD',
    40: 'EventOpticalFlow'
}

X_MASK = 2**14 - 1  # 18 zeros followed by 14 ones when formulated as a binary number.
Y_MASK = 2**28 - 2**14  # 4 zeros, 14 ones and then 14 zeros.
P_MASK = 2 ** 29 - 2**28  # 3 zeros a one and 28 zeros.

def _cd_events_to_standard_format(events, height, width):
    p = events['p'].astype('int8')
    p[p == 0] = -1
    return Events(x=events['x'].astype('uint16'),
                  y=events['y'].astype('uint16'),
                  t=events['t'].astype('int64'),
                  p=p, height=height, width=width,
                  divider=1)

def stream_events(file_handle, buffer, dtype, ev_count=-1):
    """
    Streams data from opened file_handle.
    Args :
        file_handle: file object, needs to be opened.
        buffer (events numpy array): Pre-allocated buffer to fill with events
        dtype (numpy dtype):  expected fields
        ev_count (int): Number of events
    """
    dat = np.fromfile(file_handle, dtype=dtype, count=ev_count)
    count = len(dat)
    for name, _ in dtype:
        if name == '_':
            buffer['x'][:count] = np.bitwise_and(dat["_"], X_MASK)
            buffer['y'][:count] = np.right_shift(np.bitwise_and(dat["_"], Y_MASK), 14)
            buffer['p'][:count] = np.right_shift(np.bitwise_and(dat["_"], P_MASK), 28)
        else:
            buffer[name][:count] = dat[name]


def parse_header(f):
    """
    Parses the header of a DAT file and put the file cursor at the beginning of the binary data part.

    Args:
        f (file): File handle to a DAT file.

    Returns:
        int position of the file cursor after the header
        int type of event
        int size of event in bytes
        size (height, width) tuple of int or None
    """
    f.seek(0, os.SEEK_SET)
    bod = None
    end_of_header = False
    header = []
    num_comment_line = 0
    size = [None, None]
    # parse header
    while not end_of_header:
        bod = f.tell()
        line = f.readline()
        if sys.version_info > (3, 0):
            first_item = line.decode("latin-1")[:2]
        else:
            first_item = line[:2]

        if first_item != '% ':
            end_of_header = True
        else:
            words = line.split()
            if len(words) > 1:
                if words[1] == 'Date':
                    header += ['Date', words[2] + ' ' + words[3]]
                if words[1] == 'Height' or words[1] == b'Height':
                    size[0] = int(words[2])
                    header += ['Height', words[2]]
                if words[1] == 'Width' or words[1] == b'Width':
                    size[1] = int(words[2])
                    header += ['Width', words[2]]
            else:
                header += words[1:3]
            num_comment_line += 1
    # parse data
    f.seek(bod, os.SEEK_SET)

    if num_comment_line > 0:  # Ensure compatibility with previous files.
        # Read event type
        ev_type = np.frombuffer(f.read(1), dtype=np.uint8)[0]
        # Read event size
        ev_size = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
    else:
        ev_type = 0
        ev_size = sum([int(n[-1]) for _, n in EV_TYPES[ev_type]])

    bod = f.tell()
    return bod, ev_type, ev_size, size


class EventBaseReader(object):
    """
    EventBaseReader base class to pure python event readers.

    EventBaseReader allows reading a file of events while maintaining a position of the cursor.
    Further manipulations like advancing the cursor or going backward are allowed.

    Attributes:
        path (string): Path to the file being read
        current_time (int): Indicating the position of the cursor in the file in us
        duration_s (int): Indicating the total duration of the file in seconds

    Args:
        event_file (str): file containing events
    """

    def __init__(self, event_file):
        self._file = None
        self._start = None
        self.ev_type = None
        self._ev_size = None
        self._size = None
        self._dtype = None
        self._decode_dtype = None
        self.path = event_file
        self._extension = self.path.split('.')[-1]
        self.open_file()

        # size
        self._file.seek(0, os.SEEK_END)
        self._end = self._file.tell()
        self._ev_count = (self._end - self._start) // self._ev_size
        self.done = False
        self._file.seek(self._start)
        # If the current time is t, it means that next event that will be loaded has a
        # timestamp superior or equal to t (event with timestamp exactly t is not loaded yet)
        self.current_time = 0
        self.duration_s = self.total_time() * 1e-6

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def open_file(self):
        raise NotImplementedError()

    def reset(self):
        """Resets at beginning of file."""
        self._file.seek(self._start)
        self.done = False
        self.current_time = 0

    def event_count(self):
        """Getter on event_count.

        Returns:
            An int indicating the total number of events in the file
        """
        return self._ev_count

    def current_event_index(self):
        """Returns the number of event already loaded"""
        return (self._file.tell() - self._start) // self._ev_size

    def get_size(self):
        """Function returning the size of the imager which produced the events.

        Returns:
            Tuple of int (height, width) which might be (None, None)"""
        return self._size

    def is_done(self):
        """Returns True if the end of the file has been reached."""

        return self.done

    def __repr__(self):
        """String representation of a `DatReader` object.

        Returns:
            string describing the DatReader state and attributes
        """

        wrd = 'DatReader: {}\n'.format(self.path)
        wrd += '-----------\n'
        if self._extension == 'dat':
            wrd += 'Event Type: {}\n'.format(EV_STRINGS[self.ev_type])
        elif self._extension == 'npy':
            wrd += 'Event Type: numpy array element\n'
        wrd += 'Event Size: {} bytes\n'.format(self._ev_size)
        wrd += 'Event Count: {}\n'.format(self._ev_count)
        wrd += 'Duration: {} s \n'.format(self.duration_s)
        wrd += '-----------\n'
        return wrd

    def load_n_events(self, n_events):
        """
        Loads batch of n events.

        Args:
            n_events (int): Number of events that will be loaded

        Returns:
            events (numpy array): structured numpy array containing the events.

        Note that current time will be incremented to reach the timestamp of the first event not loaded yet.
        """
        event_buffer = np.empty((n_events + 1,), dtype=self._decode_dtype)

        pos = self._file.tell()
        count = (self._end - pos) // self._ev_size
        if n_events >= count:
            self.done = True
            n_events = count
            stream_events(self._file, event_buffer, self._dtype, n_events)
            self.current_time = event_buffer["t"][n_events - 1] + 1
        else:
            stream_events(self._file, event_buffer, self._dtype, n_events + 1)
            self.current_time = event_buffer["t"][n_events]
            self._file.seek(pos + n_events * self._ev_size)

        return event_buffer[:n_events]

    def load_delta_t(self, delta_t):
        """
        Loads events corresponding to a slice of time, starting from the DatReader's `current_time`.

        Args:
            delta_t (int): slice duration (in us).

        Returns:
            events (numpy array): structured numpy array containing the events.

        Note that current time will be incremented by `delta_t`.
        If an event is timestamped at exactly current_time it will not be loaded.
        """
        delta_t = int(delta_t)
        if delta_t < 1:
            raise ValueError("load_delta_t(): Delta_t must be at least 1 micro-second: {}".format(delta_t))

        if self.done or (self._file.tell() >= self._end):
            self.done = True
            return np.empty((0,), dtype=self._decode_dtype)

        expected_time = self.current_time + delta_t
        tmp_time = self.current_time
        start = self._file.tell()
        pos = start
        nevs = 0
        batch = 100000
        event_buffer = []
        # data is read by buffers until enough events are read or until the end of the file
        while tmp_time < expected_time and pos < self._end:
            count = (min(self._end, pos + batch * self._ev_size) - pos) // self._ev_size
            buffer = np.empty((count,), dtype=self._decode_dtype)
            stream_events(self._file, buffer, self._dtype, count)
            tmp_time = buffer["t"][-1]
            event_buffer.append(buffer)
            nevs += count
            pos = self._file.tell()
        if tmp_time >= expected_time:
            self.current_time = expected_time
        else:
            self.current_time = tmp_time + 1
        assert len(event_buffer) > 0
        idx = np.searchsorted(event_buffer[-1]["t"], expected_time)
        event_buffer[-1] = event_buffer[-1][:idx]
        event_buffer = np.concatenate(event_buffer)
        idx = len(event_buffer)
        self._file.seek(start + idx * self._ev_size)
        self.done = self._file.tell() >= self._end

        return event_buffer

    def load_mixed(self, n_events, delta_t):
        """
        Loads batch of n events or delta_t microseconds, whichever comes first.

        Args:
            n_events (int): Maximum number of events that will be loaded.
            delta_t (int): Maximum allowed slice duration (in us).

        Returns:
            events (numpy array): structured numpy array containing the events.

        Note that current time will be incremented to reach the timestamp of the first event not loaded yet.
        However if the maximal time slice duration is reached, current time will be increased by delta_t instead.
        """
        event_buffer = np.empty((n_events + 1,), dtype=self._decode_dtype)
        previous_time = self.current_time

        pos = self._file.tell()
        count = (self._end - pos) // self._ev_size
        if count <= n_events:
            self.done = True
            n_events = count
            stream_events(self._file, event_buffer, self._dtype, n_events)
            self.current_time = event_buffer["t"][n_events - 1] + 1
        else:
            stream_events(self._file, event_buffer, self._dtype, n_events + 1)
            self.current_time = event_buffer["t"][n_events]
            self._file.seek(pos + n_events * self._ev_size)

        # let's check is the delta_t condition already met
        if self.current_time - previous_time >= delta_t:
            # then we only need a subset of the events.
            index = np.searchsorted(event_buffer[:n_events]['t'], previous_time + delta_t)

            event_buffer = event_buffer[:index]
            self.current_time = previous_time + delta_t
            self._file.seek(pos + index * self._ev_size)

        return event_buffer[:n_events]

    def seek_event(self, n_events):
        """
        Seeks in the file by `n_events` events

        Args:
            n_events (int): seek in the file after n_events events

        Note that current time will be set to the timestamp of the next event.
        """
        if n_events <= 0:
            self._file.seek(self._start)
            self.current_time = 0
        elif n_events >= self._ev_count:
            # we put the cursor one event before and read the last event
            # which puts the file cursor at the right place
            # current_time is set to the last event timestamp + 1
            self._file.seek(self._start + (self._ev_count - 1) * self._ev_size)
            self.current_time = np.fromfile(self._file, dtype=self._dtype, count=1)["t"][0] + 1
        else:
            # we put the cursor at the *n_events*nth event
            self._file.seek(self._start + (n_events) * self._ev_size)
            # we read the timestamp of the following event (this change the position in the file)
            self.current_time = np.fromfile(self._file, dtype=self._dtype, count=1)["t"][0]
            # this is why we go back at the right position here
            self._file.seek(self._start + (n_events) * self._ev_size)
        self.done = self._file.tell() >= self._end

    def seek_time(self, expected_time, term_criterion=100000):
        """Goes to the time expected_time inside the file.
        This is implemented using a binary search algorithm.

        Args:
            expected_time (int): Expected time
            term_criterion (int): Binary search termination criterion in nb of events

        Once the binary search has found a buffer of size *term_criterion* events, containing the
        *expected_time*. It will load them in memory and perform a `searchsorted`_ from numpy, so that the end
        of the binary search doesn't take to many iterations in python.

        .. _searchsorted:
            https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
        """
        expected_time = int(expected_time)
        if expected_time > self.total_time():
            self._file.seek(self._end)
            self.done = True
            self.current_time = self.total_time() + 1
            return

        if expected_time <= 0:
            self.reset()
            return

        low = 0
        high = self._ev_count

        # binary search
        while high - low > term_criterion:
            middle = (low + high) // 2

            self.seek_event(middle)
            mid = np.fromfile(self._file, dtype=self._dtype, count=1)["t"][0]

            if mid > expected_time:
                high = middle
            elif mid < expected_time:
                low = middle + 1
            else:
                self.current_time = expected_time
                self.done = self._file.tell() >= self._end
                return
        # we now know that it is between low and high
        self.seek_event(low)
        final_buffer = np.fromfile(self._file, dtype=self._dtype, count=high - low)["t"]
        final_index = np.searchsorted(final_buffer, expected_time)

        self.seek_event(low + final_index)
        self.current_time = expected_time
        self.done = self._file.tell() >= self._end

    def total_time(self):
        """Returns total duration of video in us, providing there is no overflow.

        Returns:
            time (int): Duration of the file in us
        """
        if not self._ev_count:
            return 0
        # save the state of the class
        pos = self._file.tell()
        current_time = self.current_time
        done = self.done
        # read the last event's timestamp
        self.seek_event(self._ev_count - 1)
        time = np.fromfile(self._file, dtype=self._dtype, count=1)["t"][0]
        # restore the state
        self._file.seek(pos)
        self.current_time = current_time
        self.done = done

        return time

    def __del__(self):
        self._file.close()


class EventDatReader(EventBaseReader):
    """
    EventDatReader class to read DAT long files.
    DAT files are a binary format with events stored
    with polarity, x and y casted into a uint32 and timestamp on another uint32.
    This format still exists in many of our datasets, so this file is used to support it.

    Attributes:
        path (string): Path to the file being read
        current_time (int): Indicating the position of the cursor in the file in us
        duration_s (int): Indicating the total duration of the file in seconds

    Args:
        event_file (str): file containing events
    """

    def __init__(self, event_file):
        super().__init__(event_file)

    def open_file(self):
        assert self._extension == "dat", 'input file path = {}'.format(self.path)
        self._file = open(self.path, "rb")
        self._start, self.ev_type, self._ev_size, self._size = parse_header(self._file)
        assert self._ev_size != 0
        assert isinstance(self._ev_size, int)
        self._dtype = EV_TYPES[self.ev_type]
        self._decode_dtype = DECODE_DTYPES[self.ev_type]