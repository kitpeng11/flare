def read_all_frames(self, filename, nat, header=2, elem_type='xyz'):
    frames = []
    with open(self.restart_from+'/'+filename) as f:
        lines = f.readlines()
        frame_num = len(lines) // (nat+header)
        for i in range(frame_num):
            start = (nat+header) * i + header
            curr_frame = lines[start:start+nat]
            properties = []
            for line in curr_frame:
                line = line.split()
                if elem_type == 'xyz':
                    xyz = [float(l) for l in line[1:]]
                    properties.append(xyz)
                elif elem_type == 'int':
                    properties = [int(l) for l in line]
            frames.append(properties)
    return np.array(frames)


def read_frame(self, filename, frame_num):
    nat = len(self.atoms.positions)
    with open(self.restart_from+'/'+filename) as f:
        lines = f.readlines()
        if frame_num == -1: # read the last frame
            start_line = - (nat+2)
            frame = lines[start_line:]
        else:
            start_line = frame_num * (nat+2)
            end_line = (frame_num+1) * (nat+2)
            frame = f.lines[start_line:end_line]

        properties = []
        for line in frame[2:]:
            line = line.split()
            properties.append([float(d) for d in line[1:]])
    return np.array(properties), len(lines)//(nat+2)



