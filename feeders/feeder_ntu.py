import numpy as np
import sys

from torch.utils.data import Dataset

from feeders import tools
import scipy as sp
from scipy.spatial.transform import Rotation as R, Slerp

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, random_miss=False, structured_degradation=False, random_degradation_type=None,
                  structured_degradation_type=None, random_miss_amount=0, decimation_frequency=1, chunks=1, FPS=30,
                  mitigation=False, spatial_deg=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        :param random_miss: if true, apply random dropout
        :param random_degradation_type: type of random degradation ('delete', 'interpolate', 'next_frame'). 
        :param random_miss_amount: amount of random dropout (between 0 and 1 for the fraction of joints to drop)
        :param structured_degradation: if true, apply structured degradation
        :param structured_degradation_type: type of structured degradation ('frame_rate', 'reduced_resolution')
        :param decimation_frequency: frequency of decimation for 'reduced_resolution' degradation (e.g., 2 means keep every 2nd frame)
        :param FPS: target frame rate for 'frame_rate' degradation (e.g., 15 means downsample to 15 FPS) Default is 30 FPS
        :param chunks: number of chunks to divide the sequence into for 'frame_rate' degradation. (currently only supports 1)
        :param mitigation: if true, apply simple linear interpolation to mitigate missing data
        :param spatial_deg: if true, apply spatial degradation of the right hand using SLERP, Spherical Linear Interpolation. 
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.random_miss = random_miss
        self.structured_degradation = structured_degradation
        self.random_degradation_type = random_degradation_type
        self.structured_degradation_type = structured_degradation_type
        self.random_miss_amount = random_miss_amount
        self.decimation_frequency = decimation_frequency
        self.chunks = chunks
        self.FPS = FPS
        self.mitigation = mitigation
        self.spatial_deg = spatial_deg

        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM

        if index == 0:
            print(' ')
            print('Random Degradation: ', self.random_miss)
            print('Structured Degradation: ', self.structured_degradation)
            print('Spatial Degradation: ', self.spatial_deg)

        if (self.random_miss or self.structured_degradation) and self.spatial_deg:
            sys.exit("Cannot have both temporal and spatial degradation")

        if self.random_miss and self.structured_degradation:
            sys.exit("Cannot have both random and structured degradation")
        
        if self.random_miss:
            data_numpy = self.get_random_degradation(data_numpy, index, valid_frame_num)
        elif self.structured_degradation:
            data_numpy = self.get_structured_degradation(data_numpy, index, valid_frame_num)

        if self.spatial_deg:
            data_numpy = self.apply_spatial_degradation(data_numpy, valid_frame_num)

        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        #statistics = self.get_statistics(data_numpy, valid_frame_num)

        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index#, np.array(statistics)

    def apply_spatial_degradation(self, skel_data, no_of_frames):

        output = skel_data.copy()

        skel_data_right_hand = skel_data[:, :, [4,5,6,7,21], 0].copy()

        joint_elbow = skel_data_right_hand[:,:,0]
        joint_wrist = skel_data_right_hand[:,:,1]
        joint_hand = skel_data_right_hand[:,:,2]
        joint_tip_of_hand = skel_data_right_hand[:,:,3]
        joint_thumb = skel_data_right_hand[:,:,4]

        FPS_drop = self.FPS / 30
        chunk_length = int(no_of_frames-no_of_frames*FPS_drop)
        #print('Chunk length:', chunk_length)
        #print('No of frames:', no_of_frames)
        if chunk_length < 1:
            return skel_data
        max_chunk_start = int(no_of_frames - chunk_length)
        chunk_start = np.random.randint(0, high=max_chunk_start) # high is exlusive
        chunk_end = chunk_start + chunk_length

        elbow_to_wrist_interp = self.get_sphereical_linear_interpolation(joint_elbow, joint_wrist, chunk_start, chunk_end)
        wrist_to_hand_interp = self.get_sphereical_linear_interpolation(elbow_to_wrist_interp, joint_hand, chunk_start, chunk_end)
        wrist_to_thumb_interp = self.get_sphereical_linear_interpolation(elbow_to_wrist_interp, joint_thumb, chunk_start, chunk_end)
        hand_to_tip_of_hand_interp = self.get_sphereical_linear_interpolation(wrist_to_hand_interp, joint_tip_of_hand, chunk_start, chunk_end)

        skel_data_right_hand[:, :, 1] = elbow_to_wrist_interp
        skel_data_right_hand[:, :, 2] = wrist_to_hand_interp
        skel_data_right_hand[:, :, 3] = hand_to_tip_of_hand_interp
        skel_data_right_hand[:, :, 4] = wrist_to_thumb_interp

        output[:, :, [4,5,6,7,21], 0] = skel_data_right_hand

        return output

    def get_sphereical_linear_interpolation(self, joint1, joint2, start_time, end_time):

        # Calculate the vectors from joint1 to joint2
        bone_start = joint2[:, start_time] - joint1[:, start_time] 
        bone_end = joint2[:, end_time] - joint1[:, end_time]

        if bone_end.all() == 0 or bone_start.all() == 0:
            return joint2

        num_frames = end_time - start_time

        start_unit_vector = bone_start / np.linalg.norm(bone_start)
        end_unit_vector = bone_end / np.linalg.norm(bone_end)

        rotation = R.align_vectors([end_unit_vector], [start_unit_vector])[0]
        key_rotations = R.concatenate([R.identity(), rotation])
        key_times = [0, 1]

        time = np.linspace(0, 1, num_frames)
        slerp_rotations = Slerp(key_times, key_rotations)
        interpolated_rotations = slerp_rotations(time)

        interp_bone_start = interpolated_rotations.apply(bone_start)
        interp_joint2 = joint1[:, start_time:end_time] + interp_bone_start.T

        interp_joint2_all_t = joint2.copy()
        interp_joint2_all_t[:, start_time:end_time] = interp_joint2

        return interp_joint2_all_t

    def get_statistics(self, data_numpy, valid_frame_num):

        left_joints = [8,9,10,11,23,24,16,17,18,19]
        right_joints = [4,5,6,7,21,22,12,13,14,15]
        spine_joints = 1

        # channels, frames, joints, num_people

        skeleton_data = data_numpy[:,:valid_frame_num,:,0]

        metrics = []

        # Joint involvmenet index
        joint_movements = np.linalg.norm(np.diff(skeleton_data, axis=1), axis=0)
        joint_activity = np.mean(joint_movements, axis=0)
        movement_threshold = 0.2
        active_joints = np.sum(joint_activity > movement_threshold)
        metrics.append(active_joints)

        # center of mass
        com = np.mean(skeleton_data, axis=2).T
        com_shift = np.linalg.norm(com[-1]-com[0])
        metrics.append(com_shift)

        # range of motion
        joint_ranges = np.ptp(skeleton_data, axis=2).T
        joint_rom = np.mean(np.linalg.norm(joint_ranges, axis=1))
        metrics.append(joint_rom)

        # symmetric score
        left_motion = joint_movements[:,left_joints]
        right_motion = joint_movements[:,right_joints]
        min_len = min(left_motion.shape[1], right_motion.shape[1])
        sym_diff = np.abs(left_motion[:,:min_len] - right_motion[:,:min_len])
        sym_score = 1.0 - np.mean(sym_diff)/(np.mean(joint_movements)+1e-6)
        sym_score = np.clip(sym_score, 0, 1)
        metrics.append(sym_score)

        # postural disruption score
        spine_angles = np.diff(skeleton_data[:,:,spine_joints], axis=1).T
        spine_velocity = np.linalg.norm(spine_angles, axis=1)
        pds = np.mean(spine_velocity)
        metrics.append(pds)
        
        return metrics

    def get_random_degradation(self, data_numpy, index, no_of_frames):

        if index == 0:
            print('Dropping random frames')
            print('Using random degradation type: ', self.random_degradation_type)
            print('Dropping ', self.random_miss_amount*100, '% of frames')

        if self.random_degradation_type == 'delete':
            data_numpy = self.get_delete(data_numpy, index, no_of_frames)
        elif self.random_degradation_type == 'interpolate':
            data_numpy = self.get_interpolate(data_numpy, index, no_of_frames)
        elif self.random_degradation_type == 'next_frame':
            data_numpy = self.drop_next_frame(data_numpy, index, no_of_frames)
        else:
            sys.exit("Random degradation type not recognized")
        
        return data_numpy

    def get_structured_degradation(self, data_numpy, index, no_of_frames):

        if index == 0:
            print('Using structured degradation type: ', self.structured_degradation_type)

        if self.structured_degradation_type == 'reduced_resolution':
            data_numpy = self.apply_decimate(data_numpy, index, no_of_frames)
        elif self.structured_degradation_type == 'frame_rate':
            data_numpy = self.reduce_frame_rate(data_numpy, index, no_of_frames)
        else:
            sys.exit("Structured degradation type not recognized")
        
        return data_numpy
    
    def get_delete(self, data_numpy, index, no_of_frames):

        channels, frames, joints, num_people = data_numpy.shape
        t_index = np.arange(no_of_frames)
        frames_to_drop = int(self.random_miss_amount*no_of_frames)
        filter = np.random.choice(t_index, size=frames_to_drop, replace=False)
        indices = np.argwhere(np.isin(t_index, filter))
        valid_t_index = np.delete(t_index, indices)
        arr = data_numpy[:, valid_t_index, :, :]
        output = np.zeros((channels, frames, joints, num_people), dtype=np.float32)
        new_size = no_of_frames - frames_to_drop
        output[:, :new_size, :, :] = arr

        return output

    def get_interpolate(self, data_numpy, index, no_of_frames):

        channels, frames, joints, num_people = data_numpy.shape
        t_index = np.arange(no_of_frames)
        t_index = t_index[1:-1]

        drop_amount = int(self.random_miss_amount*len(t_index)+2)
        expected_drop = int(self.random_miss_amount*no_of_frames)

        if drop_amount > expected_drop:
            amount_to_drop = expected_drop
        else:
            amount_to_drop = drop_amount

        if amount_to_drop > len(t_index):
            amount_to_drop = len(t_index)

        filter = np.random.choice(t_index, size=amount_to_drop, replace=False)
        indices = np.argwhere(np.isin(t_index, filter))
        indices = np.sort(indices)

        sub_arrays = np.split(indices, np.flatnonzero(np.diff(indices.T)!=1) + 1)
        for i in range(len(sub_arrays)):
            sub_arrays[i] = sub_arrays[i].flatten()

        for sub_array in sub_arrays:
            len_sub_array = len(sub_array)
            if len_sub_array > 1:
                for j in range(len_sub_array): 
                    if sub_array[len_sub_array-1]+1 == no_of_frames:
                        end = data_numpy[:,sub_array[len_sub_array-1],:,:]
                    else:
                        end = data_numpy[:,sub_array[len_sub_array-1]+1,:,:]
                    start = data_numpy[:,sub_array[0]-1,:,:]
                    
                    inc = (end-start) / (len_sub_array+1)
                    data_numpy[:,sub_array[j],:,:] = inc * (j+1) + start
            else:
                if len(sub_array) == 0:
                    continue

                data_numpy[:,sub_array[0]] = (data_numpy[:,sub_array[0]-1,:,:] + data_numpy[:,sub_array[0]+1,:,:]) / 2

        return data_numpy   
    
    def drop_next_frame(self, data_numpy, index, no_of_frames):

        channels, frames, joints, num_people = data_numpy.shape
        t_index = np.arange(no_of_frames)
        amount_to_drop = int(self.random_miss_amount*no_of_frames)
        filter = np.random.choice(t_index, size=amount_to_drop, replace=False)
        indices = np.argwhere(np.isin(t_index, filter))
        indices = np.sort(indices)
        indices = indices[::-1]

        for j in range(len(indices)):
            if indices[j] == no_of_frames-1:
                data_numpy[:,indices[j],:,:] = data_numpy[:,indices[j]+1,:,:]
            else:
                continue

        return data_numpy

    def apply_decimate(self, data_numpy, index, no_of_frames):

        if index==0:
            print('Decimating frames')
            print('Decimation frequency: ', self.decimation_frequency)
            print('Mitigation: ', self.mitigation)

        if self.mitigation:
            channels, frames, joints, num_people = data_numpy.shape
            output = np.zeros((channels, frames, joints, num_people), dtype=np.float32)
            output[:,:no_of_frames:self.decimation_frequency,:,:] = data_numpy[:,:no_of_frames:self.decimation_frequency,:,:]

            #arr = data_numpy[:,:no_of_frames:self.decimation_frequency,:,:]

            x = np.arange(no_of_frames)
            y = x[:no_of_frames:self.decimation_frequency]

            selected_indices = np.where(np.isin(x, y))[0]
            all_indices = np.arange(len(x))
            unselected_indices = np.setdiff1d(all_indices, selected_indices)

            sub_arrays = np.split(unselected_indices, np.flatnonzero(np.diff(unselected_indices.T)!=1) + 1)

            for sub_array in sub_arrays:
                if len(x)-1 in sub_array:
                    continue
                len_sub_array = len(sub_array)
                if len_sub_array > 1:
                    for i in range(len_sub_array):
                        
                        if sub_array[len_sub_array-1]+1 == len(x):
                            end = data_numpy[:,sub_array[len_sub_array-1],:,:]
                        else:
                            end = data_numpy[:,sub_array[len_sub_array-1]+1,:,:]
                        start = data_numpy[:,sub_array[0]-1,:,:]
                        
                        inc = (end-start) / (len_sub_array+1)
                        output[:,sub_array[i],:,:] = inc * (i+1) + start
                else:
                    if len(sub_array) == 0:
                        continue
                    output[:,sub_array[0]] = (data_numpy[:,sub_array[0]-1,:,:] + data_numpy[:,sub_array[0]+1,:,:]) / 2
        
        else:
            channels, frames, joints, num_people = data_numpy.shape
            arr = data_numpy[:,:no_of_frames:self.decimation_frequency,:,:]
            output = np.zeros((channels, frames, joints, num_people), dtype=np.float32)
            output[:,:arr.shape[1],:,:] = arr
            if index==0:
                print('Original frames: ', no_of_frames)
                print('After decimating frames: ', arr.shape[1])

        return output
    
    def reduce_frame_rate(self, data_numpy, index, no_of_frames):

        channels, frames, joints, num_people = data_numpy.shape
        FPS_Drop = self.FPS/30 # 30 is the original FPS for NTU-RGB+D-120

        if self.chunks == 1:
            if self.FPS == 30:
                sys.exit("FPS is already 30, cannot reduce frame rate to 30 FPS")
            elif index==0:
                print('Reducing frame rate')
                print('Original FPS: ', 30)
                print('New FPS: ', self.FPS)
                #np.save('before_frame_rate_reduction.npy', data_numpy)
            
            chunk_length = int(no_of_frames - no_of_frames*FPS_Drop)
            if chunk_length >= no_of_frames - 1:
                chunk_length = no_of_frames - 2
            max_chunk_start = no_of_frames - chunk_length
            chunk_start = np.random.randint(0, max_chunk_start-1) # -1 modification to allow for consistency when applying mitigation.
            list_of_delete_indcies = np.arange(chunk_start, chunk_start+chunk_length)
            arr = np.delete(data_numpy, list_of_delete_indcies, axis=1)

            output = np.zeros((channels, frames, joints, num_people), dtype=np.float32)
            output[:,:arr.shape[1],:,:] = arr

            if index == 0:
                print('Original frames: ', no_of_frames)
                print('After reducing frame rate: ', arr.shape[1])
                #np.save('after_frame_rate_reduction.npy', output)

            if self.mitigation:
                output = data_numpy.copy()
                start = data_numpy[:,chunk_start,:,:]
                end = data_numpy[:,chunk_start+chunk_length,:,:]

                for i in range(0, chunk_length):
                    increment = (end-start) / chunk_length
                    output[:,chunk_start+i,:,:] = start + increment * i
                
                if index == 0:
                    print('Mitigation frames: ', data_numpy.shape[1])
                    #np.save('after_mitigation.npy', output)

        else:
            sys.exit("Multi-Chunking not yet supported for frame rate reduction")

        return output

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
