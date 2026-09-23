import h5py, os
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pdb

class SumMeLLaMADataset(Dataset):

    def __init__(self, mode, split_idx, llama_embedding = 'dataset/llama_emb/summe_sum/'):
        self.mode = mode
        self.dataset = 'dataset/SumMe/eccv16_dataset_summe_google_pool5.h5'
        self.split_file = 'dataset/summe_splits.json'

        # resolve llama embedding paths: accept either provided path or prefixed with 'dataset/'
        userprompt = '{}user_prompt/user_prompt_pool.h5'.format(llama_embedding)
        generation = '{}gen/gen_pool.h5'.format(llama_embedding)

        if not os.path.exists(userprompt):
            alt = os.path.join('dataset', llama_embedding)
            userprompt_alt = '{}user_prompt/user_prompt_pool.h5'.format(alt)
            generation_alt = '{}gen/gen_pool.h5'.format(alt)
            if os.path.exists(userprompt_alt):
                userprompt = userprompt_alt
                generation = generation_alt

        self.userprompt = userprompt
        self.generation = generation

        # store paths; open h5py files lazily to avoid pickling h5py objects when using
        # multiprocessing on Windows (spawn)
        self.video_data_path = self.dataset
        self.llama_emb_userprompt_path = self.userprompt
        self.llama_emb_generation_path = self.generation

        self.video_data = None
        self.llama_emb_userprompt = None
        self.llama_emb_generation = None

        with open(self.split_file, 'r') as f:
            self.data = json.loads(f.read())
            self.data = self.data[split_idx]


    def __len__(self):
        self.len = len(self.data[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        # ensure h5py files are opened in the current process
        if self.video_data is None:
            self.video_data = h5py.File(self.video_data_path, 'r')
        if self.llama_emb_userprompt is None:
            self.llama_emb_userprompt = h5py.File(self.llama_emb_userprompt_path, 'r')
        if self.llama_emb_generation is None:
            self.llama_emb_generation = h5py.File(self.llama_emb_generation_path, 'r')
        video_name = self.data[self.mode + '_keys'][index]
        d = {}
        d['video_name'] = video_name
        d['features'] = torch.Tensor(np.array(self.video_data[video_name + '/features']))
        d['gtscore'] = torch.as_tensor(np.array(self.video_data[video_name + '/gtscore']))
        d['video_filename'] = str(np.array(self.video_data[video_name + '/video_name']))
        d['llama_embedding_userprompt'] = torch.as_tensor(np.array(self.llama_emb_userprompt[d['video_filename'][2:-1].replace(' ', '_')]))
        d['llama_embedding_generation'] = torch.as_tensor(np.array(self.llama_emb_generation[d['video_filename'][2:-1].replace(' ', '_')]))
        
        if self.mode != 'train':
            d['n_frames'] = torch.as_tensor(np.array(self.video_data[video_name + '/n_frames']))
            d['picks'] = torch.as_tensor(np.array(self.video_data[video_name + '/picks']))
            d['change_points'] = torch.as_tensor(np.array(self.video_data[video_name + '/change_points']))
            d['n_frame_per_seg'] = torch.as_tensor(np.array(self.video_data[video_name + '/n_frame_per_seg']))
            d['gt_summary'] = torch.as_tensor(np.array(self.video_data[video_name + '/user_summary']))
        
        return d
    
class TrainBatchCollator(object):
    def __call__(self, batch):
        video_name, video_filename, features, gtscore, llama_embedding_userprompt, llama_embedding_generation = [],[],[],[],[],[]

        try:
            for data in batch:
                video_name.append(data['video_name'])
                video_filename.append(data['video_filename'])
                features.append(data['features'])
                gtscore.append(data['gtscore'])
                llama_embedding_userprompt.append(data['llama_embedding_userprompt'])
                llama_embedding_generation.append(data['llama_embedding_generation'])
        except:
            print('Error in batch collator')

        lengths = torch.LongTensor(list(map(lambda x: x.shape[0], llama_embedding_userprompt)))
        max_len = max(list(map(lambda x: x.shape[0], llama_embedding_userprompt)))
        mask = torch.arange(max_len)[None, :] < lengths[:, None]
        frame_feat = pad_sequence(features, batch_first=True)
        gtscore = pad_sequence(gtscore, batch_first=True)
        llama_embedding_userprompt = pad_sequence(llama_embedding_userprompt, batch_first=True)
        llama_embedding_generation = pad_sequence(llama_embedding_generation, batch_first=True)
        
        batch_data = {'video_name' : video_name,  'features' : frame_feat, 'gtscore':gtscore, 'mask':mask,
                      'llama_embedding_userprompt': llama_embedding_userprompt, 'llama_embedding_generation':llama_embedding_generation}
        return batch_data
    
    
    
class ValBatchCollator(object):
    def __call__(self, batch):
        video_name, video_filename, features, gtscore, llama_embedding_userprompt, llama_embedding_generation = [],[],[],[],[],[]
        cps, nseg, n_frames, picks, gt_summary = [], [], [], [], []

        try:
            for data in batch:
                video_name.append(data['video_name'])
                video_filename.append(data['video_filename'])
                features.append(data['features'])
                gtscore.append(data['gtscore'])
                llama_embedding_userprompt.append(data['llama_embedding_userprompt'])
                llama_embedding_generation.append(data['llama_embedding_generation'])
                cps.append(data['change_points'])
                nseg.append(data['n_frame_per_seg'])
                n_frames.append(data['n_frames'])
                picks.append(data['picks'])
                gt_summary.append(data['gt_summary'])
        except:
            print('Error in batch collator')

        lengths = torch.LongTensor(list(map(lambda x: x.shape[0], llama_embedding_generation)))
        max_len = max(list(map(lambda x: x.shape[0], llama_embedding_generation)))
        mask = torch.arange(max_len)[None, :] < lengths[:, None]
        frame_feat = pad_sequence(features, batch_first=True)
        gtscore = pad_sequence(gtscore, batch_first=True)
        llama_embedding_userprompt = pad_sequence(llama_embedding_userprompt, batch_first=True)
        llama_embedding_generation = pad_sequence(llama_embedding_generation, batch_first=True)

        batch_data = {'video_name' : video_name, 'video_filename' : video_filename, 'features' : frame_feat, 'gtscore':gtscore, 'mask':mask,
                      'llama_embedding_userprompt': llama_embedding_userprompt, 'llama_embedding_generation':llama_embedding_generation,
                      'n_frames': n_frames, 'picks': picks, 'n_frame_per_seg': nseg, 'change_points': cps, 
                      'gt_summary': gt_summary}
        return batch_data