"""Workers crop actual RGB with epoch-dependent bbox jitter before GPU transfer."""
from data.action4_component_data import ComponentData
from data.action4_pose_rgb_batch import prepare_pose_rgb
from data.action4_epoch_batches import EpochRankBatches

class PoseRGBTrainingData:
    def __init__(self,receipt,seed,jitter_config):
        self.data=ComponentData(receipt,"train","A0")
        self.seed=int(seed);self.jitter_config=dict(jitter_config)
    def __len__(self):return len(self.data)
    def __getitem__(self,unit):
        index,epoch=map(int,unit);item=self.data[index]
        seed=self.seed+epoch*1000003+index*104729
        rois,geometry,metadata=prepare_pose_rgb([item],seeds=[seed],jitter_config=self.jitter_config)
        return {"roi":rois[0],"geometry":{k:v[0] for k,v in geometry.items()},"metadata":metadata[0],"unit_index":index,"epoch":epoch}

class PoseEpochBatches(EpochRankBatches):
    def __iter__(self):
        for batch in super().__iter__():yield [(index,self.epoch) for index in batch]
