from .nsvf import NSVFDataset, NSVFDataset_v2, NSVFDataset_all
from .colmap import ColmapDataset
from .nerfpp import NeRFPPDataset
from .objaverse import ObjaverseData


dataset_dict = {'nsvf': NSVFDataset,
                'nsvf_v2': NSVFDataset_v2,
                "nsvf_all": NSVFDataset_all,
                'colmap': ColmapDataset,
                'nerfpp': NeRFPPDataset,
                'objaverse': ObjaverseData,}