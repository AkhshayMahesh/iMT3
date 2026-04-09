import random
from collections import defaultdict
from pathlib import Path
from torch.utils.data import BatchSampler, SequentialSampler
from models.contrastive_timbre_embedding import slakh_family_id_from_metadata, _load_slakh_metadata

class TimbreContrastiveBatchSampler(BatchSampler):
    """
    Guarantees batches have `families_per_batch` distinct instrument families,
    with exactly `samples_per_family` segments each (pulled from different tracks where possible)
    prior to querying the dataset.
    """
    
    def __init__(self, sampler, batch_size=16, drop_last=False, dataset=None, families_per_batch=8, samples_per_family=2):
        self.dataset = dataset if dataset is not None else getattr(sampler, 'data_source', getattr(sampler, 'dataset', None))
        if self.dataset is None:
            raise ValueError("Could not infer dataset from sampler, please pass dataset= explicitly initially.")

        super().__init__(sampler, batch_size, drop_last)
        
        self.families_per_batch = families_per_batch
        self.samples_per_family = samples_per_family
        self.effective_batch_size = families_per_batch * samples_per_family
        
        # Mapping from family_id -> list of (track_id, global_dataset_idx)
        self.family_to_indices = defaultdict(list)
        
        print("Initializing TimbreContrastiveBatchSampler, scanning dataset...")
        for global_idx, row in enumerate(self.dataset.df):
            audio_path = row['audio_path']
            track_id = Path(audio_path).parent.name
            meta_path = Path(audio_path).parent / "metadata.yaml"
            
            try:
                meta = _load_slakh_metadata(str(meta_path))
            except Exception:
                continue

            stems = meta.get("stems", {}) or {}
            
            # Find all generic families present in this track
            for stem_id in stems.keys():
                stem_label = slakh_family_id_from_metadata(str(meta_path), stem_id)
                family_id = stem_label.family_id
                
                # To maximize inter-track variation, only append if this global_idx is not already stored for this family
                if not any(item[1] == global_idx for item in self.family_to_indices[family_id]):
                    self.family_to_indices[family_id].append((track_id, global_idx))
                    
        self.families = list(self.family_to_indices.keys())
        
        total_items = sum(len(v) for v in self.family_to_indices.values())
        self.avg_items = total_items / len(self.families) if self.families else 0
        
        print(repr(self))

    def __iter__(self):
        # Obtain the subset of indices assigned to this process (e.g. from DistributedSampler)
        # If running sequentially, this will just be a random permutation or sequential list of all indices
        distributed_indices = list(self.sampler)
        valid_indices = set(distributed_indices)

        # We create a working copy of pools for this epoch, restricted to the DDP sampled indices
        family_pools = {}
        for f, items in self.family_to_indices.items():
            pool = [item for item in items if item[1] in valid_indices]
            random.shuffle(pool)
            family_pools[f] = pool
            
        # Available families that have at least samples_per_family remaining
        available_families = [f for f, pool in family_pools.items() if len(pool) >= self.samples_per_family]
        
        while len(available_families) >= self.families_per_batch:
            # 1. Randomly select the families for this batch
            selected_families = random.sample(available_families, self.families_per_batch)
            
            batch_indices = []
            family_composition = defaultdict(int)
            
            # 2. Extract exactly samples_per_family global_idx for each selected family
            for f in selected_families:
                for _ in range(self.samples_per_family):
                    track_id, global_idx = family_pools[f].pop()
                    batch_indices.append(global_idx)
                    family_composition[f] += 1
                
                # Update availability for future batches
                if len(family_pools[f]) < self.samples_per_family:
                    available_families.remove(f)
            
            # 3. Aggressively assert invariants are upheld
            assert len(family_composition) == self.families_per_batch, \
                f"Batch family count violation: Expected {self.families_per_batch}, got {len(family_composition)}. Composition: {family_composition}"
                
            for f, count in family_composition.items():
                assert count == self.samples_per_family, \
                    f"Batch minimum pair violation for family {f}: Expected {self.samples_per_family}, got {count}. Composition: {family_composition}"
            
            assert len(batch_indices) == self.effective_batch_size, \
                f"Batch size violation: Expected {self.effective_batch_size}, got {len(batch_indices)}."

            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        # Standard approach for dynamic batch samplers
        num_batches = 0
        available_families = [f for f, items in self.family_to_indices.items() if len([item for item in items if item[1] in set(self.sampler)]) >= self.samples_per_family]
        # Rough calculation of how many batches we can possibly yield based on sampler length
        return max(1, len(list(self.sampler)) // self.effective_batch_size)

    def __repr__(self):
        return (
            f"<TimbreContrastiveBatchSampler: "
            f"total families indexed={len(self.families)}, "
            f"avg samples per family={self.avg_items:.1f}, "
            f"effective batch size={self.effective_batch_size}>"
        )
