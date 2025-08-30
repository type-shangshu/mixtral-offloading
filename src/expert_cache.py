from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Iterator, Tuple, List, TypeVar, Generic
from collections import deque, defaultdict, OrderedDict
from .expert_wrapper import MixtralExpertWrapper
from abc import ABC, abstractmethod

import torch
from torch import nn

import random

ExpertUID = Any

T = TypeVar('T')

class CachePolicy(ABC, Generic[T]):
    @abstractmethod
    def add(self, key: Any, value: T) -> None:
        pass
    @abstractmethod
    def get(self, key: Any) -> T:
        pass
    @abstractmethod
    def remove(self, key: Any) -> T:
        pass
    @abstractmethod
    def choose_to_evict(self) -> T:
        pass
    @abstractmethod
    def __len__(self) -> int:
        pass

@dataclass(frozen=False)
class ExpertInfo:
    uid: ExpertUID
    eviction_group: int
    offloaded: bool
    index: int

class RRPolicy(CachePolicy[T]):
    def __init__(self):
        self.cache = {}
    
    def add(self, key: Any, value: T) -> None:
        self.cache[key] = value

    def get(self, key: Any) -> T:
        return self.cache[key]
    
    def remove(self, key: Any) -> T:
        return self.cache.pop(key)
    
    def choose_to_evict(self) -> Any:
        if not self.cache:
            raise StopIteration("Cache is empty")
        return random.choice(list(self.cache.keys()))
    
    def __len__(self) -> int:
        return len(self.cache)
    
class LRUPolicy(CachePolicy[T]):
    def __init__(self):
        self.cache = OrderedDict()
    
    def add(self, key: Any, value: T) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        
    def get(self, key: Any) -> T:
        self.cache.move_to_end(key)
        return self.cache[key]
        
    def remove(self, key: Any) -> T:
        return self.cache.pop(key)
        
    def choose_to_evict(self) -> Any:
        return next(iter(self.cache))
    
    def __len__(self) -> int:
        return len(self.cache)

class LFUPolicy(CachePolicy[T]):
    def __init__(self):
        self.cache = {}  # key -> value
        self.frequencies = defaultdict(set)  # frequency -> set of keys
        self.key_counts = {}  # key -> frequency
        self.min_freq = 0
        
    def add(self, key: Any, value: T) -> None:
        self.cache[key] = value
        self.key_counts[key] = 1
        self.frequencies[1].add(key)
        self.min_freq = 1
        
    def get(self, key: Any) -> T:
        freq = self.key_counts[key]
        self.frequencies[freq].remove(key)
        if not self.frequencies[freq] and freq == self.min_freq:
            self.min_freq += 1
        
        self.key_counts[key] = freq + 1
        self.frequencies[freq + 1].add(key)
        return self.cache[key]
        
    def remove(self, key: Any) -> T:
        freq = self.key_counts[key]
        self.frequencies[freq].remove(key)
        if not self.frequencies[freq] and freq == self.min_freq:
            self.min_freq = min(self.key_counts.values(), default=0)
        del self.key_counts[key]
        return self.cache.pop(key)
        
    def choose_to_evict(self) -> Any:
        return next(iter(self.frequencies[self.min_freq]))
    
    def __len__(self) -> int:
        return len(self.cache)

@dataclass
class EvictionGroupInfo:
    cache_policy: str = field(default="lru") # "lru", "random" or "lfu"
    main_policy: CachePolicy[ExpertInfo] = field(init=False)
    offloaded_policy: CachePolicy[ExpertInfo] = field(init=False)
    hits: int = field(default=0)
    misses: int = field(default=0)
    
    def __post_init__(self):
        # print(self.cache_policy)
        if self.cache_policy == "lru":
            PolicyClass = LRUPolicy
        elif self.cache_policy == "lfu":
            PolicyClass = LFUPolicy
        elif self.cache_policy == "random":
            PolicyClass = RRPolicy
        else:
            raise ValueError(f"Unknown cache policy: {self.cache_policy}")
        # PolicyClass = LRUPolicy if self.cache_policy == "lru" else LFUPolicy
        self.main_policy = PolicyClass()
        self.offloaded_policy = PolicyClass()
    
    def add(self, info: ExpertInfo):
        policy = self.offloaded_policy if info.offloaded else self.main_policy
        policy.add(info.uid, info)
    
    def choose_expert_to_evict(self) -> ExpertInfo:
        try:
            uid = self.main_policy.choose_to_evict()
            return self.main_policy.get(uid)
        except StopIteration:
            raise ValueError("No evictable experts")
    
    def swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo):
        self.main_policy.add(info_to_load.uid, info_to_load)
        self.offloaded_policy.add(info_to_evict.uid, info_to_evict)
        self.main_policy.remove(info_to_evict.uid)
        self.offloaded_policy.remove(info_to_load.uid)
    
    def mark_used(self, info: ExpertInfo):
        try:
            self.main_policy.get(info.uid)
            self.hits += 1
        except KeyError:
            try:
                self.offloaded_policy.get(info.uid)
                self.misses += 1
            except KeyError:
                raise ValueError(f"Expert {info} not in group")

    def switch_policy(self, new_policy: str):
        """Switch to a new cache policy while preserving cached items"""
        if new_policy == self.cache_policy:
            return
            
        # 创建新的策略实例
        if new_policy == "lru":
            NewPolicyClass = LRUPolicy
        elif new_policy == "lfu":
            NewPolicyClass = LFUPolicy
        elif new_policy == "random":
            NewPolicyClass = RRPolicy
        else:
            raise ValueError(f"Unknown cache policy: {new_policy}")
            
        # 保存当前缓存的items
        main_items = [(k, v) for k, v in self.main_policy.cache.items()]
        offload_items = [(k, v) for k, v in self.offloaded_policy.cache.items()]
        
        # 创建新的策略实例
        self.main_policy = NewPolicyClass()
        self.offloaded_policy = NewPolicyClass()
        
        # 重新添加items
        for k, v in main_items:
            self.main_policy.add(k, v)
        for k, v in offload_items:
            self.offloaded_policy.add(k, v)
            
        self.cache_policy = new_policy

class ExpertCache:
    def __init__(self, make_module: callable, main_size: int, offload_size: int, buffer_size: int, cache_strategy: str = "lru"):
        """Dynamically loads an array of modules with identical hyperparameters"""
        self.module_type = self.module_size = self.device = None
        self.active = False

        self.registered_experts: Dict[ExpertUID, ExpertInfo] = dict()

        self.main_modules = [self._check_module(make_module()) for i in range(main_size)]
        self.main_infos: List[Optional[ExpertInfo]] = [None for _ in range(main_size)]

        assert self.module_size is not None
        self.offloaded_storages = [
            torch.UntypedStorage(self.module_size).pin_memory(self.device) for _ in range(offload_size)]
        self.offloaded_infos: List[Optional[ExpertInfo]] = [None for _ in range(offload_size)]

        # temporary storage to shave off latency
        self.device_expert_buffers = deque([self._check_module(make_module()) for _ in range(buffer_size)])
        self.offloaded_storage_buffers = deque([
            torch.UntypedStorage(self.module_size).pin_memory(self.device) for _ in range(buffer_size)])
        # self.group_infos: Dict[int, EvictionGroupInfo] = defaultdict(EvictionGroupInfo)
        self.group_infos: Dict[int, EvictionGroupInfo] = defaultdict(lambda: EvictionGroupInfo(cache_policy=cache_strategy))

    def _check_module(self, module: MixtralExpertWrapper):
        assert isinstance(module.storage, torch.UntypedStorage)
        if self.module_type is None:
            self.module_type = type(module)
            self.module_size = len(module.storage)
            self.device = module.storage.device
        else:
            assert isinstance(module, self.module_type)
            assert len(module.storage) == self.module_size
            assert module.storage.device == self.device
        return module

    def add_expert(self, uid: ExpertUID, module: MixtralExpertWrapper, eviction_group: int = 0,
                   offload: Optional[bool] = None):
        """Register an expert to the cache and associate it with uid"""
        assert self.module_type is not None
        assert isinstance(module, self.module_type)
        return self.add_expert_storage(uid, module.storage, eviction_group=eviction_group, offload=offload)

    def add_expert_storage(self, uid: ExpertUID, storage: torch.UntypedStorage,
                           eviction_group: int = 0, offload: Optional[bool] = None):
        assert uid not in self.registered_experts, f"expert {uid} already registered"
        assert isinstance(storage, torch.UntypedStorage)
        assert len(storage) == self.module_size

        if offload is None or not offload:  # False or None
            for i in range(len(self.main_modules)):
                if self.main_infos[i] is None:
                    self.main_modules[i].storage.copy_(storage)
                    info = ExpertInfo(uid, eviction_group=eviction_group, offloaded=False, index=i)
                    self.registered_experts[uid] = self.main_infos[i] = info
                    self.group_infos[eviction_group].add(info)
                    return  # done allocating; found spot on device
        if offload is None or offload:  # True or None
            for i in range(len(self.offloaded_storages)):
                if self.offloaded_infos[i] is None:
                    self.offloaded_storages[i].copy_(storage)
                    info = ExpertInfo(uid, eviction_group=eviction_group, offloaded=True, index=i)
                    self.registered_experts[uid] = self.offloaded_infos[i] = info
                    self.group_infos[eviction_group].add(info)
                    return  # done allocating; found an offloaded spot
        raise ValueError("Cache is full")

    def load_experts(
            self, *uids: ExpertUID, unordered: bool = False) -> Iterator[Tuple[ExpertUID, MixtralExpertWrapper]]:
        """
        :example:
        >>> for uid, expert in expert_cache.load_experts(*list_of_uids, unordered=True):
        >>>     for uid, expert in expert_iter:
        >>>         result += expert(x) * get_moe_weight(uid)

        :param uids: iterate over the specified expert uids. Same uids as in add_expert
        :param unordered: if True, allows cache to iterate experts in arbitrary order
            The order is chosen to minimize the total wait time.
        :returns: an iterator that yields (uid, expert) pairs, only usable inside the for loop

        """
        assert len(set(uids)) == len(uids)
        assert not self.active, "already loading experts; buffers are busy"
        if unordered:  # yield non-offloaded experts first
            uids = sorted(uids, key=lambda uid: self.registered_experts[uid].offloaded)
        infos = [self.registered_experts[uid] for uid in uids]

        assert len(set(info.eviction_group for info in infos)) == 1, "experts must be in the same evicton group"
        eviction_group = self.group_infos[infos[0].eviction_group]
        for info in infos:
            eviction_group.mark_used(info)

        try:
            self.active = True
            # save pre-loaded experts before they can be swapped
            pre_loaded_infos = deque([info for info in infos if not info.offloaded])
            pre_loaded_experts = deque([self.main_modules[info.index] for info in pre_loaded_infos])

            # begin loading experts into free buffers in background (via non-blocking copy)
            infos_to_load = deque([info for info in infos if info.offloaded])
            infos_in_loading = deque([])
            experts_in_loading = deque([])
            window_size = min(len(self.device_expert_buffers) - 1,
                              len(eviction_group.main_policy),
                              len(infos_to_load))
            for _ in range(window_size):
                info_to_load = infos_to_load.popleft()
                infos_in_loading.append(info_to_load)
                experts_in_loading.append(
                    self._swap(info_to_load, eviction_group.choose_expert_to_evict()))

            for info in infos:
                if len(pre_loaded_infos) > 0 and info is pre_loaded_infos[0]:
                    pre_loaded_infos.popleft()
                    yield (info.uid, pre_loaded_experts.popleft())
                elif len(infos_in_loading) > 0 and info is infos_in_loading[0]:
                    infos_in_loading.popleft()
                    yield (info.uid, experts_in_loading.popleft())
                    if len(infos_to_load) > 0:
                        info_to_load = infos_to_load.popleft()
                        infos_in_loading.append(info_to_load)
                        experts_in_loading.append(
                            self._swap(info_to_load, eviction_group.choose_expert_to_evict()))
                else:
                    raise RuntimeError("internal error: caching algorithm failed")
        finally:
            self.active = False

    def _swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo) -> nn.Module:
        """Swap an offloaded expert (info_to_load) with an on-device expert (info_to_evict) return the loaded expert"""
        assert info_to_load.offloaded and not info_to_evict.offloaded
        assert info_to_load.eviction_group == info_to_evict.eviction_group
        # swap a single on-device expert with a single offloaded expert using buffers for parallelism
        offloaded_storage_buffer = self.offloaded_storage_buffers.popleft()
        device_expert_buffer = self.device_expert_buffers.popleft()
        device_expert_buffer.storage.copy_(self.offloaded_storages[info_to_load.index], non_blocking=True)
        offloaded_storage_buffer.copy_(self.main_modules[info_to_evict.index].storage, non_blocking=True)

        self.device_expert_buffers.append(self.main_modules[info_to_evict.index])
        self.main_modules[info_to_evict.index] = device_expert_buffer
        self.offloaded_storage_buffers.append(self.offloaded_storages[info_to_load.index])
        self.offloaded_storages[info_to_load.index] = offloaded_storage_buffer

        self.main_infos[info_to_evict.index] = info_to_load
        self.offloaded_infos[info_to_load.index] = info_to_evict
        info_to_evict.offloaded, info_to_load.offloaded = info_to_load.offloaded, info_to_evict.offloaded
        info_to_evict.index, info_to_load.index = info_to_load.index, info_to_evict.index
        self.group_infos[info_to_load.eviction_group].swap(info_to_load, info_to_evict)
        return device_expert_buffer

    def switch_cache_strategy(self, new_strategy: str):
        """Switch cache strategy for all eviction groups"""
        print(f"Switching cache strategy from {self.cache_strategy} to {new_strategy}")
        self.cache_strategy = new_strategy
        for group_info in self.group_infos.values():
            group_info.switch_policy(new_strategy)
