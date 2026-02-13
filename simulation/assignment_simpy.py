import simpy
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from .forecast import Forecast
from gluonts.model import Forecast as GluontsForecast
from .scheduler_simpy import SchedulingSimulation
from typing import Any
from concurrent.futures import ThreadPoolExecutor

class VM:
    def __init__(self, input: dict, label: dict, forecast: Any, metadata: dict, quantile: Optional[float] = None):
        item_id = input['item_id']
        self.index =int(item_id.split('_')[1])
        self.item_id = item_id
        self.freq = metadata['freq']
        self.start = label['start']
        self.period = metadata['prediction_length']
        
        if len(input['target'].shape) == 1:
            self.past_util = input
            self.real_util = label
            self.forecast_util = forecast
        else:
            input['target'] = input['target'][0, :]
            self.past_util = input
            label['target'] = label['target'][0, :]
            self.real_util = label
            forecast.samples = forecast.samples[:, :, 0]
            self.forecast_util = forecast

        self.forecast_util_quantile = self.forecast_util.quantile(quantile)   
        self.allocation_time = None
        
    def get_arrival_time(self) -> float:
        """Get VM arrival time"""
        if isinstance(self.start, pd.Period):
            # Convert to a relative timestamp (hours since a baseline time)
            base_time = pd.Period('2010-01-01', freq=self.freq)
            return float((self.start - base_time).n)
        else:
            raise ValueError("start must be a pandas.Period")

    def get_finish_time(self) -> float:
        """Get VM finish time"""
        if isinstance(self.start, pd.Period):
            base_time = pd.Period('2010-01-01', freq=self.freq)
            # Correct: period time points correspond to the actual duration
            end_time = self.start + (self.period - 1)  # period-1 because it includes the starting point
            return float((end_time - base_time).n)
        else:
            raise ValueError("start must be a pandas.Period")
    
    def get_utilization_at_time(self, t: int) -> float:
        """Get resource utilization at a given time"""
        if t < len(self.forecast_util_quantile):
            if self.forecast_util_quantile[t] < 0:
                return 0.0
            return float(self.forecast_util_quantile[t])
        return 0.0
    
    def get_real_utilization_at_time(self, t: int) -> float:
        """Get real resource utilization at a given time"""
        if t < len(self.real_util['target']):
            return float(self.real_util['target'][t])
        return 0.0
    
    def get_peak_utilization(self) -> float:
        """Get predicted peak utilization"""
        if np.max(self.forecast_util_quantile) < 0:
            return 0.0
        return float(np.max(self.forecast_util_quantile))
    
    def get_real_peak_utilization(self) -> float:
        """Get real peak utilization"""
        return float(np.max(self.real_util['target']))

class Host:
    def __init__(self,  index: int, capacity: float = 4.0,threshold_rate: float = 1):
        self.index = index
        self.capacity = capacity         # Host capacity
        self.threshold = threshold_rate*capacity

        # Resource management
        self.allocated_vms: Dict[int, VM] = {}
        
        # Time-series data
        self.forecast_history = {}     # {timestamp: forecast_utilization}
        self.real_history = {}         # {timestamp: real_utilization}

        # Time-dimension utilization cache for quick capacity checks
        self.time_utilization_cache: Dict[float, float] = {}
        

    def can_accommodate(self, vm: VM, current_time: float) -> bool:
        """Check whether the host can accommodate a new VM using the cached time-dimension utilization"""
        # Get VM time-frequency info
        freq_multiplier = self._get_freq_multiplier(vm.freq)
        for t in range(vm.period):
            future_time = current_time + t * freq_multiplier
            future_utilization = self.time_utilization_cache.get(future_time, 0.0)
            new_vm_utilization = vm.get_utilization_at_time(t)
            total_future_utilization = future_utilization + new_vm_utilization
            if total_future_utilization > self.threshold:
                return False
        return True
    
    def _get_freq_multiplier(self, freq: str) -> float:
        """Get time multiplier from frequency string"""
        if freq.endswith('T'):
            # Extract numeric part, e.g. "5T" -> 5, "30T" -> 30
            multiplier = int(freq[:-1]) if freq[:-1] else 1
            return float(multiplier)
        elif freq.endswith('H'):
            multiplier = int(freq[:-1]) if freq[:-1] else 1
            return float(multiplier)
        else:
            return 1.0
    
    def allocate_vm(self, vm: VM, allocation_time: float) -> bool:
        """Allocate VM to this host and update the time-utilization cache"""
        if vm.index in self.allocated_vms:
            return False
        freq_multiplier = self._get_freq_multiplier(vm.freq)
        
        # Update time-utilization cache
        for t in range(vm.period):
            future_time = allocation_time + t * freq_multiplier
            vm_utilization = vm.get_utilization_at_time(t)
            if future_time in self.time_utilization_cache:
                self.time_utilization_cache[future_time] += vm_utilization
            else:
                self.time_utilization_cache[future_time] = vm_utilization
        
        # Allocate VM
        self.allocated_vms[vm.index] = vm
        vm.allocation_time = allocation_time
        if vm.allocation_time !=vm.get_arrival_time():
            raise ValueError("VM allocation time must match arrival time")

        # Clean up expired cache entries to control memory usage
        self._cleanup_cache(allocation_time)
        
        return True
    
    def deallocate_vm(self, vm: VM):
        """Release VM from host and update time-utilization cache"""
        if vm.index not in self.allocated_vms:
            return
        
        freq_multiplier = self._get_freq_multiplier(vm.freq)        
        # Subtract this VM's utilization from the time-utilization cache
        if vm.allocation_time is not None:
            for t in range(vm.period):
                future_time = vm.allocation_time + t * freq_multiplier
                vm_utilization = vm.get_utilization_at_time(t)
                if future_time in self.time_utilization_cache:
                    self.time_utilization_cache[future_time] -= vm_utilization
                    if abs(self.time_utilization_cache[future_time]) < 1e-10:
                        del self.time_utilization_cache[future_time]
        # Release VM
        # del self.allocated_vms[vm.index]
    
    def _cleanup_cache(self, current_time: float):
        """Clean up expired cache entries to control memory usage"""
        expired_times = [t for t in self.time_utilization_cache.keys() 
                        if t < current_time]  
        
        for t in expired_times:
            del self.time_utilization_cache[t]

        
    def update_current_utilization(self, timestamp: float):
        """Update total utilization at the current time"""
        total_forecast = 0.0
        total_real = 0.0
        
        for vm_id, vm in self.allocated_vms.items():
            time_offset = int((timestamp - vm.allocation_time)/self._get_freq_multiplier(vm.freq))
            if 0 <= time_offset < vm.period:
                total_forecast += vm.get_utilization_at_time(time_offset)
                total_real += vm.get_real_utilization_at_time(time_offset)

        # Record history
        self.forecast_history[timestamp] = total_forecast
        self.real_history[timestamp] = total_real
    
    def is_hotspot(self) -> bool:
        """Check whether this host is a hotspot (global check)"""
        if  all(util == 0 for util in self.real_history.values()):
            return False
        return any(util > self.capacity for util in self.real_history.values())
    
    def calculate_hot_rate(self) -> float:
        """Calculate hotspot rate (global check)"""
        violations = sum(1 for util in self.real_history.values() if util > self.capacity)
        return violations / len(self.real_history)

    def calculate_hot_level(self) -> float:
        """Calculate hotspot level (global check)"""
        violations = sum(1 for util in self.real_history.values() if util > self.capacity)
        level = sum((util-self.capacity)/self.capacity for util in self.real_history.values() if util > self.capacity)
        if violations == 0:
            return 0
        return level / violations 
    
    def get_utilization_series(self) -> pd.Series:
        """Get utilization time series"""
        timestamps = sorted(self.real_history.keys())
        values = [self.real_history[t] for t in timestamps]
        return pd.Series(values, index=timestamps)

class SimPyAssignment:
    """SimPy version of Assignment class"""
    
    def __init__(self, num_hosts: int, host_capacity: float, threshold_rate: float, scheduler_type: str, forecast_path: str, mapping: Optional[Dict[int, int]] = None, quantile: Optional[float] = None):
        self.hosts = {host.index: host for host in [Host(i, host_capacity,threshold_rate) for i in range(num_hosts)]}
        
        forecast_obj_load = Forecast.load(forecast_path)
        self.vms = []
        for i in range(len(forecast_obj_load)):
            vm = VM(
                forecast_obj_load.inputs[i],
                forecast_obj_load.labels[i], 
                forecast_obj_load.forecasts[i],
                forecast_obj_load.metadata,
                quantile,
            )
            self.vms.append(vm)
        
        # Create simulation environment
        self.simulation = SchedulingSimulation(scheduler_type)
        self._mapping = mapping
        self.mapping = {}
        
    def schedule(self):
        """Run scheduling simulation"""
        self.simulation.setup(self.vms, self.hosts, self._mapping)
        mapping = self.simulation.run()
        self.mapping = mapping
        

    def compute_util(self) -> float:
        """Compute average utilization of the entire cluster"""
        if not self.mapping:
            return -1

        total_util = 0.0
        active_hosts = 0
        
        for host in self.hosts.values():
            if  not all(util == 0 for util in host.real_history.values()):
                util_series = host.get_utilization_series()
                avg_util = util_series.mean() / host.capacity
                total_util += avg_util
                active_hosts += 1
        
        return total_util / active_hosts if active_hosts > 0 else -1
    
    def get_hotspot(self) -> List[int]:
        """Get hotspot host list"""
        if not self.mapping:
            raise ValueError("Scheduling failed!")
        return [host.index for host in self.hosts.values() if host.is_hotspot()]
    
    def compute_SLAV(self) -> float:
        """Compute violation rate, measured as the percentage of violating time over total VM runtime"""
        if not self.mapping:
            return -1

        total_rate = 0.0
        active_hosts = 0
        
        for host in self.hosts.values():
            if  not all(util == 0 for util in host.real_history.values()):
                rate = host.calculate_hot_rate()
                total_rate += rate
                active_hosts += 1
        
        return total_rate / active_hosts if active_hosts > 0 else -1

    def compute_CVS(self) ->float:
        if not self.mapping:
            return -1

        total_rate = 0.0
        active_hosts = 0
        
        for host in self.hosts.values():
            if  not all(util == 0 for util in host.real_history.values()):
                rate = host.calculate_hot_level()
                total_rate += rate
                if rate > 0:
                    active_hosts += 1
        
        return total_rate / active_hosts if active_hosts > 0 else 0
    
    def compute_PAR(self) -> float:
        """Compute peak-to-average ratio"""
        if not self.mapping:
            return -1

        max_utils = []
        
        for host in self.hosts.values():
            if  not all(util == 0 for util in host.real_history.values()):
                util_series = host.get_utilization_series()
                max_utils.append(util_series.max())
        
        if not max_utils:
            return -1
            
        return max(max_utils) / np.mean(max_utils)


    def MonteCarloSimulation(self, num_simulations: int = 100, confidence_level: float = 0.95, random_seed: Optional[int] = None) -> Dict[str, float]:
        picp_list = []
        mpiw_list = []
        winkler_scores = []
        alpha = 1 - confidence_level

        active_hosts = [host for host in self.hosts.values() if not all(util == 0 for util in host.real_history.values())]

        if not active_hosts:
            return {
                "PICP": 0.0,
                "MPIW": 0.0,
                "WinklerScore": 0.0,
            }

        if random_seed is not None:
            seeds = [random_seed + i for i in range(len(active_hosts))]
        else:
            seeds = [None] * len(active_hosts)

        def _simulate_host(args):
            host, seed = args
            if seed is not None:
                rng = np.random.default_rng(seed)
            else:
                rng = np.random.default_rng()

            vms_forecast = [vm.forecast_util for vm in host.allocated_vms.values()]
            real_util_values = host.get_utilization_series()

            num_vms = len(vms_forecast)
            u_matrix = rng.uniform(0, 1, (num_simulations, num_vms))

            prediction_length = len(real_util_values)
            sampled_util_matrix = np.zeros((num_simulations, num_vms, prediction_length), dtype=float)

            for j, f in enumerate(vms_forecast):
                levels_for_vm = u_matrix[:, j]
                from gluonts.model.forecast import SampleForecast
                if isinstance(f, SampleForecast):
                    samples = np.asarray(f.samples)
                    num_samples = samples.shape[0]
                    idx = np.rint((num_samples - 1) * levels_for_vm).astype(int)
                    idx = np.clip(idx, 0, num_samples - 1)
                    q_vals = samples[idx]
                # elif hasattr(f, "quantiles"):
                #     q_vals = f.quantiles(levels_for_vm)
                #     q_vals = np.asarray(q_vals)
                else:
                    q_vals = np.stack([np.asarray(f.quantile(float(level))) for level in levels_for_vm], axis=0)
                if q_vals.ndim > 2:
                    q_vals = q_vals.reshape(q_vals.shape[0], -1)
                sampled_util_matrix[:, j, :] = q_vals


            # simulated_host_utils shape: (num_simulations, prediction_length)
            simulated_host_utils = np.sum(sampled_util_matrix, axis=1)
    
            lower_bound = np.percentile(simulated_host_utils, (1 - confidence_level) / 2 * 100, axis=0)
            upper_bound = np.percentile(simulated_host_utils, (1 + confidence_level) / 2 * 100, axis=0)

            # Calculate metrics for this host
            # PICP: proportion of real values within bounds
            coverage = ((real_util_values >= lower_bound) & (real_util_values <= upper_bound))
            picp_host = np.mean(coverage)
            
            # MPIW: mean width of the interval
            interval_width = upper_bound - lower_bound
            mpiw_host = np.mean(interval_width)

            # Calculate Winkler Score for this host
            penalty_lower = (2 / alpha) * np.maximum(0, lower_bound - real_util_values)
            penalty_upper = (2 / alpha) * np.maximum(0, real_util_values - upper_bound)
            winkler_score_host = np.mean(interval_width + penalty_lower + penalty_upper)

            return picp_host, mpiw_host, winkler_score_host

        for args in zip(active_hosts, seeds):
            picp_host, mpiw_host, winkler_score_host = _simulate_host(args)
            picp_list.append(picp_host)
            mpiw_list.append(mpiw_host)
            winkler_scores.append(winkler_score_host)

        return {
            "PICP": float(np.mean(picp_list)),
            "MPIW": float(np.mean(mpiw_list)),
            "WinklerScore": float(np.mean(winkler_scores)),
        }

