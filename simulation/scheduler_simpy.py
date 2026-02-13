import simpy
from abc import ABC
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import copy
import random
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Optional

class SimPyScheduler(ABC):
    def __init__(self, env: simpy.Environment, vms, hosts):
        self.env = env
        self.vms = vms
        self.hosts = hosts
        self.mapping = {}

        self.batch_size = 2000  # Number of VMs per batch schedule
        self.pending_vms = []
        self.processed_vms = 0
        
    def schedule_vm_arrival(self, vm):
        """Schedule a VM arrival event"""
        yield self.env.timeout(0) 
        return self.place_vm(vm)


    def place_vm(self, vm):
        """Add a VM to the pending scheduling queue"""
        vm_arrival_time = vm.get_arrival_time()
        
        # If pending_vms is empty, or arrival_time matches the queue head, enqueue it
        if not self.pending_vms or self.pending_vms[0].get_arrival_time() == vm_arrival_time:
            self.pending_vms.append(vm)
            self.processed_vms += 1
            
            is_last_vm = (self.processed_vms == len(self.vms))
            batch_full = (len(self.pending_vms) >= self.batch_size)
            
            if batch_full or is_last_vm:
                return self.batch_schedule()
            return True
        else:
            # If arrival_time differs, schedule current queue first, then handle the new VM
            result = self.batch_schedule()
            self.pending_vms.append(vm)
            self.processed_vms += 1
            is_last_vm = (self.processed_vms == len(self.vms))
            if is_last_vm:
                return self.batch_schedule()
            
            return result
        
    def monitor_resources(self):
        """Monitor resource utilization"""

        # Compute monitoring window: from earliest VM arrival to latest finish time
        start_time = min(vm.get_arrival_time() for vm in self.vms)
        end_time = max(vm.get_finish_time() for vm in self.vms)
        
        # Compute monitoring interval (based on the minimum VM frequency)
        monitor_interval = float(int(self.vms[0].freq[:-1]) if self.vms[0].freq[:-1].isdigit() else 1 )
        
        # Monitor from start time to end time
        if start_time>self.env.now:
            yield self.env.timeout(start_time-self.env.now+0.1) # Monitoring lags scheduling events
        while self.env.now <= end_time+0.1:
            # Update utilization for all hosts at the current time point
            for host in self.hosts.values():
                host.update_current_utilization(self.env.now-0.1)
            # Wait for the next monitoring interval
            yield self.env.timeout(monitor_interval)

    def vm_lifecycle(self, vm, host):
        """Manage the full VM lifecycle"""
        start_time = self.env.now
        finish_time=vm.get_finish_time()    
        yield self.env.timeout(finish_time-start_time+1)
        host.deallocate_vm(vm)
        


class SimPyFirstFitDecreasing(SimPyScheduler):
    """SimPy FirstFitDecreasing scheduler"""
    def __init__(self, env: simpy.Environment, vms, hosts):
        super().__init__(env, vms, hosts)
        self.batch_size = 10000  # Number of VMs per batch schedule
        self.pending_vms = []
        self.processed_vms = 0

    def batch_schedule(self):
        if not self.pending_vms:
            return True
            
        vms_to_schedule = self.pending_vms.copy()
        self.pending_vms.clear()
        
        vms_to_schedule.sort(
            key=lambda vm: vm.forecast_util_quantile.mean(),
            reverse=True,
        )
        hosts_list=list(self.hosts.items())
        current_time = self.env.now
        
        for vm in vms_to_schedule:
            best_host_idx = None
            # Traverse hosts and find the first that can accommodate the VM
            for host_item in hosts_list:
                host=host_item[1]
                if host.can_accommodate(vm, current_time):
                    best_host_idx=host.index
                    break
            
            # Allocate the VM to the selected host
            if best_host_idx is not None:
                success = self.hosts[best_host_idx].allocate_vm(vm, current_time)
                if success:
                    self.mapping[vm.index] = best_host_idx
                    self.env.process(self.vm_lifecycle(vm, self.hosts[best_host_idx]))
                else:
                    self.mapping[vm.index] = -1
            else:
                self.mapping[vm.index] = -1

        return True


    

class SimPyBestFitDecreasing(SimPyScheduler):
    """SimPy BestFitDecreasing scheduler"""
    def __init__(self, env: simpy.Environment, vms, hosts):
        super().__init__(env, vms, hosts)
        self.batch_size = 10000  # Number of VMs per batch schedule
        self.pending_vms = []
        self.processed_vms = 0

    def batch_schedule(self):
        if not self.pending_vms:
            return True
            
        vms_to_schedule = self.pending_vms.copy()
        self.pending_vms.clear()
        
        vms_to_schedule.sort(
            key=lambda vm: vm.forecast_util_quantile.mean(),
            reverse=True,
        )
        sorted_hosts=sorted(self.hosts.items(), key=lambda item: sum(item[1].time_utilization_cache.values()),reverse=True)
        current_time = self.env.now
        
        for vm in vms_to_schedule:
            best_host_idx = None
            # Traverse hosts and find the first that can accommodate the VM
            for host_item in sorted_hosts:
                host=host_item[1]
                if host.can_accommodate(vm, current_time):
                    best_host_idx=host.index
                    break
            
            # Allocate the VM to the selected host
            if best_host_idx is not None:
                success = self.hosts[best_host_idx].allocate_vm(vm, current_time)
                if success:
                    self.mapping[vm.index] = best_host_idx
                    self.env.process(self.vm_lifecycle(vm, self.hosts[best_host_idx]))
                    sorted_hosts.sort(key=lambda host_item:sum(host_item[1].time_utilization_cache.values()),reverse=True)
                else:
                    self.mapping[vm.index] = -1
            else:
                self.mapping[vm.index] = -1

        return True

class SimPyACOScheduler(SimPyScheduler):
    """SimPy ant colony optimization scheduler"""
    
    def __init__(self, env: simpy.Environment, vms, hosts):
        super().__init__(env, vms, hosts)
        self.batch_size = 10000  # Number of VMs per batch schedule
        self.pending_vms = []
        self.processed_vms = 0
        
        # ACO parameters
        self.num_ants = 10  # Number of ants
        self.max_iterations = 50  # Maximum iterations
        self.alpha = 1.0  # Pheromone importance
        self.beta = 1.0   # Heuristic importance
        self.rho = 0.2    # Pheromone evaporation rate
        self.Q = 1.0      # Pheromone intensity
        
        self.pheromone_matrix = {}
        self.patience = 5

        
    def initialize_pheromone_matrix(self,vms):
        """Initialize the pheromone matrix"""
        self.pheromone_matrix = {}
        for vm in vms:
            self.pheromone_matrix[vm.index] = {}
            for host_idx in self.hosts.keys():
                self.pheromone_matrix[vm.index][host_idx] = 1.0
    
    def calculate_bestfit_heuristic(self, vm, available_hosts, temp_hosts):
        """
        Compute dynamic heuristics based on BestFit strategy
        temp_hosts: temporary host state during an ant's solution construction
        """
        heuristics = {}
        
        sorted_temp_hosts=sorted(temp_hosts.items(), key=lambda item: sum(item[1].time_utilization_cache.values()),reverse=True)
        
        # Compute heuristics based on rank
        num_valid_hosts = len(sorted_temp_hosts)
        for rank, (host_idx, host) in enumerate(sorted_temp_hosts):
            # Higher rank (smaller remaining capacity) gets higher heuristic value
            # Use reverse ranking: rank 1 gets the highest score, last gets the lowest
            rank_score = (num_valid_hosts - rank) / num_valid_hosts
            
            base_heuristic = 1.0 + rank_score * 9
            heuristics[host_idx] = base_heuristic
        
        return heuristics
    
    def calculate_probability(self, vm, available_hosts, temp_hosts):
        """Compute the probability of choosing each host"""
        probabilities = {}
        total = 0.0
        
        # Compute heuristics dynamically
        heuristics = self.calculate_bestfit_heuristic(vm, available_hosts, temp_hosts)
        
        for host_idx in available_hosts:
            pheromone = self.pheromone_matrix[vm.index][host_idx]
            heuristic = heuristics[host_idx]
            
            prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities[host_idx] = prob
            total += prob
        
        # Normalize probabilities
        if total > 0:
            for host_idx in probabilities:
                probabilities[host_idx] /= total
        else:
            # If all probabilities are 0, use a uniform distribution
            uniform_prob = 1.0 / len(available_hosts) if available_hosts else 0
            for host_idx in available_hosts:
                probabilities[host_idx] = uniform_prob
                
        return probabilities
    
    def construct_solution(self, vms, current_time):
        """Construct a solution (one ant's path)"""
        solution = {}
        temp_hosts = copy.deepcopy(self.hosts)
        sorted_vms = sorted(vms, key=lambda vm: vm.forecast_util_quantile.mean(), reverse=True)
        
        for vm in sorted_vms:
            # Find hosts that can accommodate the VM
            available_hosts = []
            for host_idx, host in temp_hosts.items():
                # Check if the host can accommodate the VM (consider temp state)
                if host.can_accommodate(vm, current_time) and host_idx in solution.values():
                    available_hosts.append(host_idx)
            
            if not available_hosts:
                # Pick a host not used in solution.values() from temp_hosts
                used_hosts = set(solution.values())
                candidate_hosts = [h_idx for h_idx in temp_hosts.keys() if h_idx not in used_hosts]
                if candidate_hosts:
                    selected_host_idx = candidate_hosts[0]
                    solution[vm.index] = selected_host_idx
                    temp_hosts[selected_host_idx].allocate_vm(vm, current_time)
                else:
                    solution[vm.index] = -1  # Cannot allocate
                continue
            
            # Compute selection probabilities (based on dynamic heuristics)
            probabilities = self.calculate_probability(vm, available_hosts, temp_hosts)
            
            # Roulette-wheel selection
            rand = random.random()
            cumulative_prob = 0.0
            selected_host_idx = available_hosts[0]  
            
            for host_idx in available_hosts:
                cumulative_prob += probabilities[host_idx]
                if rand <= cumulative_prob:
                    selected_host_idx = host_idx
                    break
            
            solution[vm.index] = selected_host_idx
            
            # Update temporary host state
            temp_hosts[selected_host_idx].allocate_vm(vm, current_time)

        max_utils=[]
        for selected_host_idx in set(solution.values()):
            host=temp_hosts[selected_host_idx]
            util_series = host.time_utilization_cache.values()
            max_utils.append(max(util_series))            
        PAR= max(max_utils) / np.mean(max_utils)
        del temp_hosts
            
        return solution,PAR
        
    
    def evaluate_solution(self, solution, vms, PAR):
        """Evaluate the solution quality"""
        used_hosts = set()
        
        for vm in vms:
            host_idx = solution.get(vm.index, -1)
            if host_idx != -1:
                used_hosts.add(host_idx)
        return len(used_hosts)
    
    def update_pheromones(self, solutions, qualities):
        """Update pheromones"""
        for vm_idx in self.pheromone_matrix:
            for host_idx in self.pheromone_matrix[vm_idx]:
                self.pheromone_matrix[vm_idx][host_idx] *= (1 - self.rho)
        
        for solution, quality in zip(solutions, qualities):                
            pheromone_deposit = self.Q / quality
            
            for vm_idx, host_idx in solution.items():
                if host_idx != -1:
                    self.pheromone_matrix[vm_idx][host_idx] += pheromone_deposit
    
    def batch_schedule(self):
        if not self.pending_vms:
            return True
            
        vms_to_schedule = self.pending_vms.copy()
        self.pending_vms.clear()
        
        current_time = self.env.now
        self.initialize_pheromone_matrix(vms_to_schedule)
        
        best_solution = None
        best_quality = float('inf')
        no_improvement_count=0

        # ACO main loop
        for iteration in tqdm(range(self.max_iterations), desc="ACO迭代"):
            results = Parallel(n_jobs=10,backend="threading")(
                delayed(self._parallel_construct_solution)(
                    ant, vms_to_schedule, current_time
                ) for ant in range(self.num_ants)
            )
            solutions = []
            qualities = []
            for solution, quality, PAR in results:
                solutions.append(solution)
                qualities.append(quality)            
                if quality < best_quality:
                    best_quality = quality
                    best_solution = solution.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

            self.update_pheromones(solutions, qualities)
            if no_improvement_count >= self.patience:
                print(f"Early stop triggered: no improvement for {self.patience} consecutive iterations, stopping at iteration {iteration+1}")
                break
        
        if best_solution:
            for vm in vms_to_schedule:
                host_idx = best_solution.get(vm.index, -1)
                if host_idx != -1:
                    host = self.hosts[host_idx]
                    host.allocate_vm(vm, current_time)
                    self.mapping[vm.index] = host_idx
                    self.env.process(self.vm_lifecycle(vm, host))
                else:
                    self.mapping[vm.index] = -1
        else:
            for vm in vms_to_schedule:
                self.mapping[vm.index] = -1
        
        return True

    def _parallel_construct_solution(self, ant_id, vms_to_schedule, current_time):
        """Helper for parallel solution construction"""
        np.random.seed(ant_id + int(current_time))
        random.seed(ant_id + int(current_time))
        
        solution, PAR = self.construct_solution(vms_to_schedule, current_time)
        quality = self.evaluate_solution(solution, vms_to_schedule, PAR)
        
        return solution, quality, PAR


class SimPyGurobiScheduler(SimPyScheduler):
    """SimPy Gurobi scheduler"""
    
    def __init__(self, env: simpy.Environment, vms, hosts):
        super().__init__(env, vms, hosts)
        self.batch_size = 10000  # Number of VMs per batch schedule
        self.pending_vms = []
        self.processed_vms = 0
    
    def batch_schedule(self):
        if not self.pending_vms:
            return True
            
        vms_to_schedule = self.pending_vms.copy()
        self.pending_vms.clear()
        result = self.solve_gurobi_batch(vms_to_schedule)
        
        if result:
            mapping = result
            for vm in vms_to_schedule:
                if vm.index in mapping and mapping[vm.index] != -1:
                    host_idx = mapping[vm.index]
                    host = self.hosts[host_idx]
                    self.mapping[vm.index] = host_idx
                    self.env.process(self.vm_lifecycle(vm, host))
                else:
                    self.mapping[vm.index] = -1
            return True
        return False
    
    def solve_gurobi_batch(self, vms):
        num_vms = len(vms)
        num_hosts = len(self.hosts)
        
        if num_vms == 0:
            return  {}
            
        current_time = self.env.now
        
        m = gp.Model("VM_BinPacking_SimPy")
        m.setParam('OutputFlag', 0)
        m.setParam("TimeLimit", 3600)
        
        # Decision variables
        x = m.addVars(num_vms, num_hosts, vtype=GRB.BINARY, name="x")
        y = m.addVars(num_hosts, vtype=GRB.BINARY, name="y")
        
        # Objective: minimize the number of used hosts
        m.setObjective(gp.quicksum(y[j] for j in range(num_hosts)), GRB.MINIMIZE)
        
        # Constraint 1: each VM must be assigned to one host
        for i in range(num_vms):
            m.addConstr(gp.quicksum(x[i, j] for j in range(num_hosts)) == 1)
            
        # Constraint 2: time-varying capacity limits
        max_period = max(vm.period for vm in vms) if vms else 0
        freq_multiplier = float(int(vms[0].freq[:-1]) if vms[0].freq[:-1].isdigit() else 1 )
        # Add capacity constraints for each host and each time point
        for j,host in zip(range(num_hosts),self.hosts.values()):
            for t in range(max_period):
                future_time = current_time + t * freq_multiplier
                time_demand = gp.quicksum(
                    vms[i].get_utilization_at_time(t) * x[i, j]
                    for i in range(num_vms)
                    if t < vms[i].period  
                )
                existing_utilization = host.time_utilization_cache.get(future_time, 0.0)
                m.addConstr(
                    time_demand + existing_utilization <= host.capacity * y[j],
                    name=f"capacity_{j}_{t}"
                )
        
        m.optimize()
        if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
            if m.SolCount > 0:
                mapping = {}
                for i in range(num_vms):
                    for j,host in zip(range(num_hosts),self.hosts.values()):
                        if x[i, j].X > 0.5:
                            mapping[vms[i].index] = host.index
                            host.allocate_vm(vms[i], current_time)
                            break
                return  mapping
            else:
                print("Gurobi reached the time limit but did not find a feasible solution")
                return None            
        else:
            print(f"Gurobi optimization failed, status code: {m.status}")
            return None
    

class SchedulingSimulation:
    
    def __init__(self, scheduler_type: str = "FirstFit"):
        self.env = simpy.Environment()
        self.scheduler_type = scheduler_type
        self.scheduler = None
        self.vms = []
        self.hosts = {}
        
    def setup(self, vms, hosts,mapping:Optional[dict[int,int]] = None):
        """Set up the simulation environment"""
        self.vms = vms
        self.hosts = hosts
        
        # Create scheduler
        if self.scheduler_type == "FFD":
            self.scheduler = SimPyFirstFitDecreasing(self.env, self.vms, self.hosts)
        elif self.scheduler_type == "Gurobi":
            self.scheduler = SimPyGurobiScheduler(self.env, self.vms, self.hosts)
        elif self.scheduler_type == "BFD":
            self.scheduler = SimPyBestFitDecreasing(self.env, self.vms, self.hosts)
        elif self.scheduler_type == "ACO":
            self.scheduler = SimPyACOScheduler(self.env, self.vms, self.hosts)
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
            
        
    def generate_vm_arrivals(self):
        """Generate VM arrival events"""
        # Sort by VM start time
        sorted_vms = sorted(self.vms, key=lambda vm: vm.get_arrival_time())
        
        for vm in sorted_vms:
            arrival_time = vm.get_arrival_time()
            if arrival_time > self.env.now:
                yield self.env.timeout(arrival_time - self.env.now)
            yield self.env.process(self.scheduler.schedule_vm_arrival(vm))
    
    def run(self, until=None):
        """Run the simulation"""
        self.env.process(self.generate_vm_arrivals()).priority = -1
        self.env.process(self.scheduler.monitor_resources()).priority = 0

        if until is None:
            max_end_time = max(vm.get_finish_time() for vm in self.vms)
            until = max_end_time + 10  # Extra buffer time
            
        self.env.run(until=until)

        if not self.scheduler.mapping:
            return None
        for host_index in self.scheduler.mapping.values():
            if host_index == -1:  
                return None 
        
        return self.scheduler.mapping