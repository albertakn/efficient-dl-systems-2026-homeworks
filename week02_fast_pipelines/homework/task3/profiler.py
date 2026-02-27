import json
import time
import torch
import os
from collections import defaultdict
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class ProfilerAction(Enum):
    SLEEP: int = 0
    WARMUP: int = 1
    ACTIVE: int = 2

class CustomSchedule(BaseModel):
    skip_first: int = Field(default=0)
    warmup: int = Field(default=0)
    active: int = Field(default=1)


class Profile:
    def __init__(self, model, name="model", schedule: Optional[CustomSchedule] = None):
        self.model = model
        self.name_map = self._build_name_map(model, name)
        
        if schedule is None:
            self.schedule = CustomSchedule()
        else:
            self.schedule = schedule
        
        self.events = []
        self.hooks = []
        self.intime_events = {}
        self.current_step = 0
    
    def _build_name_map(self, model, name="model"):
        name_map = {}
        for full_name, module in model.named_modules():
            if full_name == "":
                full_name = name

            if self._is_leaf(module):
                name_map[module] = module.__class__.__name__
            else:
                name_map[module] = f"{full_name}: {module.__class__.__name__}"

        return name_map

    def _is_leaf(self, module):
        return len(list(module.children())) == 0

    def _get_current_action(self) -> ProfilerAction:
        step = self.current_step

        if step < self.schedule.skip_first:
            return ProfilerAction.SLEEP
        
        if step < self.schedule.skip_first + self.schedule.warmup:
            return ProfilerAction.WARMUP
        
        if step < self.schedule.skip_first + self.schedule.warmup + self.schedule.active:
            return ProfilerAction.ACTIVE
        
        return ProfilerAction.SLEEP

    def _record_event(self, module, phase_name, start, end):
        duration_us = (end - start) * 1e6
        start_us = start * 1e6
        
        self.events.append({
            "name": self.name_map[module],
            "cat": phase_name, # Категория (forward/backward)
            "ph": "X",         # Phase: Complete Event (важно для визуализации!)
            "ts": start_us,    # Time stamp
            "dur": duration_us,# Duration
            "pid": os.getpid(),# Process ID (чтобы группировать строки)
            "tid": 0 if phase_name == "forward" else 1 # Разносим FWD и BWD по разным потокам визуально
        })

    def _forward_pre_hook(self, module, inputs):
        action = self._get_current_action()
        if action != ProfilerAction.SLEEP:
            torch.cuda.synchronize()
            self.intime_events[module] = time.perf_counter()

    def _forward_post_hook(self, module, inputs, outputs):
        action = self._get_current_action()
        if action != ProfilerAction.SLEEP:
            start_time = self.intime_events.pop(module)
        
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            if action == ProfilerAction.ACTIVE:
                self._record_event(module, "forward", start_time, end_time)

    def _backward_pre_hook(self, module, grad_output):
        action = self._get_current_action()
        if action != ProfilerAction.SLEEP:
            torch.cuda.synchronize()
            self.intime_events[module] = time.perf_counter()

    def _backward_post_hook(self, module, grad_input, grad_output):
        action = self._get_current_action()
        if action != ProfilerAction.SLEEP:
            start_time = self.intime_events.pop(module)
        
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            if action == ProfilerAction.ACTIVE:
                self._record_event(module, "backward", start_time, end_time)

    def __enter__(self):
        for module in self.model.modules():
            if self._is_leaf(module):
                forward_pre_hook = module.register_forward_pre_hook(self._forward_pre_hook)
                forward_post_hook = module.register_forward_hook(self._forward_post_hook)
                backward_pre_hook = module.register_full_backward_pre_hook(self._backward_pre_hook)
                backward_post_hook = module.register_full_backward_hook(self._backward_post_hook)
                self.hooks.extend(
                    [
                        forward_pre_hook,
                        forward_post_hook,
                        backward_pre_hook,
                        backward_post_hook,
                    ]
                )
        return self
 
    def __exit__(self, type, value, traceback):
        for hook in self.hooks:
            hook.remove()

        self.hooks.clear()
        self.intime_events.clear()

    def step(self):
        self.current_step += 1

    def summary(self):
        print("Summary:")
        for event in self.events:
            print(event)

    def to_perfetto(self, path="trace.json"):
        trace_data = {
            "traceEvents": self.events,
            "displayTimeUnit": "ms"
        }
        
        with open(path, 'w') as f:
            json.dump(trace_data, f)
        print(f"Trace saved to {path}")
