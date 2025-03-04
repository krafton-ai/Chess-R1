# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""
import dataclasses
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Union, Dict, Any


class Tracking(object):
    supported_backend = ['wandb', 'mlflow', 'console', 'tensorboard']

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = 'console', config=None, capture_console=True):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == 'tracking':
                import warnings
                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning)
            else:
                assert backend in self.supported_backend, f'{backend} is not supported'

        self.logger = {}

        if 'tracking' in default_backend or 'wandb' in default_backend:
            import wandb
            import os
            WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
            if WANDB_API_KEY:
                wandb.login(key=WANDB_API_KEY)
            wandb.init(project=project_name, name=experiment_name, config=config)
            self.logger['wandb'] = wandb

        if 'mlflow' in default_backend:
            import mlflow
            mlflow.start_run(run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger['mlflow'] = _MlflowLoggingAdapter()

        if 'tensorboard' in default_backend:
            try:
                from torch.utils.tensorboard import SummaryWriter
                import os
                
                # Create log directory using project and experiment names
                log_dir = os.path.join('runs', project_name, experiment_name)
                os.makedirs(log_dir, exist_ok=True)
                
                writer = SummaryWriter(log_dir=log_dir)
                
                # Log config parameters if provided
                if config:
                    flat_config = _flatten_dict(_transform_params_to_json_serializable(config, convert_list_to_dict=True), sep='/')
                    writer.add_text('config', str(flat_config))
                
                self.logger['tensorboard'] = _TensorboardLoggingAdapter(writer)
                
                # Setup console capture to text file if requested
                if capture_console:
                    self._setup_console_capture(log_dir)
                    
            except ImportError:
                import warnings
                warnings.warn("TensorBoard not found. Please install with `pip install tensorboard`.", ImportWarning)
                # Fall back to console logging if tensorboard isn't available
                if 'console' not in default_backend:
                    from verl.utils.logger.aggregate_logger import LocalLogger
                    self.console_logger = LocalLogger(print_to_console=True)
                    self.logger['console'] = self.console_logger
                    warnings.warn("Falling back to console logging.", ImportWarning)

        if 'console' in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger['console'] = self.console_logger

    def _setup_console_capture(self, log_dir):
        """Set up capturing of console output to a text file"""
        import sys
        import os
        
        # Set up console log file
        console_log_path = os.path.join(log_dir, "console_logs.txt")
        self.console_log_file = open(console_log_path, "w")
        
        # Store the original stdout and stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create a custom stdout/stderr capture class
        class StdoutCapture:
            def __init__(self, original, log_file, stream_name="stdout"):
                self.original = original
                self.log_file = log_file
                self.stream_name = stream_name
            
            def write(self, text):
                self.original.write(text)
                # Write to the log file with stream identifier
                if text:
                    prefix = f"[{self.stream_name}] " if text.strip() else ""
                    self.log_file.write(prefix + text)
                    self.log_file.flush()  # Ensure it's written immediately
            
            def flush(self):
                self.original.flush()
                self.log_file.flush()
        
        # Replace stdout and stderr with our capturing versions
        sys.stdout = StdoutCapture(sys.stdout, self.console_log_file, "stdout")
        sys.stderr = StdoutCapture(sys.stderr, self.console_log_file, "stderr")

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)
                
    def __del__(self):
        """Clean up resources when the object is destroyed"""
        # Restore original stdout and stderr if they were replaced
        if hasattr(self, 'original_stdout') and hasattr(self, 'original_stderr'):
            import sys
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            
        # Close the console log file if it was opened
        if hasattr(self, 'console_log_file'):
            self.console_log_file.close()


class _MlflowLoggingAdapter:

    def log(self, data, step):
        import mlflow
        mlflow.log_metrics(metrics=data, step=step)


class _TensorboardLoggingAdapter:
    
    def __init__(self, writer):
        self.writer = writer
        
    def log(self, data, step):
        # Handle scalar metrics
        for key, value in data.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
            elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                # For lists of scalars, log as histogram
                self.writer.add_histogram(key, value, step)
            # Could expand with more types if needed (images, audio, etc.)


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep='/')


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {'list_len': len(x)} | {f'{i}': _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd
    ans = pd.json_normalize(raw, sep=sep).to_dict(orient='records')[0]
    assert isinstance(ans, dict)
    return ans