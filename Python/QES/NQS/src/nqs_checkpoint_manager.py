'''
The NQS Checkpoint Manager
Manages saving and loading of NQS parameters and metadata.
Supports HDF5 (generic) and Orbax (JAX-optimized).

Design Principles:
- Custom filename/path takes precedence when provided
- Orbax always uses step-based subdirectories within checkpoints/
- Metadata is saved as JSON alongside all checkpoints
- Seed and other info belong in metadata, not directory names

---------------------------------------------------------
File        : NQS/src/nqs_checkpoint_manager.py
Author      : Maksymilian Kliczkowski
Email       : maxgrom97@gmail.com
Date        : 08.12.2025 (Fixed)
Description : Checkpoint Manager for NQS parameters and metadata.
---------------------------------------------------------
'''

import  os
import  json
import  shutil
import  numpy as np
import  warnings
from    pathlib import Path
from    typing import Optional, Dict, Any, Union

# Try importing HDF5 and JAX/Orbax
try:
    import h5py
    H5PY_AVAILABLE  = True
except ImportError:
    H5PY_AVAILABLE  = False

try:
    import orbax.checkpoint as ocp
    ORBAX_AVAILABLE = True
except ImportError:
    ORBAX_AVAILABLE = False

class NQSCheckpointManager:
    """
    Manages saving and loading of NQS parameters and metadata.
    Supports HDF5 (generic) and Orbax (JAX-optimized).
    """
    
    def __init__(self, 
                directory   : Union[str, Path], 
                use_orbax   : bool              = False,
                max_to_keep : int               = 3,
                logger      : Optional[Any]     = None):
        """
        Initialize the checkpoint manager.
        """
        self.directory          = Path(directory)
        self.use_orbax          = use_orbax and ORBAX_AVAILABLE
        self.max_to_keep        = max_to_keep
        self._logger            = logger
        
        # Ensure directory exists
        self.directory.mkdir(parents=True, exist_ok=True)
        self._orbax_manager     = None
        self._checkpoint_dir    = self.directory / "checkpoints"
        
        if self.use_orbax:
            self._log("Initializing Orbax CheckpointManager.", lvl=1)
            self._log(f"Checkpoints (max {self.max_to_keep}) will be saved to: {self._checkpoint_dir}", lvl=2)
            self._init_orbax()

    # ------------------------------------------------------
    #! Internal Helpers
    # ------------------------------------------------------

    def _log(self, msg: str, lvl: int = 0, color: str = None, log = 'info'):
        """Unified logging helper."""
        if self._logger is not None:
            try:
                if log == 'debug':
                    self._logger.debug(msg, lvl=lvl, color=color)
                else:
                    self._logger.info(msg, lvl=lvl, color=color)
            except TypeError:
                self._logger.info(msg)
        else:
            print(f"[NQSCheckpointManager] {msg}")

    # ------------------------------------------------------

    def _init_orbax(self, checkpoint_dir: Path = None):
        """Initialize Orbax CheckpointManager."""
        # Standard options for rotational keeping of checkpoints
        target_dir          = checkpoint_dir or self._checkpoint_dir
        options             = ocp.CheckpointManagerOptions(
                                max_to_keep                 = self.max_to_keep, 
                                create                      = True, 
                                enable_async_checkpointing  = False,
                                step_prefix                 = None
                            )
                
        # Define the checkpointer (Required!)
        # Try to use the new API if possible to avoid deprecation warnings
        try:
            # New API (>=0.5.0) - Checkpointer is provided via save_args/restore_args
            self._orbax_manager = ocp.CheckpointManager(
                                    target_dir, 
                                    options=options
                                )
            self._legacy_api = False
        except (TypeError, ValueError):
            # Legacy API - Checkpointer provided in constructor
            checkpointer        = ocp.PyTreeCheckpointer()
            self._orbax_manager = ocp.CheckpointManager(
                                    target_dir, 
                                    checkpointer, 
                                    options=options
                                )
            self._legacy_api = True
        
        self._log(f"Initialized Orbax Manager (Legacy API: {self._legacy_api})", lvl=2)
            
    def _resolve_path(self, filename: Optional[str], step: Union[int, str], extension: str = "h5") -> Path:
        """Resolve the full path for a checkpoint file."""
        if filename is not None:
            path = Path(filename)
            if path.is_absolute():
                return path
            else:
                return self.directory / filename
        else:
            return self.directory / f"checkpoint_{step}.{extension}"

    # ------------------------------------------------------
    #! Save and Load Methods
    # ------------------------------------------------------
        
    def save(self, 
            step            : Union[int, str],
            params          : Any, 
            metadata        : Optional[Dict[str, Any]]  = None, 
            filename        : Optional[str]             = None,
            **kwargs) -> Path:
        """Save parameters to disk."""
        metadata            = metadata or {}    # Initialize metadata if None
        metadata['step']    = step              # Always record step in metadata
        
        if self.use_orbax:
            return self._save_orbax(step, params, metadata, **kwargs)
        else:
            return self._save_hdf5_wrapper(step, params, metadata, filename)

    def _save_orbax(self, step: Union[int, str], params: Any, metadata: Dict[str, Any], force: bool = True, **kwargs) -> Path:
        """
        Save using Orbax backend. Always saves a single PyTree using PyTreeCheckpointer.
        This ensures the directory structure is always .../checkpoints/<step>/default/...
        """
        if isinstance(step, str) and step.isdigit():
            step = int(step)
        save_key = step if isinstance(step, int) else kwargs.get('current_epoch', 999999)
        try:
            if os.path.exists(self._checkpoint_dir / str(save_key)):
                if force:
                    self._log(f"Overwriting existing Orbax checkpoint for step {save_key}", lvl=1, color='yellow')
                    shutil.rmtree(self._checkpoint_dir / str(save_key), ignore_errors=True)
                else:
                    raise FileExistsError(f"Checkpoint for step {save_key} already exists.")
            
            # Always use PyTreeCheckpointer for a single PyTree
            save_args = ocp.args.PyTreeSave(params)
            self._orbax_manager.save(save_key, args=save_args, force=force)
            
        except Exception as e:
            self._log(f"CRITICAL: Orbax save failed: {e}", lvl=0, color='red')
            shutil.rmtree(self._checkpoint_dir / str(save_key), ignore_errors=True)
            raise e
        self._save_metadata_json(save_key, metadata)
        self._log(f"Saved Orbax checkpoint for step {save_key}", lvl=1, color='green')
        return self._checkpoint_dir / str(save_key)

    def _save_hdf5_wrapper(self, step: Union[int, str], params: Any, metadata: Dict[str, Any], filename: Optional[str]) -> Path:
        """Save using HDF5 backend."""
        path = self._resolve_path(filename, step, extension="h5")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if H5PY_AVAILABLE:
            self._save_hdf5(path, params, metadata)
        else:
            self._save_npz(path.with_suffix('.npz'), params)
            self._save_metadata_json(step, metadata, path.with_suffix('.json'))
            
        self._log(f"Saved HDF5 checkpoint to {path}", lvl=1, color='green')
        return path

    # ------------------------------------------------------

    def load(self, 
            step        : Optional[Union[int, str]]     = None, 
            filename    : Optional[str]                 = None,
            target_par  : Optional[Any]                 = None) -> Any:
        """Load parameters from disk."""
        if self.use_orbax:
            return self._load_orbax(step, filename, target_par)
        else:
            return self._load_hdf5_wrapper(step, filename)

    def _load_orbax(self, step: Optional[Union[int, str]], filename: Optional[str], target_par: Optional[Any]) -> Any:
        """
        Robust Load: Always loads a single PyTree using PyTreeCheckpointer.
        1. If 'filename' (absolute path) is provided, loads directly via PyTreeCheckpointer.
        2. If 'step' is provided, uses Manager to load from default directory.
        This matches the save logic and ensures the directory structure is always .../checkpoints/<step>/default/...
        """
        # Case 1: Direct filename load
        if filename is not None:
            path_obj = Path(filename)
            if not path_obj.exists():
                raise FileNotFoundError(f"Path not found: {path_obj}")
            
            # Direct load using Checkpointer
            ckptr = ocp.PyTreeCheckpointer()
            try:
                if target_par is not None:
                    return ckptr.restore(path_obj, item=target_par)
                return ckptr.restore(path_obj)
            except Exception as e:
                self._log(f"Direct load failed: {e}", lvl=0, color='red')
                raise e

        # Case 2: Load via Manager (step based)
        if step is None:
            step = self._orbax_manager.latest_step()
            if step is None:
                raise FileNotFoundError("No checkpoints found in manager.")
        
        if step == 'final':
            step = 999999
            
        step_int = int(step) if isinstance(step, (int, str)) and str(step).isdigit() else 0
        self._log(f"Loading Orbax step {step_int}...", lvl=1)
        
        try:
            # Try loading using arguments (New/Standard API)
            restore_args = ocp.args.PyTreeRestore(item=target_par)
            restored_params = self._orbax_manager.restore(step_int, args=restore_args)
            return restored_params
        except Exception as e:
            self._log(f"Manager load failed with args: {e}. Attempting fallback...", lvl=1, color='yellow')
            
            # Fallback 1: Legacy restore (if manager was initialized with checkpointer)
            if getattr(self, '_legacy_api', False):
                try:
                    # In legacy, sometimes restore returns the item directly without args
                    return self._orbax_manager.restore(step_int, items=target_par)
                except Exception as e2:
                    self._log(f"Legacy fallback 1 failed: {e2}", lvl=1)
            
            # Fallback 2: Manual path construction (safest resort)
            # Try to guess the path: <dir>/<step>/default or <dir>/<step>
            potential_paths = [
                self._checkpoint_dir / str(step_int) / "default",
                self._checkpoint_dir / str(step_int)
            ]
            
            for p in potential_paths:
                if p.exists():
                    self._log(f"Attempting manual load from {p}", lvl=1)
                    try:
                        ckptr = ocp.PyTreeCheckpointer()
                        if target_par is not None:
                            return ckptr.restore(p, item=target_par)
                        return ckptr.restore(p)
                    except Exception as e3:
                        self._log(f"Manual load failed from {p}: {e3}", lvl=1)
            
            # If all fail
            raise e

    def _load_hdf5_wrapper(self, step: Optional[Union[int, str]], filename: Optional[str]) -> Any:
        """Load using HDF5 backend."""
        if filename is not None:
            path    = self._resolve_path(filename, step=0) 
        elif step is not None:
            path    = self._resolve_path(None, step, extension="h5")
        else:
            files   = sorted(list(self.directory.glob("checkpoint_*.h5")))
            if not files: raise FileNotFoundError(f"No HDF5 checkpoints found in {self.directory}")
            path    = files[-1]
            self._log(f"Loading latest HDF5 checkpoint: {path.name}", lvl=1)
            
        if not path.exists(): raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        if H5PY_AVAILABLE:
            return self._load_hdf5(path)
        else:
            return self._load_npz(path.with_suffix('.npz'))

    def load_metadata(self, step: Optional[Union[int, str]] = None, filename: Optional[str] = None) -> Dict[str, Any]:
        """Load metadata for a given checkpoint."""
        if self.use_orbax:
            if step == 'final': step = 999999
            meta_path       = self._checkpoint_dir / f"metadata_{step}.json"
        else:
            if filename:
                meta_path   = self._resolve_path(filename, step=0).with_suffix('.json')
            else:
                meta_path   = self.directory / f"metadata_{step}.json"
                
        if not meta_path.exists():
            return {}
            
        with open(meta_path, 'r') as f:
            return json.load(f)

    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        if self.use_orbax:
            return list(self._orbax_manager.all_steps())
        else:
            files = sorted(list(self.directory.glob("checkpoint_*.h5")))
            steps = []
            for f in files:
                try:
                    step_str = f.stem.replace("checkpoint_", "")
                    steps.append(int(step_str) if step_str.isdigit() else step_str)
                except ValueError:
                    steps.append(f.stem)
            return steps

    @property
    def latest_step(self) -> Optional[Union[int, str]]:
        """Get the latest checkpoint step."""
        if self.use_orbax:
            return self._orbax_manager.latest_step()
        else:
            checkpoints = self.list_checkpoints()
            return checkpoints[-1] if checkpoints else None

    # ------------------------------------------------------
    #! Internal Save/Load Helpers (HDF5/NPZ/JSON)
    # ------------------------------------------------------

    def _save_hdf5(self, path, params, metadata):
        with h5py.File(path, 'w') as f:
            self._write_group(f, params)
            if metadata:
                for k, v in metadata.items():
                    try:
                        f.attrs[k] = v
                    except TypeError:
                        f.attrs[k] = str(v)

    def _write_group(self, h5_group, py_dict):
        """Recursively write dictionary to HDF5 group."""
        for k, v in py_dict.items():
            if isinstance(v, dict):
                sub_group = h5_group.create_group(str(k))
                self._write_group(sub_group, v)
            else:
                h5_group.create_dataset(str(k), data=np.array(v))

    def _load_hdf5(self, path):
        with h5py.File(path, 'r') as f:
            return self._read_tree(f)

    def _read_tree(self, node):
        """Recursively read HDF5 group into dictionary."""
        out = {}
        for k, v in node.items():
            if isinstance(v, h5py.Group):
                out[k] = self._read_tree(v)
            else:
                out[k] = v[()] # Read dataset to numpy
        return out

    def _save_npz(self, path: Path, params: Any):
        """Save parameters as NPZ file."""
        flat_dict = self._flatten_dict(params)
        np.savez(path, **flat_dict)

    def _load_npz(self, path: Path) -> dict:
        """Load parameters from NPZ file."""
        data = np.load(path, allow_pickle=True)
        return dict(data)

    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '/') -> dict:
        """Flatten a nested dictionary for NPZ storage."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _save_metadata_json(self, step: Union[int, str], metadata: Dict[str, Any], path: Path = None):
        """Save metadata as JSON file."""
        if path is None:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            meta_path = self._checkpoint_dir / f"metadata_{step}.json"
        else:
            meta_path = Path(path)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
        
        serializable_metadata = self._make_json_serializable(metadata)
        
        with open(meta_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert non-JSON-serializable objects to serializable forms."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, complex):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        elif hasattr(obj, '__dict__'):
            return str(obj) 
        else:
            return obj

# ------------------------------------------------------
#! EOF
# ------------------------------------------------------