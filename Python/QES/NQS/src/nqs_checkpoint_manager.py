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
Date        : 15.11.2025
Description : Checkpoint Manager for NQS parameters and metadata.
---------------------------------------------------------
'''

import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np
import shutil

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
    
    Path Resolution:
    ----------------
    1. If `filename` is an absolute path -> use it directly
    2. If `filename` is relative -> resolve relative to `self.directory`
    3. If `filename` is None -> use default naming (checkpoint_{step}.h5 or Orbax step-based)
    
    Orbax Behavior:
    ---------------
    - Orbax ALWAYS uses step-based subdirectories: checkpoints/{step}/
    - Custom filenames are ignored for Orbax params (step is canonical)
    - Metadata is saved as JSON in checkpoints/metadata_{step}.json
    - To load a specific checkpoint, provide the step number
    
    HDF5 Fallback Behavior:
    -----------------------
    - Custom filenames are respected
    - Metadata is saved as HDF5 attributes
    - Can load by filename or step number
    """
    
    def __init__(self, 
                directory   : Union[str, Path], 
                use_orbax   : bool              = False,
                max_to_keep : int               = 3,
                logger      : Optional[Any]     = None):
        """
        Initialize the checkpoint manager.
        
        Parameters:
        -----------
        directory : Union[str, Path]
            Base directory for saving checkpoints.
        use_orbax : bool
            Whether to use Orbax (JAX-optimized) for saving. Falls back to HDF5 if unavailable.
        max_to_keep : int
            Maximum number of checkpoints to keep (Orbax only).
        logger : Optional[Any]
            Logger instance. If None, uses Python's logging module.
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
            import logging
            logging.info(msg)

    def _init_orbax(self, checkpoint_dir: Path = None):
        """Initialize Orbax CheckpointManager for a given directory."""
        # Options: keep max N checkpoints, create dir if missing
        target_dir          = checkpoint_dir or self._checkpoint_dir
        options             = ocp.CheckpointManagerOptions(max_to_keep=self.max_to_keep, create=True)
        
        # This prevents the "Unknown key" error by treating the whole network dict as one PyTree.
        self._orbax_manager = ocp.CheckpointManager(
                                target_dir, 
                                item_names  =   ('params',), 
                                options     =   options
                            )
            
    def _resolve_path(self, filename: Optional[str], step: Union[int, str], extension: str = "h5") -> Path:
        """
        Resolve the full path for a checkpoint file.
        
        Parameters:
        -----------
        filename : Optional[str]
            Custom filename or path. If absolute, used directly.
        step : Union[int, str]
            Step number or identifier (e.g., 'final').
        extension : str
            File extension to use if generating default name.
            
        Returns:
        --------
        Path : Resolved absolute path for the checkpoint file.
        """
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
        """
        Save parameters to disk.
        
        Parameters:
        -----------
        step : Union[int, str]
            Training step or epoch number, or a string identifier (e.g., 'final').
            For Orbax, this determines the subdirectory: checkpoints/{step}/
            
        params : Any
            PyTree of parameters to save. Can be a nested dict/list of arrays.
            
        metadata : Optional[Dict[str, Any]]
            Additional metadata to save alongside parameters (seed, network info, etc.)
            For Orbax: saved as JSON in checkpoints/metadata_{step}.json
            For HDF5: saved as file attributes
            
        filename : Optional[str]
            Custom filename for saving.
            - For HDF5: Full path or relative to self.directory
            - For Orbax: Ignored for params (step-based), but metadata JSON can use this
            
        Returns:
        --------
        Path : The path where the checkpoint was saved.

        Example:
        --------
            >>> checkpoint_manager.save(step=10, params=params, metadata={'seed': 42})
            >>> checkpoint_manager.save(step='final', params=params, filename='best_model.h5')
        """
        metadata            = metadata or {}    # Initialize metadata if None
        metadata['step']    = step              # Always record step in metadata
        
        if self.use_orbax:
            return self._save_orbax(step, params, metadata, **kwargs)
        else:
            return self._save_hdf5_wrapper(step, params, metadata, filename)

    def _save_orbax(self, step: Union[int, str], params: Any, metadata: Dict[str, Any], **kwargs) -> Path:
        """
        Save using Orbax backend.
        
        Orbax requires integer steps for its directory structure.
        String steps (e.g., 'final') are mapped to step 0 with a special key.
        """
        
        if isinstance(step, int):
            save_key = step
        elif isinstance(step, str):
            save_key = 0 
            metadata['_string_step'] = step
        else:
            raise TypeError(f"Step must be int or str, got {type(step)}")

        # Prepare Arguments
        # We wrap the params in a Composite arg matching the 'params' item defined in _init
        try:
            # We use StandardSave (works for PyTrees/Dicts) wrapped in Composite
            # This tells Orbax: "Put this data into the 'params' slot we defined earlier"
            save_args = ocp.args.Composite(params=ocp.args.StandardSave(params))
            
            # Ensure write is done before proceeding
            self._orbax_manager.save(save_key, args=save_args)
            self._orbax_manager.wait_until_finished()
            
        except Exception as e:
            self._log(f"CRITICAL: Orbax save failed: {e}", lvl=0, color='red')
            raise e
        
        # Save Metadata (JSON)
        self._save_metadata_json(step, metadata)
        
        self._log(f"Saved Orbax checkpoint for step {step} (key={save_key})", lvl=1, color='green')
        return self._checkpoint_dir / str(save_key)

    def _save_hdf5_wrapper(self, step: Union[int, str], params: Any, metadata: Dict[str, Any], filename: Optional[str]) -> Path:
        """Save using HDF5 backend with proper path resolution."""
        path = self._resolve_path(filename, step, extension="h5")
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if H5PY_AVAILABLE:
            self._save_hdf5(path, params, metadata)
        else:
            self._save_npz(path.with_suffix('.npz'), params)
            # Save metadata separately as JSON
            self._save_metadata_json(step, metadata, path.with_suffix('.json'))
            
        self._log(f"Saved HDF5 checkpoint to {path}", lvl=1, color='green')
        return path

    def load(self, 
            step        : Optional[Union[int, str]]     = None, 
            filename    : Optional[str]                 = None,
            target_par  : Optional[Any]                 = None) -> Any:
        """
        Load parameters from disk.
        
        Parameters:
        -----------
        step : Optional[Union[int, str]]
            Training step or epoch number to load.
            - If None and using Orbax: loads the latest checkpoint
            - If None and using HDF5: requires filename or finds latest checkpoint_*.h5
            
        filename : Optional[str]
            Custom filename to load (HDF5 only, ignored for Orbax).
            Can be absolute path or relative to self.directory.
            
        target_par : Optional[Any]
            Target parameter structure for Orbax restore (helps with shape inference).
            
        Returns:
        --------
        params : Any
            Loaded PyTree of parameters.
            
        Example:
        --------
            >>> params = checkpoint_manager.load(step=100)
            >>> params = checkpoint_manager.load(filename='best_model.h5')
            >>> params = checkpoint_manager.load()  # Latest checkpoint
        """
        if self.use_orbax:
            return self._load_orbax(step, target_par)
        else:
            return self._load_hdf5_wrapper(step, filename)

    def _load_orbax(self, step: Optional[Union[int, str]], target_par: Optional[Any]) -> Any:
        """Load using Orbax backend."""
        
        # Resolve Key
        if step is None:
            restore_key = self._orbax_manager.latest_step()
            if restore_key is None:
                raise FileNotFoundError("No Orbax checkpoints found.")
            self._log(f"Loading latest Orbax checkpoint (step={restore_key})", lvl=1)
        elif isinstance(step, str):
            restore_key     = 0
        else:
            restore_key     = step

        # Restore
        # We must mirror the save structure: Composite -> 'params' -> StandardRestore
        try:
            restore_args    = ocp.args.Composite(params=ocp.args.StandardRestore(target_par))
            restored        = self._orbax_manager.restore(restore_key, args=restore_args)
            
            # The result is a Composite object (or dict), we need to extract 'params'
            if hasattr(restored, 'params'):
                return restored.params
            elif isinstance(restored, dict) and 'params' in restored:
                return restored['params']
            else:
                return restored # Fallback if structure is different
                
        except Exception as e:
            self._log(f"Orbax load failed: {e}", lvl=0, color='red')
            raise e        

    def _load_hdf5_wrapper(self, step: Optional[Union[int, str]], filename: Optional[str]) -> Any:
        """Load using HDF5 backend."""
        if filename is not None:
            path = self._resolve_path(filename, step=0)  # step not used when filename provided
        elif step is not None:
            path = self._resolve_path(None, step, extension="h5")
        else:
            # Find latest checkpoint
            files = sorted(list(self.directory.glob("checkpoint_*.h5")))
            if not files:
                raise FileNotFoundError(f"No HDF5 checkpoints found in {self.directory}")
            path = files[-1]
            self._log(f"Loading latest HDF5 checkpoint: {path.name}", lvl=1)
            
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        if H5PY_AVAILABLE:
            return self._load_hdf5(path)
        else:
            return self._load_npz(path.with_suffix('.npz'))

    def load_metadata(self, step: Optional[Union[int, str]] = None, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Load metadata for a given checkpoint.
        
        Parameters:
        -----------
        step : Optional[Union[int, str]]
            Step number or identifier.
        filename : Optional[str]
            Custom metadata filename (HDF5 mode only).
            
        Returns:
        --------
        Dict[str, Any] : Loaded metadata dictionary.
        """
        if self.use_orbax:
            meta_path = self._checkpoint_dir / f"metadata_{step}.json"
        else:
            if filename:
                meta_path = self._resolve_path(filename, step=0).with_suffix('.json')
            else:
                meta_path = self.directory / f"metadata_{step}.json"
                
        if not meta_path.exists():
            self._log(f"Metadata file not found: {meta_path}", lvl=0, color='yellow')
            return {}
            
        with open(meta_path, 'r') as f:
            return json.load(f)

    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.
        
        Returns:
        --------
        list : List of available checkpoint steps/identifiers.
        """
        if self.use_orbax:
            return list(self._orbax_manager.all_steps())
        else:
            files = sorted(list(self.directory.glob("checkpoint_*.h5")))
            steps = []
            for f in files:
                # Extract step from filename: checkpoint_{step}.h5
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
    #! Internal Save/Load Helpers
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
        """Save parameters as NPZ file (fallback when HDF5 is not available)."""
        # Flatten the nested dict to save in NPZ format
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
        """
        Save metadata as JSON file.
        
        Parameters:
        -----------
        step : Union[int, str]
            Step identifier for naming the metadata file.
        metadata : Dict[str, Any]
            Metadata dictionary to save.
        path : Path, optional
            Custom path for metadata file. If None, uses default location.
        """
        if path is None:
            # Ensure checkpoint directory exists for Orbax
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            meta_path = self._checkpoint_dir / f"metadata_{step}.json"
        else:
            meta_path = Path(path)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert non-JSON-serializable types
        serializable_metadata = self._make_json_serializable(metadata)
        
        with open(meta_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
        
        self._log(f"Saved metadata to {meta_path}", lvl=2, log='debug')

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
            return str(obj)  # Convert objects to string representation
        else:
            return obj
            
    # ------------------------------------------------------
    
# ----------------------------------------------------------
# End of NQSCheckpointManager
# ----------------------------------------------------------