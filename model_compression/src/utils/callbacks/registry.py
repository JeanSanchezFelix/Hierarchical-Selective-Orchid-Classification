from typing import Dict, Any

from model_compression.src.utils.callbacks import (
    Callback,
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    LRScheduler
)

# Registry mapping callback names to constructor classes
CALLBACK_REGISTRY: Dict[str, type] = {
    'EarlyStopping': EarlyStopping,
    'ReduceLROnPlateau': ReduceLROnPlateau,
    'ModelCheckpoint': ModelCheckpoint,
    'LRScheduler': LRScheduler,
}

def process_callbacks(
    args: Dict[str, Any]
) -> Dict[str, Callback]:
    """
    Instantiate callbacks based on user configuration.

    Reads 'callbacks' list from args and creates each callback with the corresponding parameters from args.

    Args:
        args: Parsed configuration dict containing keys:
              - 'callbacks': List of callback names.
              - Optional per-callback parameters, e.g., 'EarlyStopping_patience'.

    Returns:
        A dict mapping callback names to instantiated Callback objects.

    Raises:
        ValueError: If a specified callback name is not registered.
    """
    callbacks: Dict[str, Callback] = {}
    save_dir = args.get('save_dir', '.')
    model_name = args.get('model_name', 'model')
    save_name = args.get('save_name', 'checkpoint')
    default_path = f"{save_dir}/{model_name}_{save_name}.pth"
    
    for cb_name in args.get('callbacks', []):
        if cb_name not in CALLBACK_REGISTRY:
            raise ValueError(
                f"Unknown callback '{cb_name}'. Available: {list(CALLBACK_REGISTRY.keys())}"
            )
        constructor = CALLBACK_REGISTRY[cb_name]
        if cb_name == 'ModelCheckpoint':
            cb = constructor(
                monitor=args.get('ModelCheckpoint_monitor', 'val_loss'),
                save_best_only=args.get('ModelCheckpoint_save_best_only', True),
                mode=args.get('ModelCheckpoint_mode', 'min'),
                save_path=default_path,
                verbose=args.get('ModelCheckpoint_verbose', False)
            )
        elif cb_name == 'EarlyStopping':
            cb = constructor(
                monitor=args.get('EarlyStopping_monitor', 'val_loss'),
                patience=args.get('EarlyStopping_patience', 5),
                min_delta=args.get('EarlyStopping_min_delta', 1e-4),
                mode=args.get('EarlyStopping_mode', 'min'),
                save_path=default_path,
                verbose=args.get('EarlyStopping_verbose', False)
            )
        elif cb_name == 'ReduceLROnPlateau':
            cb = constructor(
                monitor=args.get('ReduceLROnPlateau_monitor', 'val_loss'),
                factor=args.get('ReduceLROnPlateau_factor', 0.1),
                patience=args.get('ReduceLROnPlateau_patience', 10),
                min_delta=args.get('ReduceLROnPlateau_min_delta', 1e-4),
                mode=args.get('ReduceLROnPlateau_mode', 'min'),
                min_lr=args.get('ReduceLROnPlateau_min_lr', 1e-6),
                verbose=args.get('ReduceLROnPlateau_verbose', False)
            )
        elif cb_name == 'LRScheduler':
            # Expect scheduler instance passed in args['scheduler']
            scheduler = args.get('scheduler')
            if scheduler is None:
                raise ValueError("LRScheduler requires 'scheduler' instance in args.")
            cb = constructor(
                scheduler=scheduler,
                verbose=args.get('LRScheduler_verbose', False)
            )
        else:
            # Generic constructor
            cb = constructor()

        callbacks[cb_name] = cb

    return callbacks
