from src.utils.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# TODO: Eliminate registry and instantiate directly?

# Registry mapping callback names to constructors
CALLBACK_REGISTRY = {
    "EarlyStopping": EarlyStopping,
    "ReduceLROnPlateau": ReduceLROnPlateau,
    "ModelCheckpoint": ModelCheckpoint,
}

def process_callbacks(args: dict) -> dict[str, Callback]:
    """
     Dynamically map callback-specific arguments based on user-specified callbacks.

    Parameters:
        args (dict): Dictionary of all parsed arguments.

    Returns:
        dict: Dictionary of callback names mapped to their parameters.
    """
    callbacks = {}
    for name in args['callbacks']:
        if name not in CALLBACK_REGISTRY:
            raise ValueError(f"Callback '{name}' is not recognized. Available callbacks: {list(CALLBACK_REGISTRY.keys())}")
        
        if name == "ModelCheckpoint":
            callbacks[name] = CALLBACK_REGISTRY[name](
                monitor=args['ModelCheckpoint_monitor'], 
                save_best_only=args['ModelCheckpoint_save_best_only'], 
                mode=args['ModelCheckpoint_mode'],
                save_path=f"{args['save_dir']}/{args['model_name']}_best_model.pth",   
                verbose=args['ModelCheckpoint_verbose'])
        elif name == "EarlyStopping":
            callbacks[name] = CALLBACK_REGISTRY[name](
                monitor=args['EarlyStopping_monitor'], 
                patience=args['EarlyStopping_patience'], 
                min_delta=args['EarlyStopping_min_delta'],
                mode=args['EarlyStopping_mode'], 
                save_path=f"{args['save_dir']}/{args['model_name']}_best_model.pth", 
                verbose=args['EarlyStopping_verbose'])
        elif name == "ReduceLROnPlateau":
            callbacks[name] = CALLBACK_REGISTRY[name](
                monitor=args['ReduceLROnPlateau_monitor'], 
                factor=args['ReduceLROnPlateau_factor'], 
                patience=args['ReduceLROnPlateau_patience'], 
                min_delta=args['ReduceLROnPlateau_min_delta'], 
                mode=args['ReduceLROnPlateau_mode'], 
                min_lr=args['ReduceLROnPlateau_min_lr'], 
                verbose=args['ReduceLROnPlateau_verbose'])
    return callbacks