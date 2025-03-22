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
    Dynamically map callback-specific arguments based on user-specified callbacks,
    with default values for missing parameters.

    Parameters:
        args (dict): Dictionary of all parsed arguments.

    Returns:
        dict: Dictionary of callback names mapped to their parameters.
    """
    callbacks = {}
    for name in args['callbacks']:
        if name not in CALLBACK_REGISTRY:
            raise ValueError(f"Callback '{name}' is not recognized. Available callbacks: {list(CALLBACK_REGISTRY.keys())}")

        # Common save path for callbacks that need it
        save_path = f"{args.get('save_dir', './')}/{args.get('model_name', 'model')}_{args.get('save_name', 'checkpoint')}.pth"

        if name == "ModelCheckpoint":
            callbacks[name] = CALLBACK_REGISTRY[name](
                monitor=args.get('ModelCheckpoint_monitor', 'val_loss'),
                save_best_only=args.get('ModelCheckpoint_save_best_only', True),
                mode=args.get('ModelCheckpoint_mode', 'min'),
                save_path=save_path,
                verbose=args.get('ModelCheckpoint_verbose', False)
            )
        elif name == "EarlyStopping":
            callbacks[name] = CALLBACK_REGISTRY[name](
                monitor=args.get('EarlyStopping_monitor', 'val_loss'),
                patience=args.get('EarlyStopping_patience', 5),
                min_delta=args.get('EarlyStopping_min_delta', 1e-4),
                mode=args.get('EarlyStopping_mode', 'min'),
                save_path=save_path,
                verbose=args.get('EarlyStopping_verbose', False)
            )
        elif name == "ReduceLROnPlateau":
            callbacks[name] = CALLBACK_REGISTRY[name](
                monitor=args.get('ReduceLROnPlateau_monitor', 'val_loss'),
                factor=args.get('ReduceLROnPlateau_factor', 0.1),
                patience=args.get('ReduceLROnPlateau_patience', 10),
                min_delta=args.get('ReduceLROnPlateau_min_delta', 1e-4),
                mode=args.get('ReduceLROnPlateau_mode', 'min'),
                min_lr=args.get('ReduceLROnPlateau_min_lr', 1e-6),
                verbose=args.get('ReduceLROnPlateau_verbose', False)
            )

    return callbacks
