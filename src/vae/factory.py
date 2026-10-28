from src.vae.VAE import VAE2D

def build_vae(cfg: dict) -> VAE2D:
    vcfg = cfg.get('vae', {})
    nrb_dec = vcfg.get('num_res_blocks_decoder', None)
    return VAE2D(in_channels=int(vcfg.get('in_channels', 1)), out_channels=int(vcfg.get('out_channels', 1)), base_channels=int(vcfg.get('base_channels', 64)), channel_mults=tuple(vcfg.get('channel_mults', [1, 2, 4, 4])), latent_channels=int(vcfg.get('latent_channels', 16)), num_res_blocks=int(vcfg.get('num_res_blocks', 2)), num_res_blocks_decoder=int(nrb_dec) if nrb_dec is not None else None, norm_groups=int(vcfg.get('norm_groups', 32)), dropout=float(vcfg.get('dropout', 0.0)), use_checkpoint=bool(vcfg.get('use_checkpoint', False)))

def load_vae_state_dict(vae: VAE2D, state: dict, key: str='vae', verbose: bool=True):
    sd = state.get(key) or state.get('vae') or state.get('ema') or state
    if any((k.startswith('_orig_mod.') for k in sd.keys())):
        sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
        if verbose:
            print('[vae load] stripped _orig_mod. prefix (torch.compile ckpt)')
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    if verbose:
        if missing or unexpected:
            print(f'[vae load] missing={len(missing)} unexpected={len(unexpected)}')
        else:
            print('[vae load] clean (all keys matched)')
    if len(missing) > 10 or len(unexpected) > 10:
        raise RuntimeError(f"VAE state-dict mismatch: missing={len(missing)} unexpected={len(unexpected)}. The model would silently run on random weights and produce noise. Check arch in cfg.vae matches the trained checkpoint, and that the right ckpt key ('vae' or 'ema') is selected via cfg.vae_ckpt_key.")
    return (missing, unexpected)