import warnings
from importlib import import_module

__all__ = []


def _optional_import(module_name, symbol_names):
    try:
        module = import_module(f'{__name__}.{module_name}')
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        warnings.warn(f'Skipping optional backbone module `{module_name}`: {exc}')
        return

    for symbol_name in symbol_names:
        globals()[symbol_name] = getattr(module, symbol_name)
        __all__.append(symbol_name)


_optional_import('pointnet', ['PointNetEncoder'])
_optional_import('pointnetv2', ['PointNet2Encoder', 'PointNet2Decoder', 'PointNetFPModule'])
_optional_import('pointnext', ['PointNextEncoder', 'PointNextDecoder'])
_optional_import('dgcnn', ['DGCNN'])
_optional_import('deepgcn', ['DeepGCN'])
_optional_import('pointmlp', ['PointMLPEncoder', 'PointMLP'])
_optional_import('pointtransformer', ['PTSeg'])
_optional_import('pointvit', ['PointViT', 'PointViTDecoder'])
_optional_import('pointvit_inv', ['InvPointViT'])
_optional_import('pct', ['Pct'])
_optional_import('curvenet', ['CurveNet'])
_optional_import('simpleview', ['MVModel'])
